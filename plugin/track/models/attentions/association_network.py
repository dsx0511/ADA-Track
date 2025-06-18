# ------------------------------------------------------------------------
# ADA-Track
# Copyright (c) 2024 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from 3DMOTFormer (https://github.com/dsx0511/3DMOTFormer)
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Pytorch Geometric (https://github.com/pyg-team/pytorch_geometric)
# Copyright (c) 2023 PyG Team <team@pyg.org>. All Rights Reserved.
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
import math
import copy
from typing import Optional, Tuple, Union

from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils import Registry
from mmcv.cnn.bricks.transformer import build_feedforward_network

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn import TransformerConv
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax

ASSO_NET = Registry('association_network')

class BipartiteData(Data):
    def __init__(self, size_s=None, size_t=None,
                       edge_index=None, edge_index_mp=None,
                       edge_attr=None):
        super().__init__()
        self.edge_index = edge_index
        self.edge_index_mp = edge_index_mp
        self.edge_attr = edge_attr
        self.size_s = size_s
        self.size_t = size_t

    def __inc__(self, key, value, *args, **kwargs):
        if 'edge_index' in key:
            return torch.tensor([[self.size_s], [self.size_t]])
        else:
            return super().__inc__(key, value, *args, **kwargs)
        
class TransformerConv(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    multi-head dot product attention:

    .. math::
        \alpha_{i,j} = \textrm{softmax} \left(
        \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
        {\sqrt{d}} \right)

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        beta (bool, optional): If set, will combine aggregation and
            skip information via

            .. math::
                \mathbf{x}^{\prime}_i = \beta_i \mathbf{W}_1 \mathbf{x}_i +
                (1 - \beta_i) \underbrace{\left(\sum_{j \in \mathcal{N}(i)}
                \alpha_{i,j} \mathbf{W}_2 \vec{x}_j \right)}_{=\mathbf{m}_i}

            with :math:`\beta_i = \textrm{sigmoid}(\mathbf{w}_5^{\top}
            [ \mathbf{W}_1 \mathbf{x}_i, \mathbf{m}_i, \mathbf{W}_1
            \mathbf{x}_i - \mathbf{m}_i ])` (default: :obj:`False`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). Edge features are added to the keys after
            linear transformation, that is, prior to computing the
            attention dot product. They are also added to final values
            after the same linear transformation. The model is:

            .. math::
                \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
                \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \left(
                \mathbf{W}_2 \mathbf{x}_{j} + \mathbf{W}_6 \mathbf{e}_{ij}
                \right),

            where the attention coefficients :math:`\alpha_{i,j}` are now
            computed via:

            .. math::
                \alpha_{i,j} = \textrm{softmax} \left(
                \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top}
                (\mathbf{W}_4\mathbf{x}_j + \mathbf{W}_6 \mathbf{e}_{ij})}
                {\sqrt{d}} \right)

            (default :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output and the
            option  :attr:`beta` is set to :obj:`False`. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = nn.Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j += edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class EdgeAugmentTransformerConv(MessagePassing):
    r"""
    From edge augment graph transformer paper: https://arxiv.org/pdf/2108.03348.pdf
    """
    _alpha: OptTensor
    _edge_feat: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_dim: int,
        heads: int = 1,
        gate: bool = False,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(EdgeAugmentTransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.gate = gate
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None
        self._edge_feat = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)

        self.lin_edge_attn = nn.Linear(edge_dim, heads)
        self.lin_edge = nn.Linear(heads, edge_dim)
        if self.gate:
            self.lin_edge_gate = nn.Linear(edge_dim, heads * out_channels, bias=False)

        if concat:
            self.lin_skip = nn.Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge_attn.reset_parameters()
        self.lin_edge.reset_parameters()
        if self.gate:
            self.lin_edge_gate.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None,
                return_edge_features=False):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        edge_attn = self.lin_edge_attn(edge_attr)
        edge_gate = None
        if self.gate:
            edge_gate = self.lin_edge_gate(edge_attr).view(-1, H, C)
            edge_gate = torch.sigmoid(edge_gate)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attn=edge_attn, edge_gate=edge_gate, size=None)

        alpha = self._alpha
        self._alpha = None
        edge_feat = self._edge_feat
        self._edge_feat = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r
        
        edge_feat = self.lin_edge(edge_feat)
        
        out = (out, edge_feat)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attn: OptTensor, edge_gate: OptTensor, 
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        # if self.lin_edge is not None:
        #     assert edge_attr is not None
        #     edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
        #                                               self.out_channels)
        #     key_j += edge_attr
        
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha += edge_attn
        self._edge_feat = alpha

        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if edge_gate is not None:
            value_j *= edge_gate

        out = value_j
        # if edge_attr is not None:
        #     out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
    
class EdgeAugmentedDecoderLayer(BaseModule):

    def __init__(self,
                 d_model,
                 d_ffn,
                 heads=1,
                 dropout=0.0,
                 norm_first=False,
                 apply_self_attn=True,
                 cross_attn_value_gate=False):
        super().__init__()

        self.d_model = d_model
        self.d_ffn = d_ffn
        self.heads = heads
        self.dropout_p = dropout
        self.norm_first = norm_first
        self.apply_self_attn = apply_self_attn
        self.cross_attn_value_gate = cross_attn_value_gate

        self.head_channels = self.d_model // self.heads
        assert self.head_channels * self.heads == self.d_model, \
            'd_model must be dividable by heads'

        if self.apply_self_attn:
            self.self_attn = TransformerConv(self.d_model, self.head_channels,
                                            heads=self.heads, dropout=self.dropout_p)
            self.norm1 = nn.LayerNorm(self.d_model)
            self.dropout1 = nn.Dropout(self.dropout_p)

        self.edge_dim_cross_attn = self.d_model
        self.cross_attn = EdgeAugmentTransformerConv(
            self.d_model, self.head_channels, gate=self.cross_attn_value_gate,
            heads=self.heads, dropout=self.dropout_p, edge_dim=self.edge_dim_cross_attn)

        self.lin1 = nn.Linear(self.d_model, self.d_ffn)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lin2 = nn.Linear(self.d_ffn, self.d_model)
        self.activation = nn.ReLU()

        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.dropout3 = nn.Dropout(self.dropout_p)

        self.lin_e1 = nn.Linear(self.d_model, self.d_ffn)
        self.dropout_e = nn.Dropout(self.dropout_p)
        self.lin_e2 = nn.Linear(self.d_ffn, self.d_model)
        self.activation_e = nn.ReLU()
        
        self.norm_e1 = nn.LayerNorm(self.d_model)
        self.norm_e2 = nn.LayerNorm(self.d_model)
        self.dropout_e1 = nn.Dropout(self.dropout_p)
        self.dropout_e2 = nn.Dropout(self.dropout_p)
    
    def forward(self, src, tgt, edge_index_cross, edge_index_tgt, 
                edge_attr_cra=None):
        x = tgt

        if self.norm_first:
            if self.apply_self_attn:
                x = x + self._sa_block(self.norm1(x), edge_index_tgt)
            x_delta, edge_attr_cra_delta, attn_weight = self._cra_block(
                src, self.norm2(x), edge_index_cross, self.norm_e1(edge_attr_cra))

            x = x + x_delta
            x = x + self._ff_block(self.norm3(x))

            edge_attr_cra = edge_attr_cra + edge_attr_cra_delta
            edge_attr_cra = self._ff_block_edge(self.norm_e2(edge_attr_cra))

        else:
            if self.apply_self_attn:
                x = self.norm1(x + self._sa_block(x, edge_index_tgt))
            x_delta, edge_attr_cra_delta, attn_weight = self._cra_block(
                src, x, edge_index_cross, edge_attr_cra)
            x = self.norm2(x + x_delta)
            x = self.norm3(x + self._ff_block(x))

            edge_attr_cra = self.norm_e1(edge_attr_cra + edge_attr_cra_delta)
            edge_attr_cra = self.norm_e2(edge_attr_cra + self._ff_block_edge(edge_attr_cra))

        return x, edge_attr_cra, attn_weight
    

    def _sa_block(self, x, edge_index):
        x = self.self_attn(x, edge_index)
        return self.dropout1(x)
    
    def _cra_block(self, src, tgt, edge_index, edge_attr=None):
        x = (src, tgt)
        assert edge_attr is not None
        out, attn_weight = self.cross_attn(x, edge_index,
                                           edge_attr=edge_attr,
                                           return_attention_weights=True,
                                           return_edge_features=True)
        x, edge_feat = out
        return self.dropout2(x), edge_feat, attn_weight

    def _ff_block(self, x):
        x = self.lin2(self.dropout(self.activation(self.lin1(x))))
        return self.dropout3(x)

    def _ff_block_edge(self, e):
        e = self.lin_e2(self.dropout_e(self.activation_e(self.lin_e1(e))))
        return self.dropout_e2(e)
    

@ASSO_NET.register_module()
class EdgeAugmentedTransformerAssociationLayer(BaseModule, ABC):

    def __init__(self, 
                 embed_dims=256,
                 ffn_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 norm_first=True,
                 cross_attn_value_gate=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.model_dims = embed_dims
        self.ffn_dims = ffn_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm_first = norm_first
        self.cross_attn_value_gate = cross_attn_value_gate

        self.embedding_edge = nn.Linear(7, self.model_dims)

        self.decoder_layer = EdgeAugmentedDecoderLayer(
                                    self.model_dims, self.ffn_dims, self.num_heads, self.dropout,
                                    norm_first=self.norm_first,
                                    apply_self_attn=False,
                                    cross_attn_value_gate=self.cross_attn_value_gate)

        if self.norm_first:
            self.norm_final = nn.LayerNorm(self.model_dims)
            self.norm_final_edge = nn.LayerNorm(self.model_dims)

    def forward(self, track_query, det_query, edge_index_det,
                      edge_index_cross, edge_attr_cross, edge_attr_cross_pos):

        edge_attr_cross_pos = self.embedding_edge(edge_attr_cross_pos)
        if edge_attr_cross is None:
            edge_attr_cross = edge_attr_cross_pos
        else:
            edge_attr_cross = edge_attr_cross + edge_attr_cross_pos

        det_query, edge_attr_cross, _ = self.decoder_layer(
            track_query, det_query, edge_index_cross, edge_index_det, edge_attr_cross
        )

        if self.norm_first:
            det_query = self.norm_final(det_query)
            edge_attr_cross = self.norm_final_edge(edge_attr_cross)

        return det_query, edge_attr_cross