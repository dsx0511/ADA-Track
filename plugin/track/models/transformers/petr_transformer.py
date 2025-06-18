# ------------------------------------------------------------------------
# ADA-Track
# Copyright (c) 2024 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 Megvii Inc. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue. All Rights Reserved.
# ------------------------------------------------------------------------


import torch
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import build_norm_layer, xavier_init
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE

import copy
import torch.utils.checkpoint as cp

from plugin.track.core.bbox.util import denormalize_bbox
from plugin.track.models.attentions.association_network import ASSO_NET

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
    
@TRANSFORMER.register_module()
class PETRAssoTrackingTransformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(PETRAssoTrackingTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True


    def forward(self, x, mask, query, pos_embed, ref_points=None, reg_branches=None, track_mask=None,
                      cls_branches=None, pc_range=None,):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, n, c, h, w = x.shape
        memory = x.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        
        # target = query[:, :self.embed_dims]
        # query_embed = query[:, self.embed_dims:]

        query_embed, target = torch.split(query, self.embed_dims, dim=1)

        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        target = target.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, n, h, w] -> [bs, n*h*w]

        # out_dec: [num_layers, num_query, bs, dim]
        out_dec, inter_edge_index, inter_edge_attr = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            ref_points=ref_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            # direction_branches=direction_branches,
            # velo_branches=velo_branches,
            track_mask=track_mask,
            pc_range=pc_range,
            )
        out_dec = out_dec.transpose(1, 2)
        memory = memory.reshape(n, h, w, bs, c).permute(3, 0, 4, 1, 2)
        return  out_dec, memory, inter_edge_index, inter_edge_attr

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PETRAssoTrackingTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 association_network=dict(
                    type='EdgeAugmentedTransformerAssociationLayer'
                 ),
                 **kwargs):

        super(PETRAssoTrackingTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None
        self.asso_layers = ModuleList()
        for i in range(self.num_layers):
            self.asso_layers.append(ASSO_NET.build(association_network))

    def _build_det_track_graph(self, track_boxes, track_logits, det_boxes, det_logits, dist_thresh=10.0):
        # Box encoding: x, y, w, l, z, h, sin, cos, v_x, v_y

        det_center = copy.deepcopy(det_boxes[:, :2])
        track_center = copy.deepcopy(track_boxes[:, :2])

        # det_class = torch.argmax(det_logits, dim=1)
        # track_class = torch.argmax(track_logits, dim=1)

        # center_dist = torch.cdist(track_center, det_center, p=2.0)
        # cls_mask = torch.eq(track_class.unsqueeze(1), det_class.unsqueeze(0))
        # cls_mask = torch.logical_not(cls_mask).float() * 1e16
        # center_dist += cls_mask

        # # TODO: class-aware graph building
        # adj = torch.le(center_dist, dist_thresh).int()
        # edge_index = torch.nonzero(adj).transpose(1, 0).long()

        num_det = det_center.size(0)
        num_track = track_center.size(0)

        adj = torch.ones(num_track, num_det, 
                         dtype=torch.int32, device=det_center.device)
        edge_index = torch.nonzero(adj).transpose(1, 0).long()

        # TODO: Find a better box difference embedding
        # TODO: Add time diff embedding
        det_boxes_denorm = denormalize_bbox(det_boxes, None)
        track_boxes_denorm = denormalize_bbox(track_boxes, None)

        edge_attr = track_boxes_denorm[edge_index[0, :], :7] - det_boxes_denorm[edge_index[1, :], :7]

        return edge_index, edge_attr

    def forward(self, query, *args, ref_points=None, track_mask=None, reg_branches=None,
                      cls_branches=None, pc_range=None,
                      **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        output_edge = None
        intermediate = []
        intermediate_edge_index = []
        intermediate_edge_attr = []

        for lid, (layer, asso_layer) in enumerate(zip(self.layers, self.asso_layers)):
            query = layer(query, *args, track_mask=track_mask, **kwargs)

            query = query.permute(1, 0, 2)

            if track_mask is not None:
                reference = inverse_sigmoid(ref_points.clone())

                tmp = reg_branches[lid](query)

                tmp[..., 0:2] += reference[..., 0:2]
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                tmp[..., 4:5] += reference[..., 2:3]
                tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

                tmp[..., 0:1] = (tmp[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
                tmp[..., 1:2] = (tmp[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
                tmp[..., 4:5] = (tmp[..., 4:5] * (pc_range[5] - pc_range[2]) + pc_range[2])

                # tmp_direction = direction_branches[lid](query)
                # tmp_velo = velo_branches[lid](query)

                # tmp_box = torch.cat([tmp, tmp_direction, tmp_velo], dim=2)
                tmp_box = tmp
                tmp_track_box = tmp_box[:, track_mask, :].squeeze(0)
                tmp_det_box = tmp_box[:, ~track_mask, :].squeeze(0)

                tmp_cls = cls_branches[lid](query).detach()
                # TODO: Maybe fix the class of tracks?
                tmp_track_cls = tmp_cls[:, track_mask, :].squeeze(0)
                tmp_det_cls = tmp_cls[:, ~track_mask, :].squeeze(0)

                edge_index_cross, edge_attr_cross_pos = self._build_det_track_graph(
                    tmp_track_box.detach(), tmp_track_cls.detach(),
                    tmp_det_box.detach(), tmp_det_cls.detach())
                
                track_query = query[:, track_mask, :]
                det_query = query[:, ~track_mask, :]

                num_det = det_query.size(1)
                adj_det = torch.ones(num_det, num_det, 
                                       dtype=torch.int32, device=track_query.device)
                edge_index_det = torch.nonzero(adj_det).transpose(1, 0).long()

                det_query, output_edge = asso_layer(track_query.squeeze(0), 
                                                    det_query.squeeze(0),
                                                    edge_index_det,
                                                    edge_index_cross,
                                                    output_edge, 
                                                    edge_attr_cross_pos)

                query = torch.cat([track_query, det_query.unsqueeze(0)], dim=1)

            query = query.permute(1, 0, 2)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
                if track_mask is not None:
                    intermediate_edge_index.append(edge_index_cross)
                    intermediate_edge_attr.append(output_edge)

        return torch.stack(intermediate), intermediate_edge_index, intermediate_edge_attr

@TRANSFORMER_LAYER.register_module()
class PETRTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 with_cp=True,
                 **kwargs):
        super(PETRTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.use_checkpoint = with_cp
    
    def _forward(self, 
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                track_mask=None,
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(PETRTransformerDecoderLayer, self).forward(
                query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                track_mask=track_mask,
                )

        return x

    def forward(self, 
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                track_mask=None,
                **kwargs
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                self._forward, 
                query,
                key,
                value,
                query_pos,
                key_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                track_mask,
                )
        else:
            x = self._forward(
            query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_masks=attn_masks,
            query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask,
            track_mask=track_mask,
            )
        return x


@TRANSFORMER.register_module()
class MaskedPETRAssoTrackingTransformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(MaskedPETRAssoTrackingTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True


    def forward(self, x, mask, query, pos_embed, scene_query_embed,
                      ref_points=None, reg_branches=None, track_mask=None,
                      cls_branches=None, pc_range=None,):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, n, c, h, w = x.shape
        memory = x.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        
        # target = query[:, :self.embed_dims]
        # query_embed = query[:, self.embed_dims:]

        query_embed, target = torch.split(query, self.embed_dims, dim=1)

        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        target = target.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, n, h, w] -> [bs, n*h*w]

        scene_query_pos, scene_query = torch.split(scene_query_embed, self.embed_dims, dim=1)
        scene_query_pos = scene_query_pos.unsqueeze(0).expand(-1, bs, -1)
        scene_query = scene_query.unsqueeze(0).expand(-1, bs, -1)

        # out_dec: [num_layers, num_query, bs, dim]
        out_dec, out_scene_query, inter_edge_index, inter_edge_attr = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            ref_points=ref_points,
            scene_query=scene_query,
            scene_query_pos=scene_query_pos,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            # direction_branches=direction_branches,
            # velo_branches=velo_branches,
            track_mask=track_mask,
            pc_range=pc_range,
            )
        out_dec = out_dec.transpose(1, 2)
        out_scene_query = out_scene_query.transpose(1, 2)
        memory = memory.reshape(n, h, w, bs, c).permute(3, 0, 4, 1, 2)
        return  out_dec, memory, out_scene_query, inter_edge_index, inter_edge_attr


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MaskedPETRAssoTrackingTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 association_network=dict(
                    type='EdgeAugmentedTransformerAssociationLayer'
                 ),
                 **kwargs):

        super(MaskedPETRAssoTrackingTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None
        self.asso_layers = ModuleList()
        for i in range(self.num_layers):
            self.asso_layers.append(ASSO_NET.build(association_network))

    def _build_det_track_graph(self, track_boxes, track_logits, det_boxes, det_logits, dist_thresh=10.0):
        # Box encoding: x, y, w, l, z, h, sin, cos, v_x, v_y

        det_center = copy.deepcopy(det_boxes[:, :2])
        track_center = copy.deepcopy(track_boxes[:, :2])

        # det_class = torch.argmax(det_logits, dim=1)
        # track_class = torch.argmax(track_logits, dim=1)

        # center_dist = torch.cdist(track_center, det_center, p=2.0)
        # cls_mask = torch.eq(track_class.unsqueeze(1), det_class.unsqueeze(0))
        # cls_mask = torch.logical_not(cls_mask).float() * 1e16
        # center_dist += cls_mask

        # # TODO: class-aware graph building
        # adj = torch.le(center_dist, dist_thresh).int()
        # edge_index = torch.nonzero(adj).transpose(1, 0).long()

        num_det = det_center.size(0)
        num_track = track_center.size(0)

        adj = torch.ones(num_track, num_det, 
                         dtype=torch.int32, device=det_center.device)
        edge_index = torch.nonzero(adj).transpose(1, 0).long()

        # TODO: Find a better box difference embedding
        # TODO: Add time diff embedding
        det_boxes_denorm = denormalize_bbox(det_boxes, None)
        track_boxes_denorm = denormalize_bbox(track_boxes, None)

        edge_attr = track_boxes_denorm[edge_index[0, :], :7] - det_boxes_denorm[edge_index[1, :], :7]

        return edge_index, edge_attr

    def forward(self, 
                query, 
                *args, 
                query_pos=None, 
                ref_points=None, 
                scene_query=None, 
                scene_query_pos=None, 
                track_mask=None, 
                reg_branches=None, 
                cls_branches=None, 
                pc_range=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        output_edge = None
        intermediate = []
        intermediate_edge_index = []
        intermediate_edge_attr = []
        intermediate_scene_query = []

        num_query = query.size(0)
        query_pos = torch.cat([scene_query_pos, query_pos], dim=0)

        for lid, (layer, asso_layer) in enumerate(zip(self.layers, self.asso_layers)):

            query = torch.cat([scene_query, query], dim=0)

            query = layer(query, 
                          *args, 
                          query_pos=query_pos,
                          track_mask=track_mask,
                          num_query=num_query,
                          **kwargs)

            query = query.permute(1, 0, 2) # [batch_size, num_queries, query_dims]

            # Remove scene token from queries
            scene_query, query = torch.split(query, [1, num_query], dim=1)

            if track_mask is not None:
                reference = inverse_sigmoid(ref_points.clone())

                tmp = reg_branches[lid](query)

                tmp[..., 0:2] += reference[..., 0:2]
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                tmp[..., 4:5] += reference[..., 2:3]
                tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

                tmp[..., 0:1] = (tmp[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
                tmp[..., 1:2] = (tmp[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
                tmp[..., 4:5] = (tmp[..., 4:5] * (pc_range[5] - pc_range[2]) + pc_range[2])

                # tmp_direction = direction_branches[lid](query)
                # tmp_velo = velo_branches[lid](query)

                tmp_scene_token_box = torch.zeros([1, 10], dtype=torch.float, device=scene_query.device)
                tmp_scene_token_cls = torch.zeros([1, 7], dtype=torch.float, device=scene_query.device)

                # tmp_box = torch.cat([tmp, tmp_direction, tmp_velo], dim=2)
                tmp_box = tmp
                tmp_track_box = tmp_box[:, track_mask, :].squeeze(0)
                tmp_det_box = tmp_box[:, ~track_mask, :].squeeze(0)

                tmp_cls = cls_branches[lid](query).detach()
                # TODO: Maybe fix the class of tracks?
                tmp_track_cls = tmp_cls[:, track_mask, :].squeeze(0)
                tmp_det_cls = tmp_cls[:, ~track_mask, :].squeeze(0)
                
                track_query = query[:, track_mask, :]
                det_query = query[:, ~track_mask, :]

                tmp_track_box = torch.cat([tmp_scene_token_box, tmp_track_box], dim=0)
                tmp_track_cls = torch.cat([tmp_scene_token_cls, tmp_track_cls], dim=0)
                track_query = torch.cat([scene_query, track_query], dim=1)

                edge_index_cross, edge_attr_cross_pos = self._build_det_track_graph(
                    tmp_track_box.detach(), tmp_track_cls.detach(),
                    tmp_det_box.detach(), tmp_det_cls.detach())

                num_track = track_query.size(1)
                num_det = det_query.size(1)
                adj_det = torch.ones(num_det, num_det, 
                                       dtype=torch.int32, device=track_query.device)
                edge_index_det = torch.nonzero(adj_det).transpose(1, 0).long()

                det_query, output_edge = asso_layer(track_query.squeeze(0), 
                                                    det_query.squeeze(0),
                                                    edge_index_det,
                                                    edge_index_cross,
                                                    output_edge, 
                                                    edge_attr_cross_pos)
                
                scene_query, track_query = torch.split(track_query, [1, num_track-1], dim=1)

                query = torch.cat([track_query, det_query.unsqueeze(0)], dim=1)

            query = query.permute(1, 0, 2)
            scene_query = scene_query.permute(1, 0, 2)

            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
                intermediate_scene_query.append(scene_query)
                if track_mask is not None:
                    intermediate_edge_index.append(edge_index_cross)
                    intermediate_edge_attr.append(output_edge)

        return torch.stack(intermediate), torch.stack(intermediate_scene_query), \
               intermediate_edge_index, intermediate_edge_attr
    
@TRANSFORMER_LAYER.register_module()
class MaskedPETRTransformerDecoderLayer(PETRTransformerDecoderLayer):

    def _forward(self,
                 query,
                 key=None,
                 value=None,
                 query_pos=None,
                 key_pos=None,
                 attn_masks=None,
                 query_key_padding_mask=None,
                 key_padding_mask=None,
                 track_mask=None,
                 num_query=None,
                 **kwargs):
        """Forward function for `TransformerDecoderLayer`.
        **kwargs contains some specific arguments of attentions.
        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                scene_query, query = torch.split(query, [1, num_query], dim=0)
                scene_query_pos, query_pos = torch.split(query_pos, [1, num_query], dim=0)
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                query = torch.cat([scene_query, query], dim=0)
                query_pos = torch.cat([scene_query_pos, query_pos], dim=0)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

    def forward(self, 
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                track_mask=None,
                num_query=None,
                **kwargs
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                self._forward, 
                query,
                key,
                value,
                query_pos,
                key_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                track_mask,
                num_query,
                )
        else:
            x = self._forward(
            query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_masks=attn_masks,
            query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask,
            track_mask=track_mask,
            num_query=num_query,
            )
        return x