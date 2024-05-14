# ------------------------------------------------------------------------
# ADA-Track
# Copyright (c) 2024 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MUTR3D (https://github.com/a1600012888/MUTR3D)
# Copyright (c) 2022 MARS Lab. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue. All Rights Reserved.
# ------------------------------------------------------------------------

import copy
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule, ModuleList

from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import inverse_sigmoid

from plugin.track.core.bbox.util import denormalize_bbox
from plugin.track.models.attentions.association_network import ASSO_NET

@TRANSFORMER.register_module()
class DETR3DAssoTrackingTransformer(BaseModule):
    """Implements the DeformableDETR transformer. 
        Specially designed for track: keep xyz trajectory, and 
        kep bbox size(which should be consisten across frames)

    Args:
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 decoder=None,
                 reference_points_aug=False,
                 **kwargs):
        super(DETR3DAssoTrackingTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.reference_points_aug = reference_points_aug
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        # self.level_embeds = nn.Parameter(
        #     torch.Tensor(self.num_feature_levels, self.embed_dims))

        # self.cam_embeds = nn.Parameter(
        #     torch.Tensor(self.num_cams, self.embed_dims))

        # move ref points to tracker
        # self.reference_points = nn.Linear(self.embed_dims, 3)
        pass

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                mlvl_feats,
                query_embed,
                reference_points,
                ref_size,
                reg_branches=None,
                cls_branches=None,
                direction_branches=None,
                velo_branches=None,
                pc_range=None,
                track_mask=None,
                **kwargs):
        """Forward function for `Transformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, 2*embed_dim], can be splitted into
                query_feat and query_positional_encoding.
            reference_points (Tensor): The corresponding 3d ref points
                for the query with shape (num_query, 3)
                value is in inverse sigmoid space
            ref_size (Tensor): the wlh(bbox size) associated with each query
                shape (num_query, 3)
                value in log space. 
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder, has shape \
                      (num_dec_layers, num_query, bs, embed_dims)
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 3).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs, num_query, 3)

        """
        assert query_embed is not None
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = reference_points.unsqueeze(dim=0).expand(bs, -1, -1)
        ref_size = ref_size.unsqueeze(dim=0).expand(bs, -1, -1)

        if self.training and self.reference_points_aug:
            reference_points = reference_points + \
                torch.randn_like(reference_points)
        reference_points = reference_points.sigmoid()
        # decoder
        query = query.permute(1, 0, 2)
        # memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references, inter_box_sizes, inter_edge_index, inter_edge_attr = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            direction_branches=direction_branches,
            velo_branches=velo_branches,
            ref_size=ref_size,
            pc_range=pc_range,
            track_mask=track_mask,
            **kwargs)

        return inter_states, inter_references, inter_box_sizes, inter_edge_index, inter_edge_attr

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DETR3DAssoTrackingTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=True, 
                 association_network=dict(
                    type='EdgeAugmentedTransformerAssociationLayer'
                 ),
                 **kwargs):

        super(DETR3DAssoTrackingTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.asso_layers = ModuleList()
        for i in range(self.num_layers):
            self.asso_layers.append(ASSO_NET.build(association_network))
        

    # TODO: refractor -> graph build class
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
                reference_points=None,
                reg_branches=None,
                cls_branches=None,
                direction_branches=None,
                velo_branches=None,
                ref_size=None,
                pc_range=None,
                track_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The 3d reference points
                associated with each query. shape (num_query, 3).
                value is in inevrse sigmoid space
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
            ref_size (Tensor): the wlh(bbox size) associated with each query
                shape (bs, num_query, 3)
                value in log space. 
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        output_edge = None
        intermediate = []
        intermediate_reference_points = []
        intermediate_box_sizes = []
        intermediate_edge_index = []
        intermediate_edge_attr = []
        for lid, (layer, asso_layer) in enumerate(zip(self.layers, self.asso_layers)):
            reference_points_input = reference_points
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                ref_size=ref_size,
                track_mask=track_mask,
                **kwargs)
            output = output.permute(1, 0, 2) # [batch_size, num_queries, query_dims]
            
            # TODO: consider update box here completely (all params, with gradients...)
            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                ref_pts_update = torch.cat(
                    [
                        tmp[..., :2],
                        tmp[..., 4:5],
                    ], dim=-1
                )
                ref_size_update = torch.cat(
                    [
                        tmp[..., 2:4],
                        tmp[..., 5:6]
                    ], dim=-1
                )
                assert reference_points.shape[-1] == 3

                new_reference_points = ref_pts_update + \
                    inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

                # add in log space
                # ref_size = (ref_size.exp() + ref_size_update.exp()).log()
                ref_size = ref_size + ref_size_update
                if lid > 0:
                    ref_size = ref_size.detach()

            ##################################################################################
            ## Association network starts here
            ##################################################################################
            if track_mask is not None: # First frame is not considered for association
                tmp_direction = direction_branches[lid](output)
                tmp_velo = velo_branches[lid](output)

                reference_points_global = copy.deepcopy(reference_points)
                reference_points_global[..., 0] = (reference_points_global[..., 0] *
                                                   (pc_range[3] - pc_range[0]) + pc_range[0])
                reference_points_global[..., 1] = (reference_points_global[..., 1] *
                                                   (pc_range[4] - pc_range[1]) + pc_range[1])
                reference_points_global[..., 2] = (reference_points_global[..., 2] *
                                                   (pc_range[5] - pc_range[2]) + pc_range[2])
                                                   
                tmp_box = torch.cat([reference_points_global[..., 0:2], ref_size[..., 0:2],
                                    reference_points_global[..., 2:3], ref_size[..., 2:3],
                                    tmp_direction, tmp_velo], dim=2)

                tmp_track_box = tmp_box[:, track_mask, :].squeeze(0)
                tmp_det_box = tmp_box[:, ~track_mask, :].squeeze(0)

                tmp_cls = cls_branches[lid](output).detach()
                # TODO: Maybe fix the class of tracks?
                tmp_track_cls = tmp_cls[:, track_mask, :].squeeze(0)
                tmp_det_cls = tmp_cls[:, ~track_mask, :].squeeze(0)

                edge_index_cross, edge_attr_cross_pos = self._build_det_track_graph(
                    tmp_track_box.detach(), tmp_track_cls.detach(),
                    tmp_det_box.detach(), tmp_det_cls.detach())
                                                                               
                track_query = output[:, track_mask, :]
                det_query = output[:, ~track_mask, :]

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

                output = torch.cat([track_query, det_query.unsqueeze(0)], dim=1)


            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_box_sizes.append(ref_size)
                if track_mask is not None:
                    intermediate_edge_index.append(edge_index_cross)
                    intermediate_edge_attr.append(output_edge)

        if self.return_intermediate:
            return (torch.stack(intermediate),
                    torch.stack(intermediate_reference_points),
                    torch.stack(intermediate_box_sizes),
                    intermediate_edge_index,
                    intermediate_edge_attr)

        return output, reference_points, ref_size