# ------------------------------------------------------------------------
# ADA-Track
# Copyright (c) 2024 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MUTR3D (https://github.com/a1600012888/MUTR3D)
# Copyright (c) 2022 MARS Lab. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import numpy as np
from copy import deepcopy
from contextlib import nullcontext

from mmdet.models import DETECTORS
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models import build_loss
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.runner import force_fp32, auto_fp16

from plugin.track.core.bbox.util import normalize_bbox, denormalize_bbox
from plugin.track.models import MUTRMVXTwoStageDetector
from ..utils.grid_mask import GridMask
from ...core.structures import Instances
from ..utils.qim import build_qim
from ..dense_heads.detr3d_head import DETR3DAssoTrackingHead
from ..dense_heads.petr_head import PETRAssoTrackingHead, pos2posemb3d
from .runtime_tracker_base import RuntimeTrackerBase


@DETECTORS.register_module()
class AlternatingDetAssoCamTracker(MUTRMVXTwoStageDetector):
    """Tracker which support image w, w/o radar."""

    def __init__(self,
                 embed_dims=256,
                 num_query=300,
                 num_classes=7,
                 bbox_coder=dict(
                     type='DETRTrack3DCoder',
                     post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                     pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                     max_num=300,
                     num_classes=7),
                 qim_args=dict(
                     qim_type='QIMBase',
                     merger_dropout=0, update_query_pos=False,
                     fp_ratio=0.3, random_drop=0.1),
                 fix_feats=True,
                 fix_neck=True,
                 score_thresh=0.2,
                 filter_score_thresh=0.1,
                 miss_tolerance=5,
                 affinity_thresh=0.2, 
                 hungarian=True,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 loss_cfg=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 history_frames=0,
                 with_data_aug=False,
                 with_logs=False,
                 feature_update_rate=0.5,
                 ):
        super(AlternatingDetAssoCamTracker,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.num_classes = num_classes
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range

        self.embed_dims = embed_dims
        self.num_query = num_query
        self.fix_feats = fix_feats
        self.fix_neck = fix_neck
        if self.fix_feats:
            self.img_backbone.eval()
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        
        if self.fix_neck:
            self.img_neck.eval()
            for param in self.img_neck.parameters():
                param.requires_grad = False

        if isinstance(self.pts_bbox_head, DETR3DAssoTrackingHead):
            self.reference_points = nn.Linear(self.embed_dims, 3)
            self.bbox_size_fc = nn.Linear(self.embed_dims, 3)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
        elif isinstance(self.pts_bbox_head, PETRAssoTrackingHead):
            self.reference_points = nn.Embedding(self.num_query, 3)
            self.query_embedding = nn.Sequential(
                nn.Linear(self.embed_dims*3//2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
            self.query_feat_embedding = nn.Embedding(self.num_query, self.embed_dims)

            nn.init.uniform_(self.reference_points.weight.data, 0, 1)
            nn.init.zeros_(self.query_feat_embedding.weight)
        else:
            raise ValueError('Not supported pts_bbox_head')

        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance,
            affinity_thresh=affinity_thresh, 
            hungarian=hungarian,)  # hyper-param for removing inactive queries

        self.query_interact = build_qim(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        self.criterion = build_loss(loss_cfg)
        self.test_track_instances = None
        self.l2g_r_mat = None
        self.l2g_t = None

        # for inference we need to determine wheter a new scene started
        self._test_current_scene_token = None

        # whether to use all loaded frames for tracking or keep x frames as history for multi-shot detectors
        self._history_frames = history_frames

        self._with_data_aug = with_data_aug
        self._with_logs = with_logs

        self.alpha = feature_update_rate


    def _associate_track_instances(self, track_instances, all_detection_instances):
        
        ## Only consider detection queries that matched with gt
        detection_instances = all_detection_instances[all_detection_instances.obj_idxes >= 0]
        track_instances = track_instances[track_instances.obj_idxes >= 0]

        track_idxes = track_instances.obj_idxes
        det_idxes = detection_instances.obj_idxes
        # track_idxes[track_idxes == -1] = -100
        match_indices = torch.eq(track_idxes.unsqueeze(1), det_idxes.unsqueeze(0)).nonzero()

        ## Merge matched track-detection queries
        if self.alpha is None:
            track_instances.output_embedding[match_indices[:, 0]] += detection_instances.output_embedding[match_indices[:, 1]]
        else:
            track_instances.output_embedding[match_indices[:, 0]] = \
                self.alpha * track_instances.output_embedding[match_indices[:, 0]] + \
                (1 - self.alpha) * detection_instances.output_embedding[match_indices[:, 1]]

        track_instances.query = torch.cat([track_instances.query[..., 0:self.embed_dims], 
                                           track_instances.output_embedding], dim=-1)
        
        track_instances.matched_gt_idxes[match_indices[:, 0]] = detection_instances.matched_gt_idxes[match_indices[:, 1]]
        track_instances.ref_pts[match_indices[:, 0]] = detection_instances.ref_pts[match_indices[:, 1]]
        track_instances.pred_logits[match_indices[:, 0]] = detection_instances.pred_logits[match_indices[:, 1]]
        # TODO: Update of motion
        track_instances.pred_boxes[match_indices[:, 0]] = detection_instances.pred_boxes[match_indices[:, 1]]

        # Handle inactivate tracks
        unmatched_tracks = torch.ones_like(track_idxes, dtype=torch.int)
        unmatched_tracks[match_indices[:, 0]] = 0
        unmatched_track_indices = unmatched_tracks.nonzero().squeeze(1)
        track_instances.matched_gt_idxes[unmatched_track_indices] = -1

        # Init detection queries 
        unmatched_dets = torch.ones_like(det_idxes, dtype=torch.int)
        unmatched_dets[match_indices[:, 1]] = 0
        unmatched_det_indices = unmatched_dets.nonzero().squeeze(1)
        unmatched_det_instances = detection_instances[unmatched_det_indices]

        merged_track_instances = Instances.cat([track_instances, unmatched_det_instances])

        # Data augmentations
        if self._with_data_aug:
            merged_track_instances = self.query_interact._random_drop_tracks(merged_track_instances)
            merged_track_instances = self.query_interact._add_fp_tracks(
                all_detection_instances, merged_track_instances)

        return merged_track_instances


    def _velo_update(self, ref_pts, track_box, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                    time_delta):
        '''
        Args:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
            velocity (Tensor): (num_query, 2). m/s
                in lidar frame. vx, vy
            global2lidar (np.Array) [4,4].
        Outs:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
        '''
        # print(l2g_r1.type(), l2g_t1.type(), ref_pts.type())
        time_delta = time_delta.type(torch.float)
        num_query = ref_pts.size(0)
        velo_pad_ = velocity.new_zeros((num_query, 1))
        velo_pad = torch.cat((velocity, velo_pad_), dim=-1)

        if isinstance(self.pts_bbox_head, DETR3DAssoTrackingHead):
            reference_points = ref_pts.sigmoid().clone()
        elif isinstance(self.pts_bbox_head, PETRAssoTrackingHead):
            reference_points = ref_pts.clone()
        else:
            raise ValueError('Not supported pts_bbox_head')

        pc_range = self.pc_range
        reference_points[..., 0:1] = reference_points[...,
                                                      0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[...,
                                                      1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[...,
                                                      2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = reference_points + velo_pad * time_delta

        ref_pts = reference_points @ l2g_r1 + l2g_t1 - l2g_t2

        g2l_r = torch.linalg.inv(l2g_r2).type(torch.float)

        ref_pts = ref_pts @ g2l_r

        xywlzh = track_box
        xywlzh[..., 0:1] = ref_pts[..., 0:1]
        xywlzh[..., 1:2] = ref_pts[..., 1:2]
        xywlzh[..., 4:5] = ref_pts[..., 2:3]

        ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]
                             ) / (pc_range[3] - pc_range[0])
        ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]
                             ) / (pc_range[4] - pc_range[1])
        ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]
                             ) / (pc_range[5] - pc_range[2])

        if isinstance(self.pts_bbox_head, DETR3DAssoTrackingHead):
            ref_pts = inverse_sigmoid(ref_pts)

        return ref_pts, xywlzh


    def _update_query_pos(self, track_instances):
        if isinstance(self.pts_bbox_head, PETRAssoTrackingHead):
            reference_points = track_instances.ref_pts
            query_embeds = self.query_embedding(pos2posemb3d(reference_points))
            track_instances.query = torch.cat([query_embeds,
                                              track_instances.query[..., self.embed_dims:]], dim=-1)
            
        return track_instances


    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x


    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        feats_context = nullcontext()
        neck_context = nullcontext()

        if self.fix_feats:
            feats_context = torch.no_grad()
        if self.fix_neck:
            neck_context = torch.no_grad()

        B = img.size(0)
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                in_shape = [input_shape for _ in range(
                    len(img_meta['img_shape']))]
                img_meta.update(input_shape=in_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            with feats_context:
                if self.use_grid_mask:
                    img = self.grid_mask(img)
                img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            with neck_context:
                img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped


    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, points, img, radar=None, img_metas=None):
        """Extract features from images and lidar points and radars."""
        img_feats = self.extract_img_feat(img, img_metas)
        return (img_feats, None, None)

    
    def _generate_empty_tracks(self):
        if isinstance(self.pts_bbox_head, DETR3DAssoTrackingHead):
            return self._generate_empty_tracks_detr()
        elif isinstance(self.pts_bbox_head, PETRAssoTrackingHead):
            return self._generate_empty_tracks_petr()
        else:
            raise ValueError('Not supported pts_bbox_head')

    def _generate_empty_tracks_detr(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape  # (300, 256 * 2)
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight
        track_instances.ref_pts = self.reference_points(
            query[..., :dim // 2])

        # dim // 2 == 256 -> embedding is 2x latent dim  (512)
        # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
        box_sizes = self.bbox_size_fc(query[..., :dim // 2])

        bbox_pred_dims = 10
        pred_boxes_init = torch.zeros(
            (len(track_instances), bbox_pred_dims), dtype=torch.float, device=device)

        pred_boxes_init[..., 2:4] = box_sizes[..., 0:2]
        pred_boxes_init[..., 5:6] = box_sizes[..., 2:3]

        track_instances.query = query

        # dim >> 1 == 256 -> embedding is 2x latent dim  (512)
        track_instances.output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device)

        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros(
            (len(track_instances), ), dtype=torch.long, device=device)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        # xy, wl, z, h, sin, cos, vx, vy, vz
        track_instances.pred_boxes = pred_boxes_init

        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes),
            dtype=torch.float, device=device)

        return track_instances.to(self.query_embedding.weight.device)

    def _generate_empty_tracks_petr(self):
        track_instances = Instances((1, 1))
        device = self.reference_points.weight.device

        reference_points = self.reference_points.weight
        query_embeds = self.query_embedding(pos2posemb3d(reference_points)) # TODO: Update every timestamp
        track_instances.ref_pts = reference_points.clone()

        query_feats = self.query_feat_embedding.weight.clone()

        # First half query feature, second half query position encoding <-- It might be wrong!!!
        # track_instances.query = torch.cat([query_feats, query_embeds], dim=1)
        
         # First half query position encoding, second half query feature
        track_instances.query = torch.cat([query_embeds, query_feats], dim=1)

        track_instances.output_embedding = torch.zeros(
            (self.num_query, self.embed_dims), device=device)

        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros(
            (len(track_instances), ), dtype=torch.long, device=device)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        
        # xy, wl, z, h, sin, cos, vx, vy, vz
        bbox_pred_dims = 10
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), bbox_pred_dims), dtype=torch.float, device=device)

        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes),
            dtype=torch.float, device=device)

        return track_instances.to(device)

    def _copy_tracks_for_loss(self, tgt_instances):

        if isinstance(self.pts_bbox_head, DETR3DAssoTrackingHead):
            device = self.query_embedding.weight.device
        elif isinstance(self.pts_bbox_head, PETRAssoTrackingHead):
            device = self.reference_points.weight.device
        else:
            raise ValueError('Not supported pts_bbox_head')

        track_instances = Instances((1, 1))

        track_instances.obj_idxes = deepcopy(tgt_instances.obj_idxes)
        track_instances.matched_gt_idxes = deepcopy(
            tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = deepcopy(tgt_instances.disappear_time)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes),
            dtype=torch.float, device=device)

        return track_instances.to(device)

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def _forward_single(self, points, img, radar, img_metas, track_instances,
                        l2g_r1=None, l2g_t1=None, l2g_r2=None, l2g_t2=None, 
                        time_delta=None, first_frame=False):
        '''
        Perform forward only on one frame. Called in  forward_train
        Warnning: Only Support BS=1
        Args:
            img: shape [B, num_cam, 3, H, W]

            if l2g_r2 is None or l2g_t2 is None:
                it means this frame is the end of the training clip,
                so no need to call velocity update
        '''
        B, num_cam, _, H, W = img.shape
        img_feats, radar_feats, pts_feats = self.extract_feat(
            points, img=img, radar=radar, img_metas=img_metas)

        if first_frame:
            track_logits = None
            track_mask = None
            merged_instances = track_instances
        else:
            track_logits = track_instances.pred_logits

            # Velo update
            ref_pts = track_instances.ref_pts
            track_box = track_instances.pred_boxes

            velo = track_box[:, -2:]  # [num_query, 2]

            ref_pts, track_box = self._velo_update(
                ref_pts, track_box, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                time_delta=time_delta)

            track_instances.ref_pts = ref_pts
            track_instances.pred_boxes = track_box

            track_instances = self._update_query_pos(track_instances)

            detection_instances = self._generate_empty_tracks()
            merged_instances = Instances.cat([track_instances, detection_instances])
            # track_mask = merged_instances.obj_idxes >= 0
            track_mask = torch.zeros(len(merged_instances), dtype=torch.bool,
                                     device=track_instances.query.device)
            track_mask[:len(track_instances)] = 1
            
        ref_box_sizes = torch.cat(
            [merged_instances.pred_boxes[:, 2:4],
             merged_instances.pred_boxes[:, 5:6]], dim=1)
        
        head_out = self.pts_bbox_head(
            mlvl_feats=img_feats,
            radar_feats=radar_feats,
            query_embeds=merged_instances.query,
            ref_points=merged_instances.ref_pts,
            ref_size=ref_box_sizes,
            img_metas=img_metas,
            track_mask=track_mask,
            track_logits=track_logits
        )

        # output_classes: [num_dec, B, num_query, num_classes]
        # output_coords: [num_dec, B, num_query, box_coder_size]
        # query_feats: [B, num_query, embed_dim]
        # last_ref_pts: [B, num_query, 3]
        output_classes, output_coords, output_affinities, inter_edge_index, \
            query_feats, last_ref_pts = head_out
        
        with torch.no_grad():
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        
        nb_dec = output_classes.size(0)
        merged_instances_list = [self._copy_tracks_for_loss(
            merged_instances) for i in range(nb_dec-1)]

        merged_instances.output_embedding = query_feats[0]
        merged_instances.ref_pts = last_ref_pts[0]
        merged_instances_list.append(merged_instances)

        for i in range(nb_dec):
            merged_instances = merged_instances_list[i]
            # track_scores = output_classes[i, 0, :].sigmoid().max(dim=-1).values

            merged_instances.scores = track_scores
            # [300, num_cls]
            merged_instances.pred_logits = output_classes[i, 0]
            merged_instances.pred_boxes = output_coords[i, 0]  # [300, box_dim]

            if track_mask is not None:
                track_instances = merged_instances[track_mask]
                det_instances = merged_instances[~track_mask]
                track_instances = self.criterion.match_for_track_queries(
                    track_instances, dec_lvl=i)
                det_instances = self.criterion.match_for_detection_queries(
                    det_instances, dec_lvl=i)
                merged_instances = Instances.cat([track_instances, det_instances])

                self.criterion.match_for_association(det_instances, track_instances,
                                                     output_affinities[i], inter_edge_index[i], dec_lvl=i)

            else:
                merged_instances = self.criterion.match_for_detection_queries(
                    merged_instances, dec_lvl=i)
                            
        if track_mask is not None:
            track_instances = merged_instances[track_mask]
            det_instances = merged_instances[~track_mask]
            output_track_instances = self._associate_track_instances(track_instances, det_instances)
        else:
            output_track_instances = merged_instances[merged_instances.obj_idxes >= 0]
            if self._with_data_aug:
                output_track_instances = self.query_interact._random_drop_tracks(output_track_instances)
                output_track_instances = self.query_interact._add_fp_tracks(
                    merged_instances, output_track_instances)
        self.criterion.step()
        return {'pred_logits': output_classes[-1],
                'pred_boxes': output_coords[-1],
                'ref_pts': last_ref_pts,
                'track_instances': output_track_instances}

    def forward_train(self,
                      points=None,
                      img=None,
                      radar=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      instance_inds=None,
                      l2g_r_mat=None,
                      l2g_t=None,
                      timestamp=None,
                      **kwargs
                      ):
        """Forward training function.
        This function will call _forward_single in a for loop

        Args:
            points (list(list[torch.Tensor]), optional): B-T-sample
                Points of each sample.
                Defaults to None.
            img (Torch.Tensor) of shape [B, T, num_cam, 3, H, W]
            radar (Torch.Tensor) of shape [B, T, num_points, radar_dim]
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            lidar2img = img_metas[bs]['lidar2img'] of shape [3, 6, 4, 4]. list
                of list of list of 4x4 array
            gt_bboxes_3d (list[list[:obj:`BaseInstance3DBoxes`]], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[list[torch.Tensor]], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            l2g_r_mat (list[Tensor]). element shape [T, 3, 3]
            l2g_t (list[Tensor]). element shape [T, 3]
                normally you should call points @ R_Mat.T + T
                here, just call points @ R_mat + T
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        # [T+1, 3, 3]
        l2g_r_mat = l2g_r_mat[0]
        # change to [T+1, 1, 3]
        l2g_t = l2g_t[0].unsqueeze(dim=1)

        timestamp = timestamp

        bs = img.size(0)
        assert bs == 1, "Currently only bs=1 is supported"
        num_frame = img.size(1) - 1
        track_instances = self._generate_empty_tracks()

        # init gt instances!
        gt_instances_list = []
        # start at 0 (if sinlge shot detector), start at x for multi-shot
        for i in range(self._history_frames, num_frame):

            gt_instances = Instances((1, 1))
            boxes = gt_bboxes_3d[0][i].tensor.to(img.device)
            # normalize gt bboxes here!
            boxes = normalize_bbox(boxes, self.pc_range)

            gt_instances.boxes = boxes
            gt_instances.labels = gt_labels_3d[0][i]
            gt_instances.obj_ids = instance_inds[0][i]
            gt_instances_list.append(gt_instances)

        self.criterion.initialize_for_single_clip(gt_instances_list)

        # start at 0 (if sinlge shot detector), start at x for multi-shot
        for i in range(self._history_frames, num_frame):

            # TODO refactor / hacky: points is None if no lidar is in use -> remove dependency to load points here
            if points:
                points_single = [p_[i] for p_ in points]
            else:
                points_single = None
            img_single = torch.stack([img_[i] for img_ in img], dim=0)
            # TODO refactor / hacky: points is None if no lidar is use -> remove dependency to load points here
            if radar:
                radar_single = torch.stack(
                    [radar_[i] for radar_ in radar], dim=0)
            else:
                radar_single = None

            img_metas_single = deepcopy(img_metas)
            # [0] bs=1
            for k in list(img_metas_single[0].keys()):
                img_metas_single[0][k] = img_metas_single[0][k][i]

            l2g_r2 = l2g_r_mat[i+1]
            l2g_t2 = l2g_t[i+1]
            time_delta = timestamp[i+1] - timestamp[i]

            first_frame = i == self._history_frames
            frame_res = self._forward_single(points_single, img_single,
                                             radar_single, img_metas_single,
                                             track_instances,
                                             l2g_r_mat[i], l2g_t[i],
                                             l2g_r2, l2g_t2,
                                             time_delta=time_delta,
                                             first_frame=first_frame)
            track_instances = frame_res['track_instances']

        outputs = self.criterion.losses_dict
        return outputs

    def _inference_single(self, points, img, radar, img_metas, track_instances,
                          l2g_r1=None, l2g_t1=None, l2g_r2=None, l2g_t2=None,
                          time_delta=None, first_frame=False):
        '''
        This function will be called at forward_test

        Warnning: Only Support BS=1
        img: shape [B, num_cam, 3, H, W]
        '''

        B, num_cam, _, H, W = img.shape
        img_feats, radar_feats, pts_feats = self.extract_feat(
            points, img=img, radar=radar, img_metas=img_metas)
        img_feats = [a.clone() for a in img_feats]

        if first_frame:
            track_logits = None
            track_mask = None
            merged_instances = track_instances
        else:
            track_logits = track_instances.pred_logits

            # Velo update
            ref_pts = track_instances.ref_pts
            track_box = track_instances.pred_boxes

            velo = track_box[:, -2:]  # [num_query, 2]

            ref_pts, track_box = self._velo_update(
                ref_pts, track_box, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                time_delta=time_delta)

            track_instances.ref_pts = ref_pts
            track_instances.pred_boxes = track_box

            track_instances = self._update_query_pos(track_instances)

            detection_instances = self._generate_empty_tracks()
            merged_instances = Instances.cat([track_instances, detection_instances])
            # track_mask = merged_instances.obj_idxes >= 0
            track_mask = torch.zeros(len(merged_instances), dtype=torch.bool,
                                     device=track_instances.query.device)
            track_mask[:len(track_instances)] = 1
            
        ref_box_sizes = torch.cat(
            [merged_instances.pred_boxes[:, 2:4],
             merged_instances.pred_boxes[:, 5:6]], dim=1)
        
        head_out = self.pts_bbox_head(
            mlvl_feats=img_feats,
            radar_feats=radar_feats,
            query_embeds=merged_instances.query,
            ref_points=merged_instances.ref_pts,
            ref_size=ref_box_sizes,
            img_metas=img_metas,
            track_mask=track_mask,
            track_logits=track_logits
        )

        # output_classes: [num_dec, B, num_query, num_classes]
        # output_coords: [num_dec, B, num_query, box_coder_size]
        # query_feats: [B, num_query, embed_dim]
        # last_ref_pts: [B, num_query, 3]
        output_classes, output_coords, output_affinities, inter_edge_index, \
            query_feats, last_ref_pts = head_out
        
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        merged_instances.scores = track_scores
        merged_instances.pred_logits = output_classes[-1, 0]
        merged_instances.pred_boxes = output_coords[-1, 0]
        merged_instances.output_embedding = query_feats[0]
        merged_instances.ref_pts = last_ref_pts[0]

        if first_frame:
            self.track_base.init_tracks(merged_instances)
            active_track_instances = merged_instances[merged_instances.obj_idxes >= 0]
        else:
            track_instances = merged_instances[track_mask]
            det_instances = merged_instances[~track_mask]
            track_instances = self.track_base.update_using_asso_score(
                det_instances, track_instances, 
                inter_edge_index[-1], output_affinities[-1].squeeze(-1).sigmoid(), self.embed_dims, self.alpha)
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return {'pred_logits': track_instances.pred_logits,
                'pred_boxes': track_instances.pred_boxes,
                'ref_pts': track_instances.ref_pts,
                'track_instances': active_track_instances}
    
    def forward_test(self,
                     points=None,
                     img=None,
                     radar=None,
                     img_metas=None,
                     timestamp=1e6,
                     l2g_r_mat=None,
                     l2g_t=None,
                     **kwargs,
                     ):
        """Forward test function.
        only support bs=1, single-gpu, num_frame=1 test
        Args:
            points (list(list[torch.Tensor]), optional): B-T-sample
                Points of each sample.
                Defaults to None.
            img (Torch.Tensor) of shape [B, T, num_cam, 3, H, W]
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            lidar2img = img_metas[bs]['lidar2img'] of shape [3, 6, 4, 4]. list
                of list of list of 4x4 array
            gt_bboxes_3d (list[list[:obj:`BaseInstance3DBoxes`]], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[list[torch.Tensor]], optional): Ground truth labels
                of 3D boxes. Defaults to None.

            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        # [3, 3]
        l2g_r_mat = l2g_r_mat[0][0]

        # change to [1, 3]
        l2g_t = l2g_t[0].unsqueeze(dim=1)[0]

        bs = img.size(0)
        num_frame = img.size(1)

        timestamp = timestamp[0]

        # assumes bs = 1
        scene_token = img_metas[0]['scene_token']

        # since t==1: scene_token = [scene_token]
        if self.test_track_instances is None or self._test_current_scene_token != scene_token[0]:
            is_first_frame = True
            # next scene -> reset all tracks
            self._test_current_scene_token = scene_token[0]
            track_instances = self._generate_empty_tracks()
            self.test_track_instances = track_instances

            self.timestamp = timestamp[0]

            time_delta = None
            l2g_r1 = None
            l2g_t1 = None
            l2g_r2 = None
            l2g_t2 = None

        else:
            is_first_frame = False
            track_instances = self.test_track_instances
            time_delta = timestamp[0] - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t
        self.timestamp = timestamp[-1]
        self.l2g_r_mat = l2g_r_mat
        self.l2g_t = l2g_t

        # TODO critical / refactor remove loop (always 1 frame?)
        for i in range(num_frame):
            # TODO refactor / hacky: points is None if no lidar is use -> remove dependency to load points here
            if points:
                points_single = [p_[i] for p_ in points]
            else:
                points_single = None
            img_single = torch.stack([img_[i] for img_ in img], dim=0)
            # TODO changed
            if radar:
                radar_single = torch.stack(
                    [radar_[i] for radar_ in radar], dim=0)
            else:
                radar_single = None
            img_metas_single = deepcopy(img_metas)
            # [0] bs=1
            for k in list(img_metas_single[0].keys()):
                img_metas_single[0][k] = img_metas_single[0][k][i]

            frame_res = self._inference_single(points_single, img_single,
                                               radar_single,
                                               img_metas_single,
                                               track_instances,
                                               l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                                               time_delta,
                                               first_frame=is_first_frame)
            track_instances = frame_res['track_instances']

        active_instances = self.query_interact._select_active_tracks(
            dict(track_instances=track_instances))

        # store tracks for next frame
        self.test_track_instances = track_instances

        results = self._active_instances2results(active_instances, img_metas)
        return results

    def _active_instances2results(self, active_instances, img_metas):
        '''
        Outs:
        active_instances. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - track_ids
            - tracking_score
        '''
        # filter out sleep querys
        active_idxes = (active_instances.scores >=
                        self.track_base.filter_score_thresh)
        active_instances = active_instances[active_idxes]
        if active_instances.pred_logits.numel() == 0:
            return [None]
        bbox_dict = dict(
            cls_scores=active_instances.pred_logits,
            bbox_preds=active_instances.pred_boxes,
            track_scores=active_instances.scores,
            obj_idxes=active_instances.obj_idxes,
        )
        bboxes_dict = self.bbox_coder.decode(bbox_dict)[0]

        bboxes = bboxes_dict['bboxes']
        # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

        # drop the turn rate here since nusc excpets bbox: xyz, wlh , rot, velx, vely
        bboxes = bboxes[..., 0:9]

        bboxes = img_metas[0]['box_type_3d'][0](bboxes, 9)
        labels = bboxes_dict['labels']
        scores = bboxes_dict['scores']

        track_scores = bboxes_dict['track_scores']
        obj_idxes = bboxes_dict['obj_idxes']
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            track_ids=obj_idxes.cpu(),
        )

        return [result_dict]
