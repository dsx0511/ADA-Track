# ------------------------------------------------------------------------
# ADA-Track
# Copyright (c) 2024 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from mmdet.models.builder import LOSSES

from .clip_matcher import ClipMatcher
from ...core.structures import Instances

@LOSSES.register_module()
class ParallelMatcher(ClipMatcher):
    def __init__(self,
                 loss_asso=dict(
                 type='FocalLoss',
                 use_sigmoid=True,
                 gamma=1.0,
                 alpha=-1,
                 loss_weight=1.0), 
                 **kwargs):
        super(ParallelMatcher, self).__init__(**kwargs)
        self.loss_asso_config = loss_asso

    def step(self):
        self._step()

    def match_for_track_queries(self, track_instances: Instances, dec_lvl: int):
        gt_instances_i = self.gt_instances[self._current_frame_idx]

        obj_idxes = gt_instances_i.obj_ids
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx,
                             obj_idx in enumerate(obj_idxes_list)}
        outputs_i = {
            'pred_logits': track_instances.pred_logits.unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes.unsqueeze(0),
        }

        for j in range(len(track_instances)):
            obj_id = track_instances.obj_idxes[j].item()
            if obj_id in obj_idx_to_gt_idx:
                track_instances.matched_gt_idxes[j] = obj_idx_to_gt_idx[obj_id]
            else:
                track_instances.matched_gt_idxes[j] = -1

        full_track_idxes = torch.arange(
            len(track_instances), dtype=torch.long).to(track_instances.pred_logits.device)

        matched_indices = torch.stack(
            [full_track_idxes, track_instances.matched_gt_idxes], dim=1).to(
            track_instances.pred_logits.device)

        for loss in self.losses:
            refine_loss = self.get_loss(loss,
                                        outputs=outputs_i,
                                        gt_instances=[gt_instances_i],
                                        indices=[
                                            (matched_indices[:, 0], matched_indices[:, 1])],
                                        )
        
            self.losses_dict.update(
                    {'frame_{}_track_{}_{}'.format(self._current_frame_idx, key, dec_lvl): value 
                    for key, value in refine_loss.items()})

        return track_instances

    def match_for_detection_queries(self, detection_instances: Instances, dec_lvl: int):
        gt_instances_i = self.gt_instances[self._current_frame_idx]

        outputs_i = {
            'pred_logits': detection_instances.pred_logits.unsqueeze(0),
            'pred_boxes': detection_instances.pred_boxes.unsqueeze(0),
        }

        # [num_new_matched, 2]
        pred_idx, gt_idx = self.matcher.assign(
            outputs_i['pred_boxes'][0], outputs_i['pred_logits'][0],
            gt_instances_i.boxes, gt_instances_i.labels)
        
        matched_indices = None
        if pred_idx is not None:
            detection_instances.obj_idxes[pred_idx] = gt_instances_i.obj_ids[gt_idx].long()
            detection_instances.matched_gt_idxes[pred_idx] = gt_idx

            matched_indices = torch.stack([pred_idx, gt_idx], dim=1)

            for loss in self.losses:
                detetecion_loss = self.get_loss(loss,
                                                outputs=outputs_i,
                                                gt_instances=[gt_instances_i],
                                                indices=[
                                                    (matched_indices[:, 0], matched_indices[:, 1])],
                                                )
                
                self.losses_dict.update(
                    {'frame_{}_det_{}_{}'.format(self._current_frame_idx, key, dec_lvl): value
                        for key, value in detetecion_loss.items()})

        return detection_instances
    
    def match_for_association(self, detection_instances, track_instances, 
                              affinity, edge_index_cross, dec_lvl):
        # Generate target of association for each edge
        det_gt = detection_instances.obj_idxes
        track_gt = track_instances.obj_idxes

        det_gt_scattered = det_gt[edge_index_cross[1, :]]
        track_gt_scattered = track_gt[edge_index_cross[0, :]]

        valid_det = (det_gt_scattered >= 0)
        valid_track = (track_gt_scattered >= 0)
        valid_mask = torch.logical_and(valid_track, valid_det).float()

        target = torch.eq(track_gt_scattered, det_gt_scattered)
        target = torch.logical_and(target, valid_mask).float().unsqueeze(1)

        # Compute loss
        # TODO: Currently the association loss is computed only if edges exist
        association_loss = torch.zeros(1, dtype=torch.float, device=affinity.device)
        if affinity.size(0) != 0:
            association_loss += self.loss_asso_config['loss_weight'] * sigmoid_focal_loss(
                affinity, target, alpha=self.loss_asso_config['alpha'], 
                gamma=self.loss_asso_config['gamma'], reduction='mean'
            )

        self.losses_dict.update(
                        {'frame_{}_asso_loss_{}'.format(self._current_frame_idx, dec_lvl): association_loss})
        
    def match_for_association_with_scene_tioken(self, detection_instances, track_instances, 
                              affinity, edge_index_cross, dec_lvl):
        # Generate target of association for each edge
        det_gt = detection_instances.obj_idxes
        track_gt = track_instances.obj_idxes

        # obj_idxes = -3 means it is a bullshit token
        track_gt = torch.cat([-3 * torch.ones(1, dtype=track_gt.dtype, device=track_gt.device), track_gt])

        det_gt_scattered = det_gt[edge_index_cross[1, :]]
        track_gt_scattered = track_gt[edge_index_cross[0, :]]

        valid_det_scattered = (det_gt_scattered >= 0)
        valid_track_scattered = (track_gt_scattered >= 0)
        valid_mask = torch.logical_and(valid_track_scattered, valid_det_scattered).float()

        target = torch.eq(track_gt_scattered, det_gt_scattered)
        target = torch.logical_and(target, valid_mask)

        # Find detections with an obj idx but is unmatched with a track
        valid_det = (det_gt >= 0)
        valid_det_idx = valid_det.nonzero().squeeze(1)
        matched_det_idx = edge_index_cross[1, target]

        matched_det_mask = torch.isin(valid_det_idx, matched_det_idx)
        unmatched_det_idx = valid_det_idx[~matched_det_mask]

        unmatched_det_scattered = torch.isin(edge_index_cross[1, :], unmatched_det_idx)
        is_scene_token = (track_gt_scattered == -3)
        target_scene_token = torch.logical_and(unmatched_det_scattered, is_scene_token)

        target = torch.logical_or(target, target_scene_token)

        # print('#' * 30)
        # print('num track origin', len(track_instances))
        # print('num det', len(detection_instances))
        # print('matched_det_idx', matched_det_idx.size())
        # print('valid_det_idx', valid_det.count_nonzero())
        # print('unmatched_det_scattered', unmatched_det_scattered.count_nonzero())
        # print('target_scene_token', target_scene_token.count_nonzero())
        # print('final_target', final_target.count_nonzero())
        # print('#' * 30)
        # assert valid_det.count_nonzero() == final_target.count_nonzero()

        target = target.float().unsqueeze(1)
        
        # Compute loss
        # TODO: Currently the association loss is computed only if edges exist
        association_loss = torch.zeros(1, dtype=torch.float, device=affinity.device)
        if affinity.size(0) != 0:
            association_loss += self.loss_asso_config['loss_weight'] * sigmoid_focal_loss(
                affinity, target, alpha=self.loss_asso_config['alpha'], 
                gamma=self.loss_asso_config['gamma'], reduction='mean'
            )

            if 'softmax_loss_weight' in self.loss_asso_config:
                target = target.reshape([len(track_instances) + 1, len(detection_instances)])
                affinity = affinity.reshape([len(track_instances) + 1, len(detection_instances)])

                softmax_asso_loss = self.loss_asso_config['loss_weight'] * F.cross_entropy(affinity.permute(1, 0), target.permute(1, 0))
                self.losses_dict.update(
                        {'frame_{}_asso_loss_softmax_{}'.format(self._current_frame_idx, dec_lvl): softmax_asso_loss})

        self.losses_dict.update(
                        {'frame_{}_asso_loss_{}'.format(self._current_frame_idx, dec_lvl): association_loss})