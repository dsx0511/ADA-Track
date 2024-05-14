# ------------------------------------------------------------------------
# ADA-Track
# Copyright (c) 2024 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MUTR3D (https://github.com/a1600012888/MUTR3D)
# Copyright (c) 2022 MARS Lab. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-model/MOTR/)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
from copy import deepcopy

from plugin.track.core.structures import Instances
from plugin.track.models.trackers.utils import greedy_assignment

class RuntimeTrackerBase(object):
    # code from https://github.com/megvii-model/MOTR/blob/main/models/motr.py#L303
    def __init__(self, score_thresh=0.7, filter_score_thresh=0.6, miss_tolerance=5, 
                 affinity_thresh=0.2, hungarian=True):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.affinity_thresh = affinity_thresh # TODO: move this parameter to config
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0
        self.hungarian = hungarian

    def clear(self):
        self.max_obj_id = 0

    def init_tracks(self, track_instances: Instances):
        for i in range(len(track_instances)):
            if track_instances.scores[i] >= self.score_thresh:
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1

    def _dets_tracks_matching(self, affinity, class_valid_mask, num_dets, num_tracks, det_scores):
        threshold_mask = affinity > self.affinity_thresh
        valid_mask = torch.logical_and(class_valid_mask, threshold_mask)

        if self.hungarian:
            det_score_mask = det_scores > 0.2
            det_score_mask = det_score_mask.repeat(num_tracks, 1)
            valid_mask = torch.logical_and(valid_mask, det_score_mask)

        invalid_mask = torch.logical_not(valid_mask)

        cost = - affinity + 1e18 * invalid_mask
        cost[cost > 1e16] = 1e18
        # row_ind: index of detections (current frame)
        # col_ind: index of tracks (last frames)
        # cost[row_ind, col_ind] = c[i, j]
        if self.hungarian:
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        else:
            match_ind = greedy_assignment(deepcopy(cost.detach().cpu().numpy()), det_scores)
            row_ind = match_ind[:, 0]
            col_ind = match_ind[:, 1]

        unmatched_tracks = [t for t in range(num_tracks) if not (t in row_ind)]
        unmatched_dets = [d for d in range(num_dets) if not (d in col_ind)]
    
        matches = []
        if self.hungarian:
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] > 1e16:
                    unmatched_dets.append(j)
                    unmatched_tracks.append(i)
                else:
                    matches.append([i, j])
        else:
            for i, j in zip(row_ind, col_ind):
                matches.append([i, j])
        
        assert len(unmatched_dets) + len(matches) == num_dets
        assert len(unmatched_tracks) + len(matches) == num_tracks
        matches = np.array(matches).reshape(-1, 2)
        
        return matches, unmatched_dets, unmatched_tracks
        
    def update_using_asso_score(self, 
                                detection_instances: Instances,
                                track_instances: Instances,
                                edge_index_cross,
                                affinity,
                                embed_dims,
                                alpha):

        num_dets = len(detection_instances)
        num_tracks = len(track_instances)

        adj_dense = torch.zeros([num_tracks, num_dets],
                           dtype=torch.bool, device=edge_index_cross.device)
        affinity_dense = torch.zeros([num_tracks, num_dets],
                                      dtype=affinity.dtype, device=edge_index_cross.device)

        adj_dense[edge_index_cross[0, :], edge_index_cross[1, :]] = 1
        affinity_dense[edge_index_cross[0, :], edge_index_cross[1, :]] = affinity

        match, unmat_det, unmat_trk = self._dets_tracks_matching(
            affinity_dense, adj_dense, num_dets, num_tracks, detection_instances.scores)
        
        # TODO: Investigate how to merge queries, same as 
        # `LatentAssociationMUTRCamTracker._associate_track_instances()`
        if alpha is None:
            track_instances.output_embedding[match[:, 0]] += detection_instances.output_embedding[match[:, 1]]
        else:
            track_instances.output_embedding[match[:, 0]] = \
                alpha * track_instances.output_embedding[match[:, 0]] + \
                (1 - alpha) * detection_instances.output_embedding[match[:, 1]]

        track_instances.query = torch.cat([track_instances.query[..., 0:embed_dims], 
                                           track_instances.output_embedding], dim=-1)
        track_instances.ref_pts[match[:, 0]] = detection_instances.ref_pts[match[:, 1]]
        track_instances.pred_logits[match[:, 0]] = detection_instances.pred_logits[match[:, 1]]
        track_instances.pred_boxes[match[:, 0]] = detection_instances.pred_boxes[match[:, 1]]

        track_instances.scores[match[:, 0]] = detection_instances.scores[match[:, 1]]

        # print('obj_idxes of matched instances: ', track_instances.obj_idxes[match[:, 1]])

        # Init new detections
        init_track_instances = detection_instances[unmat_det]
        for i in range(len(init_track_instances)):
            if init_track_instances.scores[i] >= self.score_thresh:
                init_track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1

        # Handle inactivate tracks
        for i in unmat_trk:
            # sleep time ++
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # mark deaded tracklets: Set the obj_id to -1.
                    # TODO: remove it by following functions
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1
        
        new_track_instances = Instances.cat([track_instances, init_track_instances])

        return new_track_instances