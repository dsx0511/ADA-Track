# ------------------------------------------------------------------------
# ADA-Track
# Copyright (c) 2024 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from CenterPoint (https://github.com/tianweiy/CenterPoint)
# Copyright (c) 2020-2021 Tianwei Yin and Xingyi Zhou. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import torch

def greedy_assignment(cost, scores):
    matched_indices = []

    if cost.shape[0] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    
    _, indices = torch.sort(scores, descending=True)
    indices = indices.detach().cpu().numpy()

    # for i, idx in enumerate(indices):
    for idx in indices:
        j = cost[:, idx].argmin()
        if cost[j][idx] < 1e16:
            cost[j, :] = 1e18
            matched_indices.append([j, idx])

    return np.array(matched_indices, np.int32).reshape(-1, 2)