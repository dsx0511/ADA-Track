# ------------------------------------------------------------------------
# ADA-Track
# Copyright (c) 2024 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) 2018-2019 Open-MMLab. All Rights Reserved.
# ------------------------------------------------------------------------

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import MultiheadAttention

@ATTENTION.register_module()
class MaskedDetectionAndTrackSelfAttention(MultiheadAttention):
    """
    Allows self attention inside track queries and detection queries independently
    as well as cross attention from track to detection queries.
    """

    def __init__(self, block_det_to_track, block_track_to_det, **kwargs):
        super(MaskedDetectionAndTrackSelfAttention, self).__init__(**kwargs)
        self.block_det_to_track = block_det_to_track
        self.block_track_to_det = block_track_to_det

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                track_mask=None,
                **kwargs):

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # # mask: Must be of shape (L,S), L is the target sequence length,
        # # and S is the source sequence length. 
        attn_mask = torch.zeros(query.size(0), query.size(0), 
                          dtype=torch.bool, device=query.device)

        if track_mask is not None:
            if self.block_det_to_track:
                attn_mask[:track_mask.sum(), track_mask.sum():] = 1
            if self.block_track_to_det:
                attn_mask[track_mask.sum():, :track_mask.sum()] = 1
        
        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]
        
        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))