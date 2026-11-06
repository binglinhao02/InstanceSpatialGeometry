# Copyright (c) 2026 Wuhan University, Binglin Hao. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Spatial project: channel-aware BerHu loss for geometry supervision.

import math
import torch
import torch.nn as nn
from ..builder import LOSSES


@LOSSES.register_module()
class BerHuLoss(nn.Module):
    """
    Vectorized, channel-aware BerHu (Reverse Huber) loss.

    Features:
        - Supports angular channels with periodic wrap-around (radians).
        - Supports per-channel weights (channel_weights).
        - mmseg-compatible: reduction + loss_weight.
        - Uses semantic_seg + optional imagemask.

    Notes:
        - angle_channel_indices: channels treated as angles (in radians).
        - If your data is normalized (0–1 or 0–255), convert back before feeding.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=255,
                 channel_weights=None,
                 angle_channel_indices=None,
                 eps=1e-6):

        super().__init__()
        self.reduction = reduction
        self.loss_weight = float(loss_weight)
        self.ignore_index = ignore_index
        self.eps = float(eps)

        # channel weights (list or None)
        self.channel_weights = None if channel_weights is None \
            else torch.tensor(channel_weights, dtype=torch.float32)

        # angular channels (tuple of ints)
        if angle_channel_indices is None:
            self.angle_inds = tuple()
        else:
            self.angle_inds = tuple(angle_channel_indices)

    # ----------------------------------------------------------
    # Angular difference (radians) with 2π wrap-around
    # ----------------------------------------------------------
    def _angular_diff(self, pred, target):
        """
        Compute minimal absolute angular difference in [-pi, pi].

        pred, target: tensors (radians)
        """
        pred = pred / 255.0 * 2 * math.pi
        target = target / 255.0 * 2 * math.pi
        delta = pred - target
        wrapped = torch.remainder(delta + math.pi, 2 * math.pi) - math.pi
        wrapped = wrapped / (2 * math.pi) * 255.0
        return wrapped.abs()

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------
    def forward(self,
                prediction: torch.Tensor,
                ground_truth: torch.Tensor,
                semantic_seg: torch.Tensor,
                imagemask: torch.Tensor = None):

        if prediction.shape != ground_truth.shape:
            raise ValueError(f"Shape mismatch: {prediction.shape} vs {ground_truth.shape}")

        device = prediction.device
        B, C, H, W = prediction.shape

        # -------------------------------------------
        # 1) Build valid mask
        # -------------------------------------------
        if semantic_seg.dim() == 3:
            semantic_seg = semantic_seg.unsqueeze(1)

        valid_mask = (semantic_seg != self.ignore_index).expand(-1, C, -1, -1)

        if imagemask is not None:
            valid_mask = valid_mask & imagemask.bool()

        # -------------------------------------------
        # 2) Compute elementwise difference
        # -------------------------------------------
        if len(self.angle_inds) == 0:
            diff = (prediction - ground_truth).abs()
        else:
            ang_mask = torch.zeros((1, C, 1, 1), dtype=torch.bool, device=device)
            for i in self.angle_inds:
                ang_mask[0, i, 0, 0] = True
            ang_mask = ang_mask.expand(B, C, H, W)

            normal = (prediction - ground_truth).abs()
            wrapped = self._angular_diff(prediction, ground_truth)

            diff = torch.where(ang_mask, wrapped, normal)

        # -------------------------------------------
        # 3) Per-channel max diff (only valid pixels)
        # -------------------------------------------
        diff_for_max = diff.masked_fill(~valid_mask, float('-inf'))
        max_per_channel = torch.amax(diff_for_max, dim=(0, 2, 3))
        max_per_channel = torch.where(
            torch.isfinite(max_per_channel),
            max_per_channel,
            torch.zeros_like(max_per_channel)
        )

        c_b = (0.2 * max_per_channel).view(1, C, 1, 1)

        # -------------------------------------------
        # 4) BerHu loss
        # -------------------------------------------
        linear_mask = diff <= c_b
        nonlinear = (diff * diff + c_b * c_b) / ((2 * c_b).clamp_min(self.eps))

        loss_map = torch.where(linear_mask, diff, nonlinear)
        loss_map = loss_map * valid_mask.float()

        # -------------------------------------------
        # 5) Channel weights
        # -------------------------------------------
        if self.channel_weights is not None:
            if self.channel_weights.numel() != C:
                raise ValueError("channel_weights length must match C")
            chw = self.channel_weights.to(device).view(1, C, 1, 1)
        else:
            chw = torch.ones((1, C, 1, 1), device=device)

        weighted_loss_map = loss_map * chw
        weighted_valid_map = valid_mask.float() * chw

        # -------------------------------------------
        # 6) Reduction
        # -------------------------------------------
        denom = weighted_valid_map.sum()

        if denom.item() == 0:
            return torch.tensor(0., device=device, dtype=prediction.dtype)

        if self.reduction == 'sum':
            out = weighted_loss_map.sum()
        elif self.reduction == 'mean':
            out = weighted_loss_map.sum() / denom.clamp_min(1e-12)
        elif self.reduction == 'none':
            return weighted_loss_map
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        return self.loss_weight * out
