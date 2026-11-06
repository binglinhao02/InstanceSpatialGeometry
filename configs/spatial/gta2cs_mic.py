# Copyright (c) 2026 Wuhan University, Binglin Hao. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for this release from DAFormer/HRDA/MIC configurations; see LICENSES.md.

"""GTA5 to Cityscapes, Spatial + MIC (HRDA backbone)."""

_base_ = [
    '_mic.py',
    '../_base_/datasets/uda_gtaHR_to_cityscapesHR_1024x1024.py',
]

seed = 1
data = dict(
    workers_per_gpu=1,
    train=dict(
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=2.0),
        target=dict(crop_pseudo_margins=[30, 240, 30, 30])))
name = 'gta2cs_mic_spatial'

