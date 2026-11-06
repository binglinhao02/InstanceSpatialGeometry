# Copyright (c) 2026 Wuhan University, Binglin Hao. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for this release from DAFormer/HRDA/MIC configurations; see LICENSES.md.

_base_ = ['_method.py',
          '../_base_/datasets/baseline_gta_to_cityscapes_512x512.py']

seed = 0
uda = dict(pseudo_weight_ignore_top=15,
           pseudo_weight_ignore_bottom=120)
data = dict(train=dict(rare_class_sampling=dict(
    min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))
name = 'gta2cs_daformer'

