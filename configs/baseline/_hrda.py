# Copyright (c) 2026 Wuhan University, Binglin Hao. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for this release from DAFormer/HRDA/MIC configurations; see LICENSES.md.

"""Original HRDA network, 40k baseline schedule."""

_base_ = ['_method.py']

model = dict(
    type='HRDAEncoderDecoder',
    decode_head=dict(type='HRDAHead',
                     single_scale_head='DAFormerHead',
                     attention_classwise=True,
                     hr_loss_weight=0.1),
    scales=[1, 0.5],
    hr_crop_size=[512, 512],
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=True,
    test_cfg=dict(mode='slide', batched_slide=True,
                  stride=[512, 512], crop_size=[1024, 1024]))

