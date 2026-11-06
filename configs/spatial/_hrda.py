# Copyright (c) 2026 Wuhan University, Binglin Hao. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for this release from DAFormer/HRDA/MIC configurations; see LICENSES.md.

"""Spatial-HRDA architecture and 1024-pixel native geometry extraction."""

_base_ = ['_method.py']

model = dict(
    type='SpatialHRDAEncoderDecoder',
    decode_head=dict(
        type='SpatialHRDAHead',
        single_scale_head='SpatialHead',
        attention_head='DAFormerHead',
        attention_classwise=True,
        hr_loss_weight=0.1),
    scales=[1, 0.5],
    hr_crop_size=[512, 512],
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=True,
    test_cfg=dict(mode='slide', batched_slide=True,
                  stride=[512, 512], crop_size=[1024, 1024]))

uda = dict(
    geometry_crop_size=(1024, 1024),
    geometry_library_path='spatial_extractor/libgeometry_descriptor_area50_1024.so')

