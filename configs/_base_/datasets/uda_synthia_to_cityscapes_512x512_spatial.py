# Spatial modifications Copyright (c) 2026 Wuhan University, Binglin Hao.
# Modified by the Spatial project: source/target geometry-depth data paths.
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '/home/shared/data_daformer/cityscapes'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
synthia_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadDepth'),
    dict(type='Resize', img_scale=(1280, 760)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'depth']),
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepth'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'depth']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='SynthiaDataset',
            data_root='/home/shared/data_daformer/synthia',
            img_dir='RGB',
            ann_dir='GT/LABELS',
            depth_dir='depth_any',
            pipeline=synthia_train_pipeline),
        target=dict(
            type='CityscapesDataset',
            data_root='/home/shared/data_daformer/cityscapes',
            img_dir='leftImg8bit/train',
            depth_dir='depth_any/train',
            pipeline=cityscapes_train_pipeline)),
    val=dict(
        type='CityscapesDataset',
        data_root='/home/shared/data_daformer/cityscapes',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        depth_dir='depth_any/val',
        pipeline=test_pipeline),
    test=dict(
        type='CityscapesDataset',
        data_root='/home/shared/data_daformer/cityscapes',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        depth_dir='depth_any/val',
        pipeline=test_pipeline))
