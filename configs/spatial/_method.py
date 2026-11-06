# Copyright (c) 2026 Wuhan University, Binglin Hao. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for this release from DAFormer/HRDA/MIC configurations; see LICENSES.md.

"""Shared Spatial settings with archived segmentation-model topology."""

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/uda/dacs_a999_fdthings.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
]

# The four-channel head and decoder topology match the archived models.
# Channel 0 is concentration; channels 1-3 are directional components.
model = dict(decode_head=dict(loss_decode=[
    dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    dict(type='BerHuLoss', loss_weight=0.5,
         channel_weights=[1.0, 0.33, 0.33, 0.33]),
]))

uda = dict(
    use_dual_auxiliary_weight=True,
    concentration_auxiliary_checkpoint=(
        'pretrained/gta_concentration_aux_512.pth'),
    direction_auxiliary_checkpoint='pretrained/gta_direction_aux_512.pth',
    auxiliary_input_size=(512, 512),
    geometry_crop_size=(512, 512),
    geometry_library_path='spatial_extractor/libgeometry_descriptor_area50_512.so',
    geometry_min_component_area=50,
    # Zero-based local_iter: the 36,001st update first applies weighting.
    pseudo_weight_adjust_start_iter=36000,
    pseudo_weight_alpha=1.1,
    pseudo_weight_beta=0.8,
)

concentration_auxiliary = dict(
    type='AuxiliaryEncoderDecoder',
    pretrained=None,
    backbone=dict(type='mit_b5', style='pytorch', in_chans=1),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='aspp', sep=True, dilations=(1, 6, 12, 18),
                pool=False, act_cfg=dict(type='ReLU'),
                norm_cfg=dict(type='BN', requires_grad=True))),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

direction_auxiliary = dict(
    concentration_auxiliary,
    backbone=dict(type='mit_b5', style='pytorch', in_chans=3),
)

optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(custom_keys=dict(
        head=dict(lr_mult=10.0),
        pos_block=dict(decay_mult=0.0),
        norm=dict(decay_mult=0.0))))
runner = dict(type='IterBasedRunner', max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU')

