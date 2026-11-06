# Copyright (c) 2026 Wuhan University, Binglin Hao. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for this release from DAFormer/HRDA/MIC configurations; see LICENSES.md.

"""Original DAFormer DACS training recipe, without geometry."""

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/uda/dacs_a999_fdthings.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
]

model = dict(
    type='EncoderDecoder',
    decode_head=dict(
        type='DAFormerHead',
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False,
                          loss_weight=1.0)]))
uda = dict(mask_mode=None)
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(custom_keys=dict(
        head=dict(lr_mult=10.0),
        pos_block=dict(decay_mult=0.0),
        norm=dict(decay_mult=0.0))))
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=40000,
                         max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')

