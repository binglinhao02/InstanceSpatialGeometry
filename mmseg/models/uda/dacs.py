# Derived from DAFormer/HRDA/MIC semantic-segmentation DACS implementations:
# https://github.com/lhoyer/DAFormer
# https://github.com/lhoyer/HRDA
# https://github.com/lhoyer/MIC/tree/master/seg
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0; see resources/license_mic.
# EMA updates and cross-domain mixing are based on DACS:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License;
# see resources/license_dacs.
# Spatial modifications Copyright (c) 2026 Wuhan University, Binglin Hao.
# Modifications by the Spatial project: four-channel geometry supervision,
# dual-auxiliary pseudo-label weighting, and training-memory management.

"""DACS training with four-channel geometry and dual auxiliary weighting.

The frozen concentration and direction segmentors are training-only.
Inference uses the underlying segmentation model alone.
"""

import random
from copy import deepcopy

import torch
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import (UDA, HRDAEncoderDecoder,
                          SpatialHRDAEncoderDecoder, build_segmentor)
from mmseg.models.segmentors.spatial_hrda_encoder_decoder import crop
from mmseg.models.uda.masking_consistency_module import MaskingConsistencyModule
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (get_class_masks, get_mean_std,
                                                strong_transform)
from mmseg.utils.utils import downscale_label_ratio
from spatial_extractor import GeometryDescriptorExtractor


def refine_pseudo_weight(weight, concentration_label, direction_label,
                         alpha=1.0, beta=1.0):
    """Scale pseudo-label weights by agreement of the two auxiliary models."""
    if (concentration_label.shape != direction_label.shape or
            weight.shape != concentration_label.shape):
        raise ValueError(
            'Pseudo weights and auxiliary labels must have equal sizes')
    if alpha <= 0 or beta <= 0:
        raise ValueError('Pseudo-label weight factors must be positive')
    if alpha != 1.0 or beta != 1.0:
        factor = concentration_label.eq(direction_label).to(dtype=weight.dtype)
        factor.mul_(alpha - beta).add_(beta)
        weight.mul_(factor)
        del factor
    return weight


@torch.no_grad()
def resized_argmax(scores, output_size, channel_chunk=4):
    """Bilinearly resize class scores before argmax with bounded memory."""
    if scores.shape[-2:] == tuple(output_size):
        return scores.argmax(dim=1)
    batch_size = scores.shape[0]
    best_score = scores.new_full((batch_size, *output_size), -float('inf'))
    best_label = torch.zeros((batch_size, *output_size),
                             dtype=torch.long, device=scores.device)
    for first_channel in range(0, scores.shape[1], channel_chunk):
        resized = F.interpolate(
            scores[:, first_channel:first_channel + channel_chunk],
            size=output_size, mode='bilinear', align_corners=False)
        local_score, local_label = resized.max(dim=1)
        improved = local_score > best_score
        best_score = torch.where(improved, local_score, best_score)
        best_label = torch.where(improved, local_label + first_channel,
                                 best_label)
        del resized, local_score, local_label, improved
    del best_score
    return best_label


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.local_iter = 0
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        assert cfg['mix'] == 'class'

        self.ema_model = build_segmentor(deepcopy(cfg['model']))
        self.ema_model.requires_grad_(False)
        self.imnet_model = (build_segmentor(deepcopy(cfg['model']))
                            if self.enable_fdist else None)
        if self.imnet_model is not None:
            self.imnet_model.requires_grad_(False)

        self.use_spatial = cfg['model']['type'] in (
            'SpatialEncoderDecoder', 'SpatialHRDAEncoderDecoder')
        self.geometry_extractor = (
            GeometryDescriptorExtractor(tuple(cfg['geometry_crop_size']),
                                        cfg['geometry_library_path'])
            if self.use_spatial else None)
        if self.geometry_extractor is not None:
            expected_area = cfg.get('geometry_min_component_area')
            if (expected_area is not None and
                    self.geometry_extractor.min_component_area !=
                    int(expected_area)):
                raise ValueError(
                    'Native geometry area threshold does not match config: '
                    f'{self.geometry_extractor.min_component_area} != '
                    f'{expected_area}')
        self.concentration_auxiliary = None
        self.direction_auxiliary = None
        self.use_dual_auxiliary_weight = bool(
            cfg.get('use_dual_auxiliary_weight', False))
        self.concentration_auxiliary_checkpoint = cfg.get(
            'concentration_auxiliary_checkpoint')
        self.direction_auxiliary_checkpoint = cfg.get(
            'direction_auxiliary_checkpoint')
        self.auxiliary_input_size = tuple(
            cfg.get('auxiliary_input_size', (512, 512)))
        self.pseudo_weight_alpha = float(cfg.get('pseudo_weight_alpha', 1.0))
        self.pseudo_weight_beta = float(cfg.get('pseudo_weight_beta', 1.0))
        self.pseudo_weight_adjust_start_iter = int(
            cfg.get('pseudo_weight_adjust_start_iter', 0))
        if self.use_dual_auxiliary_weight and not self.use_spatial:
            raise ValueError('Dual-auxiliary pseudo weighting requires Spatial')
        if self.use_dual_auxiliary_weight and not all((
                self.concentration_auxiliary_checkpoint,
                self.direction_auxiliary_checkpoint)):
            raise ValueError('Both auxiliary checkpoints must be configured')
        if self.pseudo_weight_adjust_start_iter < 0:
            raise ValueError('Pseudo-weight start iteration must be nonnegative')
        if self.pseudo_weight_alpha <= 0 or self.pseudo_weight_beta <= 0:
            raise ValueError('Pseudo-weight factors must be positive')
        self.mic = (MaskingConsistencyModule(require_teacher=False, cfg=cfg)
                    if cfg.get('mask_mode') is not None else None)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    @torch.no_grad()
    def _update_teacher(self):
        momentum = (0 if self.local_iter == 0 else
                    min(1 - 1 / (self.local_iter + 1), self.alpha))
        for teacher, student in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            teacher.data.mul_(momentum).add_(student.data,
                                              alpha=1 - momentum)
        if self.local_iter == 0:
            for teacher, student in zip(self.get_ema_model().buffers(),
                                        self.get_model().buffers()):
                teacher.copy_(student)

    def train_step(self, data_batch, optimizer, **kwargs):
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()
        log_vars.pop('loss', None)
        return dict(log_vars=log_vars,
                    num_samples=len(data_batch['img_metas']))

    def _feature_distance(self, img, gt, features):
        with torch.no_grad():
            self.get_imnet_model().eval()
            reference = self.get_imnet_model().extract_feat(img)
        model = self.get_model()
        if (isinstance(model, (HRDAEncoderDecoder,
                               SpatialHRDAEncoderDecoder)) and
                model.feature_scale in model.feature_scale_all_strs):
            pairs = zip([f[-1] for f in features],
                        [f[-1] for f in reference])
        else:
            pairs = [(features[-1], reference[-1])]
        distance = 0
        for scale, (student, frozen) in enumerate(pairs):
            pixel_distance = torch.norm(student - frozen.detach(),
                                        dim=1, p=2)
            if self.fdist_classes is not None:
                label = gt
                if (isinstance(model, (HRDAEncoderDecoder,
                                       SpatialHRDAEncoderDecoder)) and
                        scale in model.last_train_crop_box):
                    label = crop(label, model.last_train_crop_box[scale])
                factor = label.shape[-1] // student.shape[-1]
                label = downscale_label_ratio(
                    label, factor, self.fdist_scale_min_ratio,
                    self.num_classes, 255).long()
                classes = torch.tensor(self.fdist_classes, device=gt.device)
                mask = torch.any(label[..., None] == classes, dim=-1)
                pixel_distance = pixel_distance[mask.squeeze(1)]
            distance = distance + pixel_distance.mean()
        loss, logs = self._parse_losses(
            {'loss_imnet_feat_dist': self.fdist_lambda * distance})
        logs.pop('loss', None)
        return loss, logs

    @torch.no_grad()
    def _predict_auxiliary_labels(self, geometry_map, metas):
        """Run the frozen 1- and 3-channel models at auxiliary resolution."""
        if (self.concentration_auxiliary is None or
                self.direction_auxiliary is None):
            raise RuntimeError('Both frozen geometry auxiliaries are required')
        if geometry_map.shape[1] != 4:
            raise ValueError('Dual auxiliaries require a four-channel map')
        output_size = geometry_map.shape[-2:]
        predictions = []
        for channels, auxiliary in ((slice(0, 1), self.concentration_auxiliary),
                                    (slice(1, 4), self.direction_auxiliary)):
            auxiliary.eval()
            auxiliary = (auxiliary.module if hasattr(auxiliary, 'module')
                         else auxiliary)
            auxiliary_input = geometry_map[:, channels]
            if output_size != self.auxiliary_input_size:
                auxiliary_input = F.interpolate(
                    auxiliary_input, size=self.auxiliary_input_size,
                    mode='bilinear', align_corners=False)
            logits = auxiliary.encode_decode(auxiliary_input, metas)
            if logits.shape[-2:] != output_size:
                # The archived HRDA/MIC auxiliary path resized softmax scores
                # from 512² to 1024² before class selection.
                probabilities = logits.softmax(dim=1)
                del logits
                predictions.append(resized_argmax(probabilities, output_size))
                del probabilities
            else:
                predictions.append(logits.argmax(dim=1))
                del logits
            del auxiliary_input
        return predictions[0], predictions[1]

    def forward_train(self, img, img_metas, gt_semantic_seg,
                      target_img, target_img_metas, depth=None,
                      target_depth=None,
                      rare_class=None, valid_pseudo_mask=None):
        # DACS order: source supervision, EMA pseudo-labels, class mixing,
        # then optional MIC consistency. Geometry follows the same samples.
        self._update_teacher()
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

        means, stds = get_mean_std(img_metas, img.device)
        transform = dict(
            mix=None, color_jitter=random.uniform(0, 1),
            color_jitter_s=self.color_jitter_s,
            color_jitter_p=self.color_jitter_p,
            blur=random.uniform(0, 1) if self.blur else 0,
            mean=means[0].unsqueeze(0), std=stds[0].unsqueeze(0))

        source_geometry = None
        if self.use_spatial:
            if depth is None or target_depth is None:
                raise ValueError('Spatial training requires source and target depth')
            source_geometry = self.geometry_extractor(
                gt_semantic_seg, depth).to(img.device)
            source_losses = self.get_model().forward_train(
                img, img_metas, gt_semantic_seg,
                spatial_map=source_geometry, return_feat=True)
        else:
            source_losses = self.get_model().forward_train(
                img, img_metas, gt_semantic_seg, return_feat=True)
        features = source_losses.pop('features')
        source_loss, source_logs = self._parse_losses(source_losses)
        log_vars = dict(source_logs)
        source_loss.backward(retain_graph=self.enable_fdist)
        del source_loss, source_losses, source_logs
        if self.enable_fdist:
            feat_loss, feat_logs = self._feature_distance(
                img, gt_semantic_seg, features)
            log_vars.update(add_prefix(feat_logs, 'src'))
            feat_loss.backward()
            del feat_loss, feat_logs
        del features

        # Generate labels and confidence without retaining the teacher graph.
        teacher = self.get_ema_model()
        for module in teacher.modules():
            if isinstance(module, (_DropoutNd, DropPath)):
                module.training = False
        with torch.no_grad():
            predict = (teacher.generate_pseudo_label
                       if hasattr(teacher, 'generate_pseudo_label')
                       else teacher.encode_decode)
            logits = predict(target_img, target_img_metas)
            maximum_logit, pseudo_label = logits.max(dim=1)
            confidence = (maximum_logit - logits.logsumexp(dim=1)).exp()
            pseudo_weight = confidence.ge(
                self.pseudo_threshold).float().mean().expand_as(
                    confidence).clone()
            del logits, maximum_logit, confidence

        target_geometry = None
        if self.use_spatial:
            target_geometry = self.geometry_extractor(
                pseudo_label, target_depth).to(img.device)
        # Agreement weighting starts at a zero-based iteration set by config.
        if (self.use_dual_auxiliary_weight and
                self.local_iter >= self.pseudo_weight_adjust_start_iter):
            concentration_label, direction_label = self._predict_auxiliary_labels(
                target_geometry, target_img_metas)
            refine_pseudo_weight(pseudo_weight, concentration_label,
                                 direction_label,
                                 self.pseudo_weight_alpha,
                                 self.pseudo_weight_beta)
            del concentration_label, direction_label
        if self.psweight_ignore_top:
            pseudo_weight[:, :self.psweight_ignore_top] = 0
        if self.psweight_ignore_bottom:
            pseudo_weight[:, -self.psweight_ignore_bottom:] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)

        # Use one class-mix mask for RGB, labels, geometry, and pixel weights.
        mixed_img, mixed_label, mixed_geometry, mixed_weight = [], [], [], []
        mix_masks = get_class_masks(gt_semantic_seg)
        for index in range(img.shape[0]):
            transform['mix'] = mix_masks[index]
            image, label = strong_transform(
                transform, data=torch.stack((img[index], target_img[index])),
                target=torch.stack((gt_semantic_seg[index, 0],
                                    pseudo_label[index])))
            if self.use_spatial:
                _, geometry_sample = strong_transform(
                    transform, target=torch.stack((source_geometry[index],
                                                   target_geometry[index])))
                mixed_geometry.append(geometry_sample)
            _, weight = strong_transform(
                transform, target=torch.stack((torch.ones_like(
                    pseudo_weight[index]), pseudo_weight[index])))
            mixed_img.append(image)
            mixed_label.append(label)
            mixed_weight.append(weight)
        del image, label, weight, mix_masks, transform, means, stds
        if self.use_spatial:
            del geometry_sample, source_geometry
        mixed_img = torch.cat(mixed_img)
        mixed_label = torch.cat(mixed_label)
        mixed_weight = torch.cat(mixed_weight).squeeze(1)
        if self.use_spatial:
            mixed_geometry = torch.cat(mixed_geometry)
            mixed_losses = self.get_model().forward_train(
                mixed_img, img_metas, mixed_label,
                spatial_map=mixed_geometry,
                seg_weight=mixed_weight, return_feat=False)
        else:
            mixed_losses = self.get_model().forward_train(
                mixed_img, img_metas, mixed_label,
                seg_weight=mixed_weight, return_feat=False)
        mixed_losses = add_prefix(mixed_losses, 'mix')
        mixed_loss, mixed_logs = self._parse_losses(mixed_losses)
        log_vars.update(mixed_logs)
        mixed_loss.backward()
        del mixed_img, mixed_label, mixed_geometry, mixed_weight
        del mixed_losses, mixed_loss, mixed_logs

        if self.mic is not None:
            masked_losses = self.mic(
                self.get_model(), img, img_metas, gt_semantic_seg,
                target_img, target_img_metas, valid_pseudo_mask,
                pseudo_label, pseudo_weight, target_geometry)
            masked_losses = add_prefix(masked_losses, 'masked')
            masked_loss, masked_logs = self._parse_losses(masked_losses)
            log_vars.update(masked_logs)
            masked_loss.backward()
            del masked_losses, masked_loss, masked_logs

        del target_geometry, pseudo_label, pseudo_weight
        self.local_iter += 1
        return log_vars
