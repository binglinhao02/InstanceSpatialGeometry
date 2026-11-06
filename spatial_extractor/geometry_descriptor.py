# Copyright (c) 2026 Wuhan University, Binglin Hao. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python bindings for the fixed-crop, four-channel geometry descriptor."""

from ctypes import POINTER, c_float, c_int, cdll
from pathlib import Path

import numpy as np
import torch


class GeometryDescriptorExtractor:
    """Compute geometry maps from segmentation labels and depth maps."""

    channels = 4

    def __init__(self, crop_size, library_path):
        if len(crop_size) != 2 or min(crop_size) <= 0:
            raise ValueError('crop_size must be a positive (height, width) pair')
        self.height, self.width = map(int, crop_size)
        self.pixel_count = self.height * self.width

        library_path = Path(library_path)
        if not library_path.is_file():
            raise FileNotFoundError(f'Native geometry library: {library_path}')
        self._library = cdll.LoadLibrary(str(library_path.resolve()))
        # A shape mismatch would otherwise make the C ABI access past buffers.
        self._library.geometry_descriptor_height.argtypes = ()
        self._library.geometry_descriptor_height.restype = c_int
        self._library.geometry_descriptor_width.argtypes = ()
        self._library.geometry_descriptor_width.restype = c_int
        compiled_size = (self._library.geometry_descriptor_height(),
                         self._library.geometry_descriptor_width())
        if compiled_size != (self.height, self.width):
            raise ValueError(
                f'Library was compiled for {compiled_size}, not '
                f'{(self.height, self.width)}: {library_path}')
        self._library.geometry_descriptor_min_area.argtypes = ()
        self._library.geometry_descriptor_min_area.restype = c_int
        self.min_component_area = self._library.geometry_descriptor_min_area()
        self._compute_native = self._library.compute_geometry_descriptor
        self._compute_native.argtypes = (
            POINTER(c_int),
            *(POINTER(c_float) for _ in range(5)),
        )
        self._compute_native.restype = None

        self._labels_buffer = (c_int * self.pixel_count)()
        self._depth_buffer = (c_float * self.pixel_count)()
        self._output_buffers = tuple(
            (c_float * self.pixel_count)() for _ in range(self.channels))
        self._output_views = tuple(
            np.ctypeslib.as_array(buffer).reshape(self.height, self.width)
            for buffer in self._output_buffers)

    def compute_geometry_map(self, labels, depth):
        """Return one CPU tensor of shape ``[4, height, width]``."""
        result = torch.empty((self.channels, self.height, self.width),
                             dtype=torch.float32)
        self._compute_into(labels, depth, result)
        return result

    def _compute_into(self, labels, depth, result):
        expected_size = (self.height, self.width)
        if tuple(labels.shape) != expected_size:
            raise ValueError(f'Expected labels {expected_size}, got {labels.shape}')
        if tuple(depth.shape) != expected_size:
            raise ValueError(f'Expected depth {expected_size}, got {depth.shape}')

        label_values = labels.detach().cpu().numpy().astype(np.int32,
                                                             copy=False)
        depth_values = depth.detach().cpu().numpy().astype(np.float32,
                                                             copy=False)
        np.copyto(np.ctypeslib.as_array(self._labels_buffer),
                  label_values.ravel())
        np.copyto(np.ctypeslib.as_array(self._depth_buffer),
                  depth_values.ravel())
        del label_values, depth_values

        self._compute_native(self._labels_buffer, self._depth_buffer,
                             *self._output_buffers)
        for channel, values in enumerate(self._output_views):
            result[channel].copy_(torch.from_numpy(values))

    @torch.no_grad()
    def __call__(self, labels, depth):
        """Return a CPU tensor of shape ``[batch, 4, height, width]``."""
        if labels.ndim not in (3, 4) or depth.ndim not in (3, 4):
            raise ValueError('Expected batched labels and depth with 3 or 4 dimensions')
        if labels.shape[0] != depth.shape[0]:
            raise ValueError('Label and depth batch sizes do not match')
        if labels.ndim == 4:
            if labels.shape[1] != 1:
                raise ValueError('Labels must have one channel')
            labels = labels[:, 0]
        if depth.ndim == 4:
            if depth.shape[1] != 1:
                raise ValueError('Depth must have one channel')
            depth = depth[:, 0]
        expected_size = (self.height, self.width)
        if tuple(labels.shape[-2:]) != expected_size:
            raise ValueError(f'Expected labels {expected_size}, got {labels.shape[-2:]}')
        if tuple(depth.shape[-2:]) != expected_size:
            raise ValueError(f'Expected depth {expected_size}, got {depth.shape[-2:]}')

        result = torch.empty((labels.shape[0], self.channels, *expected_size),
                             dtype=torch.float32)
        for index in range(labels.shape[0]):
            self._compute_into(labels[index], depth[index], result[index])
        return result
