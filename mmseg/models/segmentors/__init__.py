# Spatial modifications Copyright (c) 2026 Wuhan University, Binglin Hao.
# Modified for the Spatial project release; upstream notices and licenses remain in force.
# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add HRDAEncoderDecoder

from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder
from .hrda_encoder_decoder import HRDAEncoderDecoder
from .spatial_encoder_decoder import SpatialEncoderDecoder
from .spatial_hrda_encoder_decoder import SpatialHRDAEncoderDecoder
from .auxiliary_encoder_decoder import AuxiliaryEncoderDecoder

__all__ = ['BaseSegmentor', 'EncoderDecoder', 'HRDAEncoderDecoder',
           'SpatialEncoderDecoder', 'SpatialHRDAEncoderDecoder',
           'AuxiliaryEncoderDecoder']
