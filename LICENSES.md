# Licenses and third-party notices

Original Spatial contributions and modifications:
Copyright (c) 2026 Wuhan University, Binglin Hao. All rights reserved.

The original Spatial geometry code, release-specific configurations, and
modifications contributed by this project are offered under Apache-2.0; the
full text is in [`LICENSE`](LICENSE). This does **not** replace the licenses
of code inherited from MIC, HRDA, DAFormer, MMSegmentation, or their own
dependencies. File-level notices and the component-specific terms below
remain in force. This attribution applies only to the Spatial contributions,
not to inherited third-party code.

| Component and files | Applicable terms |
| --- | --- |
| MIC semantic-segmentation code, including `mmseg/models/uda/masking_consistency_module.py` and `mmseg/models/uda/teacher_module.py` | [MIC Apache-2.0](resources/license_mic) |
| HRDA code, including `mmseg/models/segmentors/hrda_encoder_decoder.py` and `mmseg/models/decode_heads/hrda_head.py` | [HRDA Apache-2.0](resources/license_hrda) |
| DAFormer code, including `mmseg/models/decode_heads/daformer_head.py`, datasets, and base configurations | [DAFormer Apache-2.0](resources/license_daformer) |
| MMSegmentation-derived code under `mmseg/` and `tools/` unless excepted below | [MMSegmentation Apache-2.0](resources/license_mmseg) |
| `mmseg/models/backbones/mix_transformer.py` and `mmseg/models/decode_heads/segformer_head.py` | [NVIDIA Source Code License for SegFormer](resources/license_segformer); non-commercial research/evaluation restriction |
| `mmseg/models/utils/dacs_transforms.py` and DACS-derived portions of `mmseg/models/uda/dacs.py` | [DACS MIT](resources/license_dacs); `dacs.py` also retains Apache-2.0 upstream portions and Spatial modifications |
| `mmseg/models/uda/advseg.py` and `mmseg/models/uda/fcdiscriminator.py` | [AdaptSegNet notice](resources/license_adaptsegnet); non-commercial research only |
| Advent-derived portions of `mmseg/models/uda/minent.py` | [Advent Apache-2.0](resources/license_advent) |
| Original Spatial implementation in `spatial_extractor/`, `mmseg/models/losses/berhu_loss.py`, and release-specific code in `configs/{baseline,spatial}/` | [Project Apache-2.0](LICENSE), subject to any inherited notices in individual files |
| Spatial-derived `mmseg/models/decode_heads/spatial_*` and `mmseg/models/segmentors/spatial_*` | Project modifications under [Apache-2.0](LICENSE); original DAFormer, HRDA, MIC, and MMSegmentation notices remain applicable |

The upstream [MIC file-to-license inventory](https://github.com/lhoyer/MIC/blob/master/seg/LICENSES.md)
provides the original provenance of its dependencies. The presence of a root
Apache-2.0 `LICENSE` must not be interpreted as permission to use the NVIDIA
or AdaptSegNet components commercially. Dataset and pretrained-weight terms
are separate and are not granted by this repository.
