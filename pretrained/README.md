# Required weights

This code-only distribution does not bundle model weights. Place the following
files here before training:

- `mit_b5.pth`: SegFormer MiT-B5 backbone initialization.
- `gta_concentration_aux_512.pth`: frozen one-channel geometry auxiliary.
- `gta_direction_aux_512.pth`: frozen three-channel geometry auxiliary.

The two geometry auxiliaries are needed only for Spatial training with
`use_dual_auxiliary_weight=True`; segmentation-only inference needs only the
trained segmentation checkpoint passed to `test.sh`. Use Git LFS or a separate
model-hosting service if distributing large weight files later. Check the
redistribution terms of any third-party pretrained weights first.
