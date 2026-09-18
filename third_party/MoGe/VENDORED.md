# MoGe inference library

Source: https://github.com/microsoft/MoGe
Revision: `0286b495230a074aadf1c76cc5c679e943e5d1c6`

This is a vendored source snapshot, not a symlink or Git submodule. Keep it when
copying/distributing JointLight. Only `moge/model`, `moge/utils`, the package
initializer, packaging metadata, upstream README and license are included.
Upstream training, evaluation, demo/CLI scripts, datasets and weights are omitted.
Python source files match this revision except for two compatibility changes in
`moge/model/v2.py`, preserved from the working integration: use
`utils3d.pt.intrinsics_from_focal_center` with tensor-valued principal points, and
`utils3d.pt.depth_map_to_point_map` in place of the old torch APIs.
`SOURCE_FILES.sha256` records the bundled source checksums.

Packaging-only changes: replace `opencv-python` with `opencv-python-headless`,
remove `click`/`gradio` dependencies and remove the `moge` CLI entry point, whose
scripts are not bundled. `utils3d` is pinned to `3fab839f0be9931dac7c8488eb0e1600c236e183` (version 1.3),
matching the working environment and the geometry APIs used by the renderer.
Install from the JointLight root with `python -m pip install -e third_party/MoGe`.
JointLight's rendering integration is in `scripts/generate_shading.py`.
