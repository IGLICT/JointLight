# Changelog

## 0.1.1

- Restore the missing DiffSynth model and video-data packages, including the
  JointLight modality identifier and Wan model implementations.
- Restore MoGe's DINOv2 model package, matching the bundled source checksums.
- Anchor runtime `models/` and `data/` ignore rules to the repository root so
  nested Python packages are included in future commits.
- Add source and distribution completeness checks to GitHub Actions.
- Document the verified single-video demo and troubleshooting for missing
  packages, checkpoint loading, and RGB/shading output confusion.

The diffusion algorithm, checkpoint format, and inference defaults are unchanged.
This release fixes source distribution completeness; it does not establish the
cause of every quality issue reported in GitHub issue #1.
