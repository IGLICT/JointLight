# Third-party notices

## Source code

- JointLight retains the Apache-2.0 license and upstream notices from
  [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). See [LICENSE](LICENSE).
- The Wan components derive from [Wan](https://github.com/Wan-Video/Wan2.2).
  Model weights carry the terms published with each model repository.
- [MoGe](https://github.com/microsoft/MoGe), copyright Microsoft Corporation,
  is bundled under `third_party/MoGe`. Its upstream license includes the MIT
  terms for MoGe and Apache-2.0 terms for its DINOv2 components. The complete
  license is retained in [third_party/MoGe/LICENSE](third_party/MoGe/LICENSE).
- [DINOv2](https://github.com/facebookresearch/dinov2), copyright Meta Platforms,
  Inc. and affiliates, is included through MoGe. Its source headers are retained.
- See [MoGe vendoring notes](third_party/MoGe/VENDORED.md) for the source revision,
  compatibility changes and file checksums. Dependencies installed through pip
  keep their respective licenses; they are not relicensed by this project.

## Example media

The maintainer has confirmed public redistribution of the six example videos.
They are demonstration assets, not a grant of the source-code license to the
underlying video content. Training datasets are not distributed.

The four HDRI names correspond to the following Poly Haven assets. Their official
pages publish them under [CC0](https://polyhaven.com/license):

- [Blinds](https://polyhaven.com/a/blinds)
- [Cobblestone Street Night](https://polyhaven.com/a/cobblestone_street_night)
- [Dry Field](https://polyhaven.com/a/dry_field)
- [Park Bench](https://polyhaven.com/a/park_bench)

Model weights are distributed separately from the source code.
