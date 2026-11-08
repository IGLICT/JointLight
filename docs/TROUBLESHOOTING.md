# Installation and inference troubleshooting

## Missing model or video modules

Earlier commits omitted `diffsynth/models/`, `diffsynth/data/`, and
`third_party/MoGe/moge/model/dinov2/models/`. Unanchored `models/` and `data/`
Git ignore rules matched these source directories as well as runtime assets.
Version 0.1.1 includes them and limits those rules to the repository root.

Update the checkout and run `python scripts/check_release.py` before installing.
Do not fill these directories with arbitrary versions of DiffSynth or MoGe:
matching import names does not establish checkpoint or forward-pass compatibility.
Create a new conda environment and follow the README's constrained installation.

If imports still resolve to another checkout, inspect their locations:

```bash
python - <<'PY'
import sys
import diffsynth
import moge
print(sys.executable)
print(diffsynth.__file__)
print(moge.__file__)
PY
```

`diffsynth` must resolve to this checkout and `moge` to its `third_party/MoGe`.
The full README smoke check also loads MoGe v2 and initializes the CUDA renderer.

## Predicted shading resembles colored RGB

Check the raw `_shading.mp4` file, not only the comparison layout. The pipeline
returns RGB and shading separately, decoding the first and second temporal
latent segments respectively. The comparison script must receive distinct files.

Use the JointLight checkpoint, not only the Wan base weights. Verify the release
checkpoint with `sha256sum -c CHECKPOINTS.sha256`. The inference entry point
creates the modality identifier before loading the full checkpoint with
`strict=True`; preserve that order. Calling `.init()` after loading would reset
its final layer. Do not suppress missing or unexpected keys with `strict=False`.
A successful strict load confirms parameter compatibility, not every detail of
an independently modified forward pass.

The model uses modality identifiers 0/1/2 for RGB/shading/source conditioning.
Check those identifiers, their modulation, latent ordering and separate decoding
if adapting the pipeline. These are diagnostic checks, not a confirmed cause
of every report of similar-looking outputs.

## Shading adherence versus output quality

`--shading_timestep_init` selects the shading noise initialization and denoising
start; it is not a linear guidance weight. Lower values retain more of the cue,
including reconstruction artifacts; higher values permit more regeneration.
The README full-pipeline example uses 0.6, while the direct `inference.py` CLI
default is 1.0. Pass the value explicitly when comparing entry points.

First reproduce a single video/HDRI pair with the README parameters. Keep seed,
resolution, frame count, prompt and weights fixed while changing one setting.
Source and shading must have matching crops and temporal alignment. Also check
HDRI yaw, exposure, highlight compression and color conversion; the rendering
convention changes the model input. A single successful run does not establish
paper-level visual equivalence.

For an issue report, include the Git commit, full command, environment versions,
checkpoint SHA-256, loading errors, any local modifications, and raw RGB/shading
outputs or aligned frames. This makes implementation mismatches distinguishable
from conditioning and parameter effects.
