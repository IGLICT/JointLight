# Configurable relighting

Run from the project directory after installing the environment and weights:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/relight.sh --config configs/inference.json
```

For eight GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/relight.sh --config configs/inference.json
```

The default config explicitly selects seven videos and four HDRIs, producing 28
combinations. Only `dry_field_1k` rotates clockwise. Each worker processes one
combination at a time and runs geometry, shading, diffusion and comparison in
sequence. With no arguments, `scripts/relight.sh` uses this default config.
`scripts/relight_examples.sh` is a shortcut to the same configurable runner.

Inspect the complete plan without loading models or writing files:

```bash
python run_relighting.py --config configs/inference.json --gpus 0,1 --dry_run
```

## Custom config

Copy `configs/inference.json` and edit its lists. All selected videos are paired
with all selected HDRIs. Use one entry in each list for a single run.

```json
{
  "path_root": "..",
  "defaults": {
    "prompt": "A person standing in a room",
    "output_dir": "outputs/custom",
    "num_frames": 81,
    "shading_timestep_init": 0.6,
    "input_timestep_init": 1.0,
    "shading_exposure_ev": "auto",
    "shading_spp": 64,
    "no_comparison": false
  },
  "videos": [
    {"path": "examples/videos/video_0.mp4", "prompt": "A boy sitting indoors"}
  ],
  "hdris": [
    {"path": "examples/hdris/dry_field_1k.hdr", "rotate": "CW", "envmap_angle": 0.0},
    {"name": "dry_field_fixed", "path": "examples/hdris/dry_field_1k.hdr", "rotate": "NONE"}
  ]
}
```

`path_root` resolves relative to the config file's directory. Every input,
model and output path resolves relative to that root, independently of the shell
working directory. In the default `configs/inference.json`, `".."` is the project
root. Absolute paths and `~` are supported. If you move a config elsewhere,
adjust `path_root` accordingly. MoGe model identifiers are kept as identifiers;
local `.pt`/`.pth` paths resolve against `path_root`.

| Section | Supported fields |
| --- | --- |
| `light_comparison` | Boolean; after all jobs finish, compose original + selected relit videos with synchronized chrome balls. Enabled in the default config; omitted means disabled. |
| `defaults` | Shared single-run options: checkpoint, model_config, tokenizer_path, moge_checkpoint, prompt, device, height, width, num_frames, cfg_scale, seed, noise strengths, rotate, envmap_angle, threshold, exposure, sample count, output_dir, no_comparison, etc. See `python run_relit_pipeline.py --help`. |
| `videos` | Required `path`; optional `name`, `prompt`, `height`, `width`, `input_video_path`. |
| `hdris` | Required `path`; optional `name`, `rotate`, `envmap_angle`, `shading_exposure_ev`, `shading_spp`. |

Entry values override `defaults`; omitted options use the single-run CLI defaults.
JSON booleans must be `true`/`false`. Unknown keys and duplicate names are rejected
before launching jobs. `video_path`, `envmap_path`, `output_subdir` and `dry_run`
are managed by the runner and cannot appear in `defaults`.

`rotate` is `NONE`, `CW` or `CCW`; `envmap_angle` sets the initial yaw in degrees.
Rotation uses the existing full-turn schedule across the requested frame count.
Provide both `height` and `width`, or omit both for automatic portrait/landscape
selection. Dimensions must be divisible by 32; frame count must be `4n+1`.
Short inputs are padded with their last frame.

`--gpus` overrides `CUDA_VISIBLE_DEVICES`. If neither is set, the runner detects
GPU indices with `nvidia-smi`, falling back to GPU 0. The number of simultaneous
jobs is bounded by the GPU count. A failed child process makes the launcher exit
with an error; other workers already running finish their work.

## Outputs

Each run writes `outputs/<video-name>/<hdri-name>/` (under your `output_dir`):

- `inference_config.json`: the selected parameters and resolved paths.
- `assets/`: processed input and HDRI copy.
- `moge_shading/`: rendered shading video and exposure metadata.
- `relit/`: generated RGB and shading videos.
- `comparison.mp4`: input / MoGe shading / predicted shading / relit / envmap.
- `result.json`: exact output paths and lighting settings for later comparisons.

With `light_comparison` enabled, the batch also saves
`outputs/examples/four_lights/video_N_four_lights.mp4` (under your `output_dir`).
Columns follow `hdris` order, with the original video first. Each relit column
has its chrome-ball preview below it, using the rendering yaw and reflection;
preview exposure is fixed across the rotation. Static lights remain static.
Custom configs can select any number of lights (the suffix is `_lights.mp4`
when the count is not four). Chrome-ball previews use the CPU and are cached.

Rebuild comparisons without rerunning inference:

```bash
python run_relighting.py --config configs/inference.json --comparison_only
```

Use the same config as the completed run; all selected results must exist and
have matching processed inputs, dimensions, frame counts and frame rates.

Names default to the input filename stem. Specify unique `name` values when
using the same file with multiple settings. Re-running the same named combination
uses the same output paths; choose a new output directory to retain an earlier run.

The lower-level single-run interface remains available through
`python run_relit_pipeline.py --video_path ...`, including `--skip_moge` for a
previously rendered shading cue. See [pipeline details](RELIGHT_PIPELINE.md).
