# Video + environment map → relit video

`run_relit_pipeline.py` is the complete inference entry point. It adapts the original
launcher and uses the bundled `scripts/generate_shading.py`, extracted from the
project's custom `MoGe/moge/scripts/moge_mitsuba.py`. A vanilla MoGe installation
provides the geometry model; it does not need to contain our custom rendering script.

## Dependencies

Use the unified `jointlight` environment described in the
[README installation checklist](../README.md#installation-from-scratch).
The checklist creates an empty Python 3.10 environment, installs CUDA PyTorch,
FFmpeg and the local `third_party/MoGe` library, and adds Mitsuba/Dr.Jit alongside
JointLight using the release `constraints.txt`. No pre-existing model environment
is required.
The included MoGe source is an inference-only external library, with two
compatibility changes for the pinned `utils3d` APIs; see
[vendoring notes](../third_party/MoGe/VENDORED.md).

MoGe loads `Ruicheng/moge-2-vitl-normal` by default and may download its weights on
first use. Pass `--moge_checkpoint` for a local file. Download the JointLight
checkpoint into `models/JointLight/` and the Wan2.2 base files into
`models/Wan-AI/Wan2.2-TI2V-5B/` before running inference.

By default, both subprocesses use the active Python (`sys.executable`). Do not
pass `--moge_env` or `--infer_env` when using the unified environment. Separate
Conda environments remain supported explicitly via those flags; `--moge_repo`
can override the installed MoGe library with another source checkout.
`--conda_exe` is only used when an environment override is requested. Both stages
inherit `CUDA_VISIBLE_DEVICES` and execute sequentially, releasing MoGe's GPU
memory before loading the relighting model.

## Run

From the JointLight root, after `conda activate jointlight`:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/relight.sh \
  --video_path data/input.mp4 \
  --envmap_path data/lighting.hdr \
  --prompt "A person standing in a room" \
  --height 832 --width 480 --num_frames 81 \
  --shading_timestep_init 0.6 --input_timestep_init 1.0 \
  --output_dir outputs/inference
```

No shading cue input or separate MoGe environment is required. The local MoGe
package is installed in the same environment as JointLight.

## Batch the release examples

The repository contains the prepared MP4 examples in `examples/videos/` (currently
`video_0.mp4` to `video_5.mp4`) and four HDRIs in `examples/hdris/`. The default
`configs/inference.json` selects these files and their lighting settings:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/relight_examples.sh
```

The number of workers follows the comma-separated GPU list in
`CUDA_VISIBLE_DEVICES`; `--gpus` can override it. If neither is set,
GPU indices are discovered with `nvidia-smi`. The current six videos
and four HDRIs produce 24 runs. `dry_field_1k.hdr` is rendered with
`--rotate CW`; all other HDRIs use fixed lighting. Results are grouped below
`outputs/examples/video_N/<hdri_name>/`, so parallel runs do not overwrite one
another. The wrapper uses 81 frames, shading timestep 0.6 and input timestep 1.0
by default. Edit the config's `defaults`, `videos` and `hdris` sections to change
these values. Pass `--config /path/to/custom.json` for a custom configuration.
See [the configuration guide](INFERENCE_CONFIG.md) for the complete schema.

Both height and width are optional together. Without them, the launcher uses
480 × 832 (width × height) for portrait video or 832 × 480 for landscape video. Frames must be
`4n+1`, and explicit dimensions must be divisible by 32. Only the requested video
prefix is processed. Short videos are padded to the requested length by repeating
their last frame (for example, 57 frames become 81 by appending 24 copies). This
also applies to a separate `--input_video_path`. Empty videos are rejected.
The first stage crops/resizes the video once, so geometry estimation
and relighting receive the same spatial/temporal input. Outputs use 30 FPS without
audio, following the original workflow.

The shading renderer retains MoGe v2, depth-edge threshold 0.1, half-resolution
rendering, the original camera/coordinate transforms and Mitsuba's internal
environment transform, 0.5 diffuse reflectance, 64 rendering samples per pixel
and path-integrator depth 3.
HDR data is loaded by Mitsuba, preserving its periodic seam column; rotating the
light updates only its transform. Inference upsamples the
shading to the requested resolution using its existing video loader.

## Single-image mesh and novel-view preview

For a still image, `scripts/render_moge_orbit.py` exports the MoGe reconstruction
and renders a one-second, 30-frame orbit. The default material is uniform gray,
so the preview does not depend on the source image texture:

```bash
python scripts/render_moge_orbit.py \
  --input /path/to/image.png \
  --output_video outputs/moge_orbit/orbit_1s_15deg_gray.mp4 \
  --mesh_output outputs/moge_orbit/moge_mesh \
  --orbit_angle_deg 15 --orbit_degrees 360 \
  --fps 30 --duration 1 --width 480 --height 832 --spp 16
```

The camera positions lie on a cone 15 degrees from the original view ray and
travel 360 degrees around it. The saved mesh prefix produces `.ply`, `.glb`,
`.obj` and a JSON metadata file. Add `--texture` only when the input image should
be attached to the mesh and rendered. A full 360-degree orbit can expose areas
that are not visible in a single image; those surfaces are MoGe extrapolations.

Use `--envmap_angle 90` to change the initial yaw. `--rotate CW` or `--rotate CCW`
rotates the light through one turn over the processed clip; `NONE` keeps it fixed.
`--threshold` controls depth-edge filtering. `--checkpoint`, `--model_config`,
`--tokenizer_path`, `--cfg_scale`, `--seed` and `--input_video_path` are forwarded to
the relighting stage. The bundled JointLight checkpoint is the default.

## Shading exposure and previews

The default `--shading_exposure_ev auto` meters the **entire rendered clip** and
chooses one exposure. Visible non-black surface pixels (alpha > 0.99 and linear
luminance > 1e-8) contribute to bounded-memory log histograms. The gain is the
smaller of `0.35 / median_luminance` and `0.95 / percentile_99.5(max(R,G,B))`,
limited to +/-8 EV. This balances midtone brightness and highlight headroom;
empty or all-black inputs use 0 EV. Metering uses 1/64-EV bins.

After this linear gain, a smooth highlight shoulder starts at maximum RGB 0.8:
`peak_out = 0.8 + 0.2 * (1 - exp(-(peak - 0.8) / 0.2))` for brighter pixels.
RGB channels are scaled together, preserving their ratios. The same gain and
curve apply to every frame, so rotation-induced changes in lighting remain;
the exposure never pumps in response to individual frames. Very dark regions
may still remain dark when the scene's dynamic range is large.

Auto mode temporarily stores raw EXRs, computes the clip statistics, and then
encodes the frames. It renders geometry and lighting only once, at the cost of
temporary disk space for the linear frames. `--save_shading_exr` retains those
frames after encoding. Otherwise they are removed when the job exits.

The full pipeline accepts `--shading_exposure_ev`, `--shading_spp` and
`--save_shading_exr`. A numeric EV (e.g. `-1` or `0`) explicitly selects manual
mode: only `linear_rgb * 2**EV`, with no highlight shoulder. `0` retains the old
exposure; the environment-map seam fix still applies. Mitsuba performs the
single sRGB conversion in both modes. Auto exposure and highlight compression
change the diffusion model's conditioning; compare final relighting quality
with the training-data convention when choosing a dataset-wide policy.

For example, automatic exposure with a full clockwise light rotation:

```bash
bash scripts/relight.sh \
  --video_path examples/videos/video_1.mp4 \
  --envmap_path examples/hdris/park_bench_1k.hdr \
  --prompt "A cat" --rotate CW --shading_exposure_ev auto
```

For a quick PNG preview without diffusion, run:

```bash
python scripts/generate_shading.py \
  --input examples/videos/video_1.mp4 \
  --env examples/hdris/dry_field_1k.hdr \
  --output outputs/preview/dry_field.png \
  --frame_index 0 --exposure_ev auto --spp 256 --save_exr
```

PNG previews process one zero-based video frame (default 0); MP4 output processes
the full input clip. `--device cpu` runs MoGe and Mitsuba's LLVM renderer on CPU;
the default remains CUDA. `--frame_index` is only accepted for PNG output.

Each output has a JSON sidecar with requested mode, resolved exposure, metering
statistics, sample count and per-frame clipping fractions before/after the
highlight shoulder, measured at pixels with alpha > 0.99. With
`--save_exr`, PNG previews retain a sibling `.exr`; MP4 output retains frames in
`<output_stem>/linear/`. EXRs contain raw linear RGBA **before exposure**, with
alpha unchanged. They allow re-exposure without rerendering; lowering brightness
on an already clipped PNG/MP4 cannot restore highlight detail. PNG writes finish
before FFmpeg starts encoding.

## Outputs and reuse

For `data/input.mp4`, outputs follow this layout:

```text
outputs/inference/input/
├── assets/
│   ├── input.mp4                  # cropped/resized video prefix
│   └── envmap.hdr                 # copy of the chosen environment map
├── moge_shading/
│   ├── lighting_0.0_threshold_0.1_ev_auto_spp_64.mp4
│   └── lighting_0.0_threshold_0.1_ev_auto_spp_64.json
├── comparison.mp4                 # input / MoGe / predicted shading / relit / envmap
└── relit/
    └── ...                        # relit RGB and predicted shading videos
```

`comparison.mp4` is generated automatically after the relighting stage. It has
five equal-width panels in this order: processed input, MoGe shading, predicted
shading, relit RGB, and the HDRI. The HDRI panel keeps the source file's
left-to-right orientation, then applies the initial yaw and frame-by-frame
CW/CCW rotation. This makes the first-frame panorama directly comparable with
the light direction visible in the MoGe cue. Pass
`--envmap_display_transform mitsuba` when you need to inspect the panorama after
Mitsuba's internal horizontal reflection. Use `--no_comparison` when only the
model outputs are needed.

Shading PNG frames are temporary and removed after MP4 encoding. The shading MP4
is retained. Chrome-ball previews, circle-EXR exports and unused debug geometry
from the old workflow are omitted because the relighting stages do not consume them.
Use a different output directory when preserving multiple runs with the same inputs.

To reuse the intermediate video without loading MoGe or Mitsuba:

```bash
bash scripts/relight.sh \
  --video_path data/input.mp4 \
  --prompt "A person standing in a room" \
  --height 832 --width 480 --num_frames 81 \
  --skip_moge \
  --shading_cue_path outputs/inference/input/moge_shading/lighting_0.0_threshold_0.1_ev_auto_spp_64.mp4
```

`--envmap_path` is optional in this mode. The reused cue must match the processed
input's crop, frame count and lighting. Add `--dry_run` to inspect commands without
creating files or starting models; specify both dimensions to avoid opening the
input video during command inspection.

## Validation scope

The launcher and renderer CLI, argument forwarding, intermediate file handling,
video preparation and failure propagation are checked locally.
GPU geometry inference and Mitsuba rendering require the optional environment and
CUDA; a full real-model relighting run has not been performed in the sandbox.
MoGe's original license notices are retained in `third_party/MoGe/LICENSE`.
