# JointLight

Video relighting with joint RGB/shading diffusion, diffusion forcing, shared RoPE, and modality-conditioned adaLN. This initial source release retains the active Wan2.2-TI2V-5B training and inference implementation.

## Installation from scratch

### 1. Create the environment

Open a terminal in the JointLight source directory, then run:

```bash
conda create -n jointlight -c conda-forge python=3.10 pip git ffmpeg -y
conda activate jointlight
```


### 2. Install the build tools and CUDA PyTorch wheels

```bash
python -m pip install -c constraints.txt setuptools wheel
python -m pip install -c constraints.txt \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install JointLight, local MoGe and rendering dependencies

```bash
python -m pip install -c constraints.txt -r requirements-shading.txt -e .
python -m pip check
```

This installs all of the following in the same environment:

- JointLight and its training/inference dependencies from `requirements.txt`.
- MoGe from `third_party/MoGe`.
- The pinned `utils3d` Git revision used by the geometry/rendering integration.
- Mitsuba and Dr.Jit for CUDA shading rendering.


### 4. Verify CUDA and the shared environment

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PYTHON'
from pathlib import Path
import sys
import torch
import cv2
import moge
import mitsuba as mi
import drjit as dr
import utils3d
from moge.model import import_model_class_by_version
from diffsynth.pipelines.wan_video_relit import WanVideoPipeline
import train

print("Python:", sys.executable)
print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is not available; check the NVIDIA driver and GPU visibility"
vendor = (Path.cwd() / "third_party/MoGe").resolve()
print("MoGe:", moge.__file__)
assert Path(moge.__file__).resolve().is_relative_to(vendor), "MoGe must resolve to third_party/MoGe"
import_model_class_by_version("v2")
assert callable(utils3d.np.build_mesh_from_map)
assert callable(utils3d.pt.depth_map_to_point_map)
mi.set_variant("cuda_ad_rgb")
assert dr.all(mi.Float([1, 2]) + 1 == mi.Float([2, 3]))
print("Mitsuba:", mi.__version__, "Dr.Jit:", dr.__version__, "OpenCV:", cv2.__version__)
print("JointLight + MoGe + CUDA renderer imports OK")
PYTHON
python run_relit_pipeline.py --help
python scripts/generate_shading.py --help
python inference.py --help
python train.py --help
```

This checks imports and CUDA initialization without loading model weights. Full
inference additionally requires the model files and input assets below. 

### 5. Prepare weights

Download the JointLight checkpoint:

```bash
hf download sagileo/JointLight \
  --local-dir models/JointLight
```

Download the Wan2.2 base files separately:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  --include 'models_t5_umt5-xxl-enc-bf16.pth' 'Wan2.2_VAE.pth' \
    'diffusion_pytorch_model-*.safetensors' 'google/umt5-xxl/*' \
  --local-dir models/Wan-AI/Wan2.2-TI2V-5B
```

Verify the checkpoint from the repository root:

```bash
sha256sum -c CHECKPOINTS.sha256
```

The model repository has this layout:

```text
models/JointLight/
├── JointLight.safetensors
```

```text
models/Wan-AI/Wan2.2-TI2V-5B/
├── models_t5_umt5-xxl-enc-bf16.pth
├── Wan2.2_VAE.pth
├── diffusion_pytorch_model-00001-of-00003.safetensors
├── diffusion_pytorch_model-00002-of-00003.safetensors
├── diffusion_pytorch_model-00003-of-00003.safetensors
└── google/umt5-xxl/
```

MoGe downloads `Ruicheng/moge-2-vitl-normal` on first use.

## Inference

### Config-driven inference 

Edit [configs/inference.json](configs/inference.json) to choose input videos,
HDRIs, rotation, prompts, model paths and inference settings. By default it runs
all 24 combinations of the six example videos and four HDRIs; only `dry_field_1k`
rotates clockwise.

```bash
conda activate jointlight
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/relight.sh --config configs/inference.json
```

Use a single GPU with `CUDA_VISIBLE_DEVICES=0`. Validate your config first with:

```bash
python run_relighting.py --config configs/inference.json --gpus 0,1 --dry_run
```

See [the config guide](docs/INFERENCE_CONFIG.md) for per-video/per-HDRI overrides,
path resolution, GPU assignment and output locations. Each run saves the selected
config and generates the five-column comparison automatically.

### From a video and an environment map

Activate `jointlight` first. The complete pipeline generates shading internally,
then runs JointLight in the same environment:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/relight.sh \
  --video_path examples/videos/video_0.mp4 \
  --envmap_path examples/hdris/park_bench_1k.hdr \
  --prompt "A person standing in a room" \
  --height 832 --width 480 --num_frames 81 \
  --shading_timestep_init 0.6 --input_timestep_init 1.0 \
  --output_dir outputs/inference
```

This uses `third_party/MoGe` and the bundled Mitsuba shading script in the unified
environment configured above; see
[the complete pipeline guide](docs/RELIGHT_PIPELINE.md) for installation, local
MoGe weights, environment rotation, intermediate outputs and cue reuse.
After relighting, the pipeline also writes `comparison.mp4` beside the run's
assets. Its five panels are processed input, MoGe shading, predicted shading,
relit output and the source-oriented HDRI. Use `--no_comparison` to disable it.

### With an existing shading cue

Provide a source video, an aligned target shading cue video and a prompt:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/infer.sh \
  --video_path data/input.mp4 \
  --shading_cue_paths data/shading.mp4 \
  --prompt "A person standing in a room" \
  --height 832 --width 480 --num_frames 81 \
  --shading_timestep_init 0.6 --input_timestep_init 1.0 \
  --output_dir outputs/inference
```

The selected checkpoint is the default. Override it with
`--checkpoint /path/to/model.safetensors`. Multiple shading paths, or a directory
of MP4 shading cues, are supported. Each cue produces a relit RGB video and a
`_shading.mp4` output under `--output_dir`. The original output encoding uses
30 FPS and omits audio.

Use dimensions divisible by 32 and `4n+1` frames (e.g. 1, 49 or 81). Each input
must contain at least that many frames; frames are taken from the video prefix.
Source and shading frames must be spatially and temporally aligned. `--num_frames 1`
uses the first frame of the input videos. Use `--input_video_path` only when a
separate RGB initialization video is needed; it defaults to the source video.

The original noise controls are retained. The example uses shading noise 0.6;
the `inference.py` CLI default is 1.0. The full pipeline defaults to 0.6 and
generates cues internally; `scripts/infer.sh` is the direct entry point for
already prepared cues.

## Attribution

The implementation is derived from DiffSynth-Studio and Wan. The original
[LICENSE](LICENSE) is retained, together with copyright/license notices in source
files. See [third-party notices](THIRD_PARTY_NOTICES.md) for MoGe, DINOv2
and example-asset attribution. This repository does not replace the terms accompanying external models,
data or separately distributed weights.
