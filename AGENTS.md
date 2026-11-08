# Repository Guidelines

## Project Structure & Module Organization

- `diffsynth/`: diffusion pipelines, training utilities, schedulers, and VRAM management.
- `train.py` and `inference.py`: training and inference with prepared shading cues.
- `run_relit_pipeline.py`: video/HDRI processing, shading generation, and inference; `run_relighting.py`: configurable batch execution.
- `scripts/`: shell launchers, shading rendering, and comparison-video tools; `utils/`: shared rendering helpers.
- `configs/` and `model_configs.json`: inference, Accelerate, and model-path configuration.
- `examples/videos/` and `examples/hdris/`: demonstration assets; `docs/`: detailed usage guides.
- `third_party/MoGe/`: vendored inference library; preserve its licenses and vendoring notes.

## Build, Test, and Development Commands

Run commands from the repository root. Follow `README.md` to prepare Python 3.10, FFmpeg, and CUDA-compatible PyTorch first.

- `python -m pip install -c constraints.txt -r requirements-shading.txt -e .`: install JointLight, local MoGe, and rendering dependencies for development.
- `python -m pip check`: check installed dependency compatibility.
- `python run_relighting.py --config configs/inference.json --gpus 0 --dry_run`: validate inputs and print planned jobs without loading models.
- `CUDA_VISIBLE_DEVICES=0 bash scripts/relight.sh --config configs/inference.json`: execute configured relighting jobs; the default selects 28 combinations.
- `bash scripts/infer.sh --help`: inspect prepared-cue inference options.
- `bash scripts/train.sh --help`: inspect training arguments; actual training requires dataset paths and model weights. Override Accelerate settings with `ACCELERATE_CONFIG`.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions and variables, and `PascalCase` classes. Match surrounding quote and import conventions. Preserve underscore-based CLI flags such as `--num_frames`. Shell launchers use `set -euo pipefail`. No formatter or linter is configured; avoid unrelated formatting changes.

## Testing Guidelines

No automated test suite or coverage threshold is configured. Run affected CLI help commands, the batch dry run, and the README's CUDA/import smoke check. For rendering changes, validate one video/HDRI pair and inspect RGB, shading, and comparison outputs. Record commands, seed, dimensions, frame count, and hardware limitations. Dimensions must be multiples of 32; frame counts must follow `4n+1`.

## Commit & Pull Request Guidelines

History uses short, plain-language subjects, including “Refresh demo videos and add synchronized lighting comparisons”. Use concise imperative subjects. PRs should describe behavior changes, link relevant issues, list validation performed, and include before/after frames or clips for visual changes. Update affected usage documentation.

## Configuration & Artifacts

Keep weights, datasets, outputs, logs, and secrets untracked; use the ignored `models/`, `data/`, and `outputs/` directories. Verify downloaded JointLight weights with `sha256sum -c CHECKPOINTS.sha256`. Resolve relative configuration paths through `path_root`, and choose separate output directories when preserving earlier runs.
