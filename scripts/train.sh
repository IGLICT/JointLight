#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec accelerate launch --config_file "${ACCELERATE_CONFIG:-configs/single_gpu.yaml}" train.py \
  --model_json model_configs.json \
  --trainable_models dit \
  --height 416 --width 256 --num_frames 81 \
  --dataset_type 3 --dataset_ratio 1 --dataset_repeat 1 \
  --data_file_keys video,cond_video,shading_gt,shading_cue \
  --learning_rate 1e-5 --num_epochs 20 --save_steps 500 \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path outputs/train \
  --hdri_base_path "" \
  --shading_cue_as_input True --start_match_shading 0 --end_match_shading 500 \
  "$@"
