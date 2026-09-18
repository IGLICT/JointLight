#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# A video argument selects the low-level single-run interface. Otherwise use
# the configurable batch interface (default: configs/inference.json).
for argument in "$@"; do
  if [[ "$argument" == '--video_path' || "$argument" == --video_path=* ]]; then
    exec "${JOINTLIGHT_PYTHON:-python}" run_relit_pipeline.py "$@"
  fi
done
exec "${JOINTLIGHT_PYTHON:-python}" run_relighting.py "$@"
