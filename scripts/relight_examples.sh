#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${JOINTLIGHT_PYTHON:-python}" run_relighting.py "$@"
