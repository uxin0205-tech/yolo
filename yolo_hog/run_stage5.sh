#!/usr/bin/env bash
set -euo pipefail
stage5_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$stage5_root/yolo_hog:$stage5_root/yolo_attention/src:$stage5_root/yolo_binaryqk:$stage5_root/yolo_activation/src:$stage5_root/yolo_achitechure/achitechure_1/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
exec "$stage5_root/.venv/bin/python" -m hog_yolo.cli "$@"
