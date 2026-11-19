#!/usr/bin/env bash
set -euo pipefail
stage4_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$stage4_root/yolo_masf:$stage4_root/yolo_attention/src:$stage4_root/yolo_binaryqk:$stage4_root/yolo_activation/src:$stage4_root/yolo_achitechure/achitechure_1/src:$stage4_root/yolo_combine/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
exec "$stage4_root/.venv/bin/python" -m masf_yolo.cli stage4 "$@"
