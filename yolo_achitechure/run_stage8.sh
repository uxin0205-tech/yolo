#!/usr/bin/env bash
set -euo pipefail
stage8_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$stage8_root/yolo_achitechure:$stage8_root/yolo_activation:$stage8_root/yolo_activation/src:$stage8_root/yolo_combine:$stage8_root/yolo_combine/src:$stage8_root/yolo_attention/src:$stage8_root/yolo_binaryqk:$stage8_root/yolo_masf:$stage8_root/yolo_achitechure/achitechure_1/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec "$stage8_root/.venv/bin/python" -m stage8.runner "$@"
