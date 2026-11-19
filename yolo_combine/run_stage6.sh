#!/usr/bin/env bash
set -euo pipefail
stage6_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$stage6_root/yolo_combine:$stage6_root/yolo_combine/src:$stage6_root/yolo_attention/src:$stage6_root/yolo_binaryqk:$stage6_root/yolo_masf:$stage6_root/yolo_activation/src:$stage6_root/yolo_achitechure/achitechure_1/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
exec "$stage6_root/.venv/bin/python" -m stage6.cli "$@"
