#!/usr/bin/env bash
set -euo pipefail
stage7_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$stage7_root/yolo_activation:$stage7_root/yolo_activation/src:$stage7_root/yolo_combine:$stage7_root/yolo_combine/src:$stage7_root/yolo_attention/src:$stage7_root/yolo_binaryqk:$stage7_root/yolo_masf:$stage7_root/yolo_achitechure/achitechure_1/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec "$stage7_root/.venv/bin/python" -m stage7.runner "$@"
