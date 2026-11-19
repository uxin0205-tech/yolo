#!/usr/bin/env bash
set -euo pipefail
portrait_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$portrait_root/yolo_actshape:$portrait_root/yolo_achitechure:$portrait_root/yolo_activation:$portrait_root/yolo_activation/src:$portrait_root/yolo_combine:$portrait_root/yolo_combine/src:$portrait_root/yolo_attention/src:$portrait_root/yolo_binaryqk:$portrait_root/yolo_masf:$portrait_root/yolo_achitechure/achitechure_1/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
exec "$portrait_root/.venv/bin/python" -B -m stage9_portrait.runner "$@"
