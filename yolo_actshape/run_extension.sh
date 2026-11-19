#!/usr/bin/env bash
set -euo pipefail
stage9_extra_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$stage9_extra_root/yolo_actshape:$stage9_extra_root/yolo_achitechure:$stage9_extra_root/yolo_activation:$stage9_extra_root/yolo_activation/src:$stage9_extra_root/yolo_combine:$stage9_extra_root/yolo_combine/src:$stage9_extra_root/yolo_attention/src:$stage9_extra_root/yolo_binaryqk:$stage9_extra_root/yolo_masf:$stage9_extra_root/yolo_achitechure/achitechure_1/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
exec "$stage9_extra_root/.venv/bin/python" -B -m stage9_extension.runner "$@"
