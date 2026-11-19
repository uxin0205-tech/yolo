#!/usr/bin/env bash
set -euo pipefail
activation_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$activation_workspace/yolo_combine:$activation_workspace/yolo_combine/src:$activation_workspace/yolo_attention/src:$activation_workspace/yolo_binaryqk:$activation_workspace/yolo_masf:$activation_workspace/yolo_activation/src:$activation_workspace/yolo_achitechure/achitechure_1/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
exec "$activation_workspace/.venv/bin/python" "$activation_workspace/yolo_activation/stage7/prepare.py"
