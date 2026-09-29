#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="/home/uxin/yolo/yolo_achitechure/achitechure_2"
readonly PYTHON_BIN="/home/uxin/yolo/.venv/bin/python"
readonly TRAINING_UNIT="yolo-architecture2-full35-j3-float20-seed0-retry1.service"
readonly RUN_CONFIG="${PROJECT_ROOT}/configs/runs/full35-float-screen-20.yaml"
readonly RUN_ROOT="${PROJECT_ROOT}/artifacts/runs/full35-j3-float20-seed0"
readonly QUEUE_ROOT="${PROJECT_ROOT}/artifacts/queue"
readonly QUEUE_STATE="${QUEUE_ROOT}/full35-j3-float20-seed0.json"
readonly TRAINING_LOG="${QUEUE_ROOT}/full35-j3-float20-seed0.log"
readonly QUEUE_LOCK="${QUEUE_ROOT}/full35-j3-float20-seed0.lock"
readonly PROFILE_LOG="${QUEUE_ROOT}/full35-j3-float20-profile-seed0.log"
readonly HISTORY_ROOT="${QUEUE_ROOT}/history"
readonly LOCAL_PYTHONPATH="${PROJECT_ROOT}/src:/home/uxin/yolo/yolo_combine/final/full35/code/project/src"
readonly ARCHIVED_STATE="${HISTORY_ROOT}/full35-j3-float20-seed0-retry1-completed.json"
readonly ARCHIVED_LOG="${HISTORY_ROOT}/full35-j3-float20-seed0-retry1-completed.log"

while systemctl --user is-active --quiet "${TRAINING_UNIT}"; do
  sleep 30
done

# systemd --collect 與 queue 原子寫檔之間可能相差極短時間；最多等 60 秒後 fail closed。
for _ in $(seq 1 20); do
  if [[ -f "${RUN_ROOT}/matrix-complete.json" ]]; then
    break
  fi
  sleep 3
done

"${PYTHON_BIN}" -c '
import json
import sys
from pathlib import Path

state_path, archived_state_path, matrix_path = map(Path, sys.argv[1:])
states = []
for path in (state_path, archived_state_path):
    if path.is_file():
        states.append(json.loads(path.read_text(encoding="utf-8")))
state = next(
    (value for value in states if value.get("status") == "completed" and value.get("return_code") == 0),
    None,
)
if state is None:
    raise SystemExit(f"training queue 未正常完成：{states}")
matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
matrix_status = matrix.get("status")
if matrix_status != "completed_screening_matrix":
    raise SystemExit(f"Float20 matrix 未正常完成：{matrix_status}")
if matrix.get("c_best") is not None:
    raise SystemExit("matrix 不得自動選出 C_best")
' "${QUEUE_STATE}" "${ARCHIVED_STATE}" "${RUN_ROOT}/matrix-complete.json"

mkdir -p "${HISTORY_ROOT}"
if [[ -e "${ARCHIVED_STATE}" && -e "${ARCHIVED_LOG}" ]]; then
  :
elif [[ -e "${ARCHIVED_STATE}" || -e "${ARCHIVED_LOG}" ]]; then
  echo "訓練 queue 封存不完整；拒絕繼續" >&2
  exit 1
else
  cp --preserve=timestamps "${QUEUE_STATE}" "${ARCHIVED_STATE}"
  cp --preserve=timestamps "${TRAINING_LOG}" "${ARCHIVED_LOG}"
fi

env PYTHONPATH="${LOCAL_PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  "${PYTHON_BIN}" -c 'from yolo_combine.gpu_queue import main; raise SystemExit(main())' \
  --queue-id architecture2-full35-j3-float20-seed0 --gpu-index 0 \
  --min-free-mib 30000 --max-utilization 10 --stable-polls 3 --poll-seconds 60 \
  --state "${QUEUE_STATE}" --log "${PROFILE_LOG}" --lock "${QUEUE_LOCK}" \
  --working-directory "${PROJECT_ROOT}" -- \
  "${PYTHON_BIN}" -m achitechure_2 --project-root "${PROJECT_ROOT}" \
  float20-profile --config "${RUN_CONFIG}" --queue-state "${QUEUE_STATE}" --execute

env PYTHONPATH="${LOCAL_PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  "${PYTHON_BIN}" -m achitechure_2 --project-root "${PROJECT_ROOT}" \
  export-float20-results --config "${RUN_CONFIG}" --execute
