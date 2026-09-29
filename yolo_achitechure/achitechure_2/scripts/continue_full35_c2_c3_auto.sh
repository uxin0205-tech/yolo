#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="/home/uxin/yolo/yolo_achitechure/achitechure_2"
readonly PYTHON_BIN="/home/uxin/yolo/.venv/bin/python"
readonly FULL35_SRC="/home/uxin/yolo/yolo_combine/final/full35/code/project/src"
readonly LOCAL_PYTHONPATH="${PROJECT_ROOT}/src:${FULL35_SRC}"
readonly POSTPROCESS_UNIT="yolo-architecture2-full35-j3-float20-postprocess-seed0.service"
readonly CONFIG="${PROJECT_ROOT}/configs/runs/full35-c2-c3-auto-continuation.yaml"
readonly FULL_STATE="${PROJECT_ROOT}/artifacts/queue/full35-j3-c2-c3-full-seed0.json"
readonly FULL_LOG="${PROJECT_ROOT}/artifacts/queue/full35-j3-c2-c3-full-seed0.log"
readonly FULL_LOCK="${PROJECT_ROOT}/artifacts/queue/full35-j3-c2-c3-full-seed0.lock"
readonly QUANT_STATE="${PROJECT_ROOT}/artifacts/queue/full35-j3-c2-c3-quant-seed0.json"
readonly QUANT_LOG="${PROJECT_ROOT}/artifacts/queue/full35-j3-c2-c3-quant-seed0.log"
readonly QUANT_LOCK="${PROJECT_ROOT}/artifacts/queue/full35-j3-c2-c3-quant-seed0.lock"
readonly FULL_MATRIX="${PROJECT_ROOT}/artifacts/runs/full35-j3-c2-c3-full-seed0/matrix-complete.json"
readonly QUANT_MATRIX="${PROJECT_ROOT}/results/full35-j3-c2-c3-quant-seed0/matrix-complete.json"
readonly FINAL_MANIFEST="${PROJECT_ROOT}/results/full35-j3-c2-c3-final-seed0/manifest.json"

while systemctl --user is-active --quiet "${POSTPROCESS_UNIT}"; do
  sleep 30
done


cd "${PROJECT_ROOT}"
set +e
env PYTHONPATH="${LOCAL_PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=   "${PYTHON_BIN}" -c '
from pathlib import Path
from achitechure_2.full_training import FullRunConfig, eligible_full_candidates
config = FullRunConfig.load(Path("configs/runs/full35-c2-c3-auto-continuation.yaml"))
report = eligible_full_candidates(config)
raise SystemExit(0 if report["eligible_candidates"] else 3)
'
eligibility_status=$?
set -e
if [[ "${eligibility_status}" -eq 3 ]]; then
  env PYTHONPATH="${LOCAL_PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=     "${PYTHON_BIN}" -c '
import json
from pathlib import Path
from achitechure_2.full_training import FullRunConfig, eligible_full_candidates
from achitechure_2.screen_training import _write_json
config = FullRunConfig.load(Path("configs/runs/full35-c2-c3-auto-continuation.yaml"))
report = eligible_full_candidates(config)
_write_json(config.run_root / "matrix-complete.json", {
    "schema_version": 1,
    "status": "completed_no_eligible_candidates",
    "eligible_candidates": [],
    "eligibility": report,
})
'
  env PYTHONPATH="${LOCAL_PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=     "${PYTHON_BIN}" -m achitechure_2 --project-root "${PROJECT_ROOT}"     export-downstream-results --config "${CONFIG}" --execute
  exit 0
elif [[ "${eligibility_status}" -ne 0 ]]; then
  echo "Float20資格判定失敗：exit=${eligibility_status}" >&2
  exit "${eligibility_status}"
fi

env PYTHONPATH="${LOCAL_PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=   "${PYTHON_BIN}" -c 'from yolo_combine.gpu_queue import main; raise SystemExit(main())'   --queue-id architecture2-full35-j3-c2-c3-full-seed0 --gpu-index 0   --min-free-mib 30000 --max-utilization 10 --stable-polls 3 --poll-seconds 60   --state "${FULL_STATE}" --log "${FULL_LOG}" --lock "${FULL_LOCK}"   --working-directory "${PROJECT_ROOT}" --   "${PYTHON_BIN}" -m achitechure_2 --project-root "${PROJECT_ROOT}"   full-run --config "${CONFIG}" --queue-state "${FULL_STATE}" --execute

env PYTHONPATH="${LOCAL_PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=   "${PYTHON_BIN}" -c '
import json
from pathlib import Path
path = Path("artifacts/runs/full35-j3-c2-c3-full-seed0/matrix-complete.json")
payload = json.loads(path.read_text(encoding="utf-8"))
status = payload.get("status")
if status != "completed_formal_training_matrix":
    raise SystemExit(f"full matrix 未正常完成：{status}")
'

env PYTHONPATH="${LOCAL_PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=   "${PYTHON_BIN}" -c 'from yolo_combine.gpu_queue import main; raise SystemExit(main())'   --queue-id architecture2-full35-j3-c2-c3-quant-seed0 --gpu-index 0   --min-free-mib 30000 --max-utilization 10 --stable-polls 3 --poll-seconds 60   --state "${QUANT_STATE}" --log "${QUANT_LOG}" --lock "${QUANT_LOCK}"   --working-directory "${PROJECT_ROOT}" --   "${PYTHON_BIN}" -m achitechure_2 --project-root "${PROJECT_ROOT}"   quant-run --config "${CONFIG}" --queue-state "${QUANT_STATE}" --execute

env PYTHONPATH="${LOCAL_PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=   "${PYTHON_BIN}" -c '
import json
from pathlib import Path
path = Path("results/full35-j3-c2-c3-quant-seed0/matrix-complete.json")
payload = json.loads(path.read_text(encoding="utf-8"))
status = payload.get("status")
if status != "completed_q0_q1_q2l_matrix":
    raise SystemExit(f"quant matrix 未正常完成：{status}")
'

env PYTHONPATH="${LOCAL_PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=   "${PYTHON_BIN}" -m achitechure_2 --project-root "${PROJECT_ROOT}"   export-downstream-results --config "${CONFIG}" --execute

test -f "${FULL_MATRIX}"
test -f "${QUANT_MATRIX}"
test -f "${FINAL_MANIFEST}"
