#!/usr/bin/env bash
set -euo pipefail

# 從 production source 組裝可攜式 final 訓練包；不複製歷史 run 與 cache。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
final_root="${project_root}/final"

mkdir -p "${final_root}/yolo_attention"
mkdir -p "${final_root}/configs"
mkdir -p "${final_root}/data"
mkdir -p "${final_root}/weights"
mkdir -p "${final_root}/artifacts/runs/s0-phase-b-bittrue/checkpoints"
mkdir -p "${final_root}/artifacts/runs/s0-phase-b-bittrue/metrics"
mkdir -p "${final_root}/artifacts/runs/s0-phase-b-bittrue/profiles"
mkdir -p "${final_root}/tests"
mkdir -p "${final_root}/scripts"

cp -p "${project_root}"/src/yolo_attention/*.py "${final_root}/yolo_attention/"
cp -p "${project_root}/src/yolo_attention/README.md" "${final_root}/yolo_attention/README.md"
cp -a "${project_root}/configs/." "${final_root}/configs/"
cp -p "${project_root}/data/coco2017.yaml" "${final_root}/data/coco2017.yaml"
sed -i 's|^path:.*|path: datasets/coco2017|' "${final_root}/data/coco2017.yaml"
cp -p "${project_root}/weights/v1-br-best.pt" "${final_root}/weights/v1-br-best.pt"
cp -p \
  "${project_root}/artifacts/runs/s0-phase-b-bittrue/checkpoints/evaluated-variant.pt" \
  "${final_root}/artifacts/runs/s0-phase-b-bittrue/checkpoints/evaluated-variant.pt"
cp -p \
  "${project_root}/artifacts/runs/s0-phase-b-bittrue/manifest.json" \
  "${final_root}/artifacts/runs/s0-phase-b-bittrue/manifest.json"
cp -p \
  "${project_root}/artifacts/runs/s0-phase-b-bittrue/metrics/queue-result.json" \
  "${final_root}/artifacts/runs/s0-phase-b-bittrue/metrics/queue-result.json"
cp -p \
  "${project_root}/artifacts/runs/s0-phase-b-bittrue/profiles/analytical.json" \
  "${final_root}/artifacts/runs/s0-phase-b-bittrue/profiles/analytical.json"
for test_source in "${project_root}"/tests/*.py; do
  if test "${test_source##*/}" = "test_portable_final.py"; then
    continue
  fi
  cp -p "${test_source}" "${final_root}/tests/"
done
cp -p "${project_root}/scripts/smoke_cpu.py" "${final_root}/scripts/smoke_cpu.py"
sed -i \
  's|Path(__file__).resolve().parents\[1\] / "src"|Path(__file__).resolve().parents[1]|' \
  "${final_root}/tests/conftest.py"
sed -i 's|root / "final/pwl-final-best.pt"|root / "pwl-final-best.pt"|' \
  "${final_root}/tests/test_checkpoint_contract.py"

echo "Portable final assets synchronized at ${final_root}"
