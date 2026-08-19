# architecture_2 執行手冊

所有命令都從本目錄執行，且預設只預覽。除非命令明確帶有 `--execute`，否則不寫入模型或啟動訓練；Pose 另要求 `--enable-pose`。
訓練一律推薦使用 `--config` 指定一份完整正式 YAML；檔案總表在 `configs/README.md`。

```bash
export PYTHONPATH="$PWD/src:$PWD/../achitechure_1/src:$PWD/../../yolo_attention_final/src"
PY=/home/uxin/yolo/.venv/bin/python
$PY -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__)"
$PY -m achitechure_2 config-check
$PY -m achitechure_2 prepare-pose-data
$PY -m pytest
$PY -m achitechure_2 preflight
```

`config-check` 必須通過。`preflight` 另外要求 CUDA、pycocotools 與已驗收上游交付。

## 1. 驗收 architecture_1 交付

由 `handoff-manifest.example.json` 建立不納入版本控制的 manifest：

```bash
$PY -m achitechure_2 intake --manifest /absolute/path/handoff.json
$PY -m achitechure_2 intake --manifest /absolute/path/handoff.json --execute
```

正式記錄位於 `artifacts/intake/accepted.json`，會驗證 checkpoint hashes/backends、MASF、attention paths、YOLO26m 三尺度 geometry、版本、selection manifest 與 fresh-process reload。

## 2. 獨立建立候選

```bash
for id in C0 C1 C2 C3; do
  $PY -m achitechure_2 build --candidate "$id"
done
for id in C0 C1 C2 C3; do
  $PY -m achitechure_2 build --candidate "$id" --execute
done
```

每個輸出包含 `transfer-report.json`、`architecture-snapshot.yaml` 與 `lineage.json`。Snapshot 只供審查，不可獨立載入。

## 3. Detect D0／D1／D2

```bash
$PY -m achitechure_2 train --config configs/training/detect/d0-smoke.yaml --candidate C0 --checkpoint artifacts/candidates/c0/float-parent.pt --run-id c0-d0 --smoke-epochs 3
$PY -m achitechure_2 train --config configs/training/detect/d0-smoke.yaml --candidate C0 --checkpoint artifacts/candidates/c0/float-parent.pt --run-id c0-d0 --smoke-epochs 3 --execute
$PY -m achitechure_2 train --config configs/training/detect/d1-main.yaml --candidate C0 --checkpoint artifacts/candidates/c0/float-parent.pt --run-id c0-d1 --execute
```

C1/C2/C3 使用完全相同命令與預算。D0 不排名。D1 跑滿後，用 100 個從 epoch 1 起算的 Bit-True 指標檢查延長門檻：

```bash
$PY -m achitechure_2 extension-gate --metrics /path/to/epoch-map.json --best-epoch 91
$PY -m achitechure_2 train --config configs/training/detect/d2-extension.yaml --candidate C0 --checkpoint /path/to/d1/last.pt --run-id c0-d2 --execute
```

只有 `extend=true` 才執行 D2。選定 checkpoint 要在獨立程序 materialize、reload、validate 與 profile：

```bash
$PY -m achitechure_2 materialize-bittrue --candidate C0 --checkpoint /path/best-float.pt --output /path/best-bittrue.pt --execute
$PY -m achitechure_2 validate-bittrue --checkpoint /path/best-bittrue.pt --run-id c0-fresh --execute
$PY -m achitechure_2 profile --checkpoint /path/best-bittrue.pt --output artifacts/runs/c0-d1/profile.json --warmup 20 --iterations 100 --execute
```

## 4. 選型與條件候選

```bash
$PY -m achitechure_2 assess --c0 /path/c0.json --candidate /path/c1.json --candidate /path/c2.json --candidate /path/c3.json
$PY -m achitechure_2 assess --c0 /path/c0.json --candidate /path/c1.json --candidate /path/c2.json --candidate /path/c3.json --execute
```

只有 `triggers.c3_p5` 可授權 C3-P5，只有 `triggers.r1` 可授權 R1。R1 必須先提供機器可讀的 RepConv fusion 證據：

```bash
$PY -m achitechure_2 rep-fusion --checkpoint /path/r1/best-float.pt --output artifacts/runs/r1/fusion.json
$PY -m achitechure_2 rep-fusion --checkpoint /path/r1/best-float.pt --output artifacts/runs/r1/fusion.json --execute
$PY -m achitechure_2 assess --c0 /path/c0.json --candidate /path/c1.json --candidate /path/c2.json --candidate /path/c3.json --candidate /path/r1.json --r1-fusion-report artifacts/runs/r1/fusion.json --execute
```

C_best 為 null 時，停止下游比較與量化主線。

## 5. 建立 grouped Pose 資料（不執行 Pose）

```bash
$PY -m achitechure_2 prepare-pose-data
$PY -m achitechure_2 prepare-pose-data --execute
```

執行只建立 `artifacts/datasets/pose_grouped`：影像 symlink、標籤副本、`split-manifest.json` 與 `patch-manifest.json`。來源保持唯讀，不建立 test split。這項命令完全不需要 `--enable-pose`，因為它只準備資料，不會建模或訓練。

## 6. 可選的 Pose P0–P4

本節全部是選用流程。未提供 `--enable-pose --execute` 就不會執行 Pose。

先預覽 graft：

```bash
$PY -m achitechure_2 build-pose --candidate C0 --checkpoint artifacts/candidates/c0/float-parent.pt
```

你決定執行後才使用：

```bash
$PY -m achitechure_2 build-pose --candidate C0 --checkpoint artifacts/candidates/c0/float-parent.pt --enable-pose --execute
$PY -m achitechure_2 train --config configs/training/pose/p0-smoke.yaml --candidate C0 --checkpoint /path/to/c0-pose-graft.pt --run-id pose-c0-p0 --enable-pose --execute
$PY -m achitechure_2 train --config configs/training/pose/p1-head-only.yaml --candidate C0 --checkpoint /path/to/c0-pose-graft.pt --run-id pose-c0-p1 --enable-pose --execute
$PY -m achitechure_2 train --config configs/training/pose/p2-neck-head.yaml --candidate C0 --checkpoint /path/to/p1/best.pt --run-id pose-c0-p2 --enable-pose --execute
$PY -m achitechure_2 train --config configs/training/pose/p3-full.yaml --candidate C0 --checkpoint /path/to/p2/best.pt --run-id pose-c0-p3 --enable-pose --execute
```

P3 跑滿且通過晚期改善門檻後才可 P4：

```bash
$PY -m achitechure_2 train --config configs/training/pose/p4-extension.yaml --candidate C0 --checkpoint /path/to/p3/last.pt --run-id pose-c0-p4 --enable-pose --execute
$PY -m achitechure_2 validate-bittrue --task pose --checkpoint /path/to/pose-bittrue.pt --run-id pose-c0-fresh --enable-pose --execute
```

實際 C_best 使用相同流程。P1/P2/P3 建立新 optimizer；P4 是真正 resume。

## 7. 未來雙權重／雙架構融合

先複製 `configs/fusion/source-pair.template.yaml` 到實驗產物目錄，填入來源 A/B checkpoint、architecture YAML、head contract 與 hashes。模板目前 `enabled=false`，且不得填寫 `fusion_mode` 或 `switch_policy`，直到你明確選擇以下其中一種語意：weight interpolation、ensemble、routed switch 或 staged transfer。此版本不會自動融合。

## 8. Q0／Q1／Q2

只允許 C0 與已記錄的 C_best：

```bash
$PY -m achitechure_2 fuse-reference --candidate C0 --checkpoint /path/c0/best-float.pt --output artifacts/quant/c0/q0/fused.pt --execute
$PY -m achitechure_2 quant-prepare --candidate C0 --checkpoint artifacts/quant/c0/q0/fused.pt --output artifacts/quant/c0/q1/prepared.pt --execute
$PY -m achitechure_2 quant-calibrate --checkpoint artifacts/quant/c0/q1/prepared.pt --calibration-tensors /path/fixed-images.pt --output artifacts/quant/c0/q1/calibrated.pt --execute
$PY -m achitechure_2 quant-prepare --candidate C0 --checkpoint /path/c0/best-float.pt --output artifacts/quant/c0/q2/prepared.pt --execute
$PY -m achitechure_2 train --config configs/training/detect/q2-qat.yaml --candidate C0 --checkpoint artifacts/quant/c0/q2/prepared.pt --run-id c0-q2 --execute
$PY -m achitechure_2 quant-report --q0 0.50 --q1 0.49 --q2 0.497
```

所有量化記錄都必須保留 `simulation_only=true`。

## 9. 收尾驗證

```bash
$PY -m achitechure_2 config-check
$PY -m pytest
git status --short --branch
```

完成所有 fresh-process validation 後，才由範本建立 `results/results.csv` 與 `REPORT.md`，並記錄硬體、seed、physical batch、resolved config、transfer 結果、hashes、停止原因與指標。
