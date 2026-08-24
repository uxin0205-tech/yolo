# achitechure_1 runbook

請從 `/home/uxin0/yolo/yolo_achitechure/achitechure_1` 執行。本工作項目使用獨立且固定版本的虛擬環境，
避免 repository 共用的 `yolo26m` 環境變動影響正式結果。載入 parent checkpoint 時仍需使用隨附的
attention implementation。

第一次使用時建立環境：

```bash
BASE_PYTHON="$PWD/../../ENV_setup/miniconda3/envs/yolo26m/bin/python"
$BASE_PYTHON -m venv "$PWD/.venv"
$PWD/.venv/bin/python -m pip install --upgrade pip==26.2.1
$PWD/.venv/bin/python -m pip install -r requirements-lock.txt
```

每次開啟新 shell 時設定執行環境：

```bash
export PYTHONPATH="$PWD/src:$PWD/../../yolo_attention_final/src"
export YOLO_CONFIG_DIR="$PWD/.runtime/ultralytics"
export MPLCONFIGDIR="$PWD/.runtime/matplotlib"
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR"
PYTHON="$PWD/.venv/bin/python"
```

正式環境契約為 Python 3.12、PyTorch `2.11.0+cu128`、Ultralytics `8.4.90`。只要任一套件版本不同，
`preflight` 就會 fail closed。

Commands are dry-run by default where they can start training, validation, conversion, or profiling. Add
`--execute` only on the intended GPU host. A1、A2、Phase B 固定 physical batch 16、`nbs=16`；只有
Phase C 使用 physical batch 8、`nbs=16`（accumulate 2）。Validation 另外固定 batch 8。

## Environment and tests

```bash
$PYTHON -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__)"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
$PYTHON -m achitechure_1.cli preflight
$PYTHON -m pytest
$PYTHON -m achitechure_1.cli smoke --variant full35 --device cuda:0
$PYTHON -m achitechure_1.cli smoke --variant partial75 --device cuda:0
```

Preflight must report `valid: true`, three Detect inputs `[16,19,22]`, strides `[8,16,32]`, `end2end: true`,
the two attention paths, MuSGD, fresh-process reload, and at least 95% Float reconstruction coverage. Formal COCO
validation also requires the optional evaluation dependency:

```bash
$PYTHON -m pip install -e '.[evaluation]'
```

## A0 immutable baseline

A0 is `inputs/parent/best.pt`; do not train it. Profile and validate the actual Bit-True checkpoint:

```bash
$PYTHON -m achitechure_1.cli profile \
  --checkpoint inputs/parent/best.pt --kind inference \
  --output artifacts/profiles/a0-inference.json --execute

$PYTHON -m achitechure_1.cli validate \
  --checkpoint inputs/parent/best.pt --run-id a0-bittrue --execute
```

## Prepare A1 and A2

These commands add MASF without changing any original parent tensor or graph index:

```bash
$PYTHON -m achitechure_1.cli prepare \
  --variant full35 --output artifacts/prepared/a1-full35.pt --execute
$PYTHON -m achitechure_1.cli prepare \
  --variant partial75 --output artifacts/prepared/a2-partial75.pt --execute
```

Before formal epochs, prove fixed batch 16/640 fits using the official YOLO loss. Do not pass a smaller batch:

```bash
$PYTHON -m achitechure_1.cli profile \
  --checkpoint artifacts/prepared/a1-full35.pt --kind training \
  --output artifacts/profiles/a1-phase-c-train-smoke-batch16-amp-rtx5060ti-2026-08-19.json \
  --steps 3 --amp --execute
$PYTHON -m achitechure_1.cli profile \
  --checkpoint artifacts/prepared/a2-partial75.pt --kind training \
  --output artifacts/profiles/a2-phase-c-train-smoke-batch16-amp-rtx5060ti-2026-08-19.json \
  --steps 3 --amp --execute
```

The 2026-08-17 attempt stopped with OOM while the separate parent attention queue held about 15.5 GiB VRAM.
After that queue completed, the exact probes were repeated on an RTX 5090 on 2026-08-18 and both passed without batch
reduction or accumulation: Full35 peaked at `18,397,301,760` bytes and Partial75 at `17,761,890,304` bytes. The
committed reports are `artifacts/profiles/a1-train-smoke.json` and `artifacts/profiles/a2-train-smoke.json`.

2026-08-19 的本機 preflight 偵測到 RTX 5060 Ti 16 GB。固定版本環境下，Phase C 的兩步 real-loss probe
已使用 physical batch 8、`nbs=16` 通過：Full35 峰值 `9,474,681,856` bytes，Partial75 峰值
`9,161,244,672` bytes。報告分別位於
`artifacts/profiles/a1-phase-c-train-smoke-rtx5060ti-2026-08-19.json` 與
`artifacts/profiles/a2-phase-c-train-smoke-rtx5060ti-2026-08-19.json`。這些檢查不是正式訓練。最終 A0、
Full35 A2、Partial75 A2 已於 2026-08-24 在同一張 RTX 5060 Ti，以 FP16、batch 1、imgsz 640、20 warmup、
100 iterations 補齊正式 latency；原始 JSON 位於
`artifacts/profiles/*bittrue-inference-fp16-rtx5060ti-2026-08-24.json`。Peak VRAM 只作容量診斷，不參與架構排名。

同日補測與正式 `amp=true` 一致的 official-loss probe，physical batch 16 可安全使用；先前 FP32 probe
超過實體 VRAM 的配置量不代表 AMP 正式訓練會 OOM：

| 架構 | physical batch / accumulate | Effective batch | Peak VRAM | 穩態 p50 |
|---|---:|---:|---:|---:|
| Full35 | 8 / 2 | 16 | 5,328,103,424 bytes | 445.75 ms |
| Full35 | 16 / 1 | 16 | 10,265,171,456 bytes | 433.39 ms |
| Partial75 | 8 / 2 | 16 | 5,165,854,720 bytes | 432.38 ms |
| Partial75 | 16 / 1 | 16 | 9,950,969,344 bytes | 443.36 ms |

雖然 training-step probe 顯示 physical batch 16 可容納，2026-08-19 的完整 Bit-True validation 實測在後段
只剩約 0.48 GiB VRAM，因此 validation 與訓練批次分離並固定為 batch 8。A1、A2、Phase B 仍維持
physical batch 16、`nbs=16`，只有 Phase C 使用 physical batch 8、`nbs=16`（accumulate 2）。兩個架構在
相同 phase 都使用相同契約；Full35 B 仍需由 accepted A2 重跑，是因本次 `fraction=0.3` 的比較邊界，不是
因為要把 Phase B 改成 batch 8。

## DataLoader workers 與主機 RAM

正式 `workers=6`、`fraction=0.3`、AMP 可見且 pinned memory 啟用的實際 COCO2017 量測可用下列命令
重現：

```bash
$PYTHON scripts/profile_dataloader_workers.py \
  --workers 6 --batch 8 --fraction 0.3 --measured-batches 100 \
  --output artifacts/profiles/dataloader-workers6-batch8-fraction03.json
```

Ultralytics 8.4.90 在本環境使用 `prefetch_factor=4`。先前 batch-16 實測載入 35,486 張訓練影像，
DataLoader PSS 增量約 2.08 GiB、穩態 17.28 ms/batch、926 images/s；這可作為 batch-8 queue 的保守 RAM
上界，workers 6 有充分餘裕。`fraction=0.3` 只縮短 training epoch；validation 仍使用完整 5,000 張
COCO2017 validation split。

Phase B 的 batch 16 每個 epoch 約有 2,218 batches，每個 batch 更新一次；Phase C 的 batch 8 每個 epoch
約有 4,436 microbatches，每兩個 microbatches 更新一次。正式執行還要加上完整 validation、checkpoint、
EMA 與資料處理；Phase C 可能由 patience 9 提前停止。

## 目前 30% 正式續跑佇列

舊工作站的 Full35 已跑完 B、Partial75 正要跑 B；為避免用不同契約比較，新的佇列不沿用 Full35 舊 B。
請從舊工作站各取出一個「Phase A2 gate 完成後實際被接受、且尚未進入 B」的 Float checkpoint。它們可以
直接續跑，不必回到 prepared checkpoint；匯入後會先以目前固定版本檢查架構與 Float-PWL backend，重新
materialize Bit-True checkpoint 並跑完整 COCO validation，然後才啟動新 B。

匯入命令會複製 checkpoint、計算 SHA256，並建立可稽核的 `candidate.json`。不要把 Full35 舊 B 的
`best.pt` 誤填成 accepted A2：

```bash
$PYTHON scripts/import_fraction03_source.py \
  --variant full35 --checkpoint /PATH/TO/FULL35_ACCEPTED_A2_FLOAT_BEST.pt
$PYTHON scripts/import_fraction03_source.py \
  --variant partial75 --checkpoint /PATH/TO/PARTIAL75_ACCEPTED_A2_FLOAT_BEST.pt
```

來源固定落在：

- `inputs/continuation/full35-accepted-a2/candidate.json`
- `inputs/continuation/partial75-accepted-a2/candidate.json`

先確認 `$PYTHON -m achitechure_1.cli preflight` 為 `valid: true`。下列命令不是 dry-run；兩個來源齊全時會
持續執行 Full35 B → Partial75 B → Full35 C → Partial75 C，每一步都含 Bit-True validation 與 gate：

```bash
$PYTHON scripts/run_fraction03_continuation_queue.py \
  --run-tag rtx5060ti-fraction03-b16-phasec8x2-workers6-r3
```

若來源尚未同步，命令只會寫入 `waiting_for_inputs` 後安全結束，不會使用
`artifacts/prepared/*.pt` 代跑。來源補齊後以相同命令重啟即可。佇列狀態位於
`artifacts/queues/fraction03-rtx5060ti-fraction03-b16-phasec8x2-workers6-r3/state.json`；已完成且 metadata、
parent path、SHA256、phase 專屬 batch、validation batch、workers、fraction、AMP 都吻合的工作會直接復用，
不重跑。舊的 `batch8x2-workers6-r2` 曾誤把 Phase B 設為 8×2，已停止並標記無效，不得拿來續接或比較。

每個 GPU job 啟動前至少要有 8 GiB 可用 RAM 與 12 GiB 可用 VRAM；不足時每 30 秒等待。訓練期間每
100 batches 記錄一次資源，每個 epoch 在 validation/checkpoint 後執行 GC、CUDA cache 清理與
`malloc_trim`。各 run 的紀錄在 `artifacts/runs/<run-id>/resource-telemetry.jsonl`。若可用 RAM 低於
0.5 GiB 或可用 VRAM 低於 1 GiB，該子程序會 fail closed；不會自動縮 batch。每個子程序退出後，佇列也
會等 RAM/VRAM 回到啟動門檻才執行下一步。

這裡不執行 Linux `drop_caches`：檔案 cache 本來就包含於 `MemAvailable` 的可回收估算，強制清除會讓
下一個 epoch 重新讀取資料，通常更慢且無助於避免 process OOM。

## 完整資料 Phase B 對照與 Phase C

30% queue 完成且兩個 Phase B child 都 rollback 後，使用同一對 accepted A2 source 執行
`fraction=1.0` Phase B。腳本只覆寫 fraction；Phase B 仍固定 physical batch 16、`nbs=16`、workers 6、
patience 4、validation batch 8：

```bash
$PYTHON scripts/run_fraction10_phase_b_control_queue.py \
  --run-tag rtx5060ti-fraction10-b16-workers6-r1 \
  --minimum-free-vram-gib 9
```

完成 state 位於
`artifacts/queues/fraction10-phase-b-rtx5060ti-fraction10-b16-workers6-r1/state.json`。Full35／Partial75
child 的 Bit-True mAP50-95 分別為 `0.5035027108`、`0.5040094610`，仍低於各自 A2，因此 accepted parent
仍為 A2。

Phase C 從上述 completed state 的 accepted candidates 接續，使用 physical batch 8、`nbs=16`
（accumulate 2）、workers 6。基準 `phase-c.yaml` 的 patience 是 9；下列 2026-08-22 queue 依使用者要求
明確覆寫為 7，並把 override 寫進 state 與 training manifest：

```bash
$PYTHON scripts/run_fraction10_phase_c_queue.py \
  --run-tag rtx5060ti-fraction10-phasec8x2-workers6-patience7-ram05-r2 \
  --source-state \
  artifacts/queues/fraction10-phase-b-rtx5060ti-fraction10-b16-workers6-r1/state.json \
  --workers 6 \
  --patience 7 \
  --minimum-available-ram-gib 0.5 \
  --minimum-free-vram-gib 9
```

State 位於
`artifacts/queues/fraction10-phase-c-rtx5060ti-fraction10-phasec8x2-workers6-patience7-ram05-r2/state.json`。
先前 `patience7-r1` 在第一個 epoch 完成前觸發舊的 1.5 GiB RAM floor，沒有可續接的 `last.pt`；其 run
目錄保留為中斷證據。`ram05-r2` 因此從相同 accepted Phase B checkpoint 重啟 Phase C，不重跑 Phase B。
重開 shell、WSL 或主機後，以相同命令接續；存在合法 `manifest.json` 與 `last.pt` 時，queue 會驗證契約後
傳入 `--resume-incomplete`，不從 epoch 1 重跑。不要用不同 run tag 指向同一個 incomplete run。

此 queue 已於 2026-08-23 完成。Full35／Partial75 分別在 10／11 epochs early stopping；Bit-True
COCO mAP50-95 為 `0.5013946996`／`0.5004282491`，均低於各自 A2，因此 rollback。最終報告位於
`final/reports/RTX5060TI_FINAL_0824.md`；`results/rtx5060ti-half-0822.md` 保留為當時刻意排除 Phase C 的
階段性快照，不回填或改寫。

## Ball／bat detection 驗證

將 `/home/uxin0/yolo/original/pose/dataset` 的兩類 pose bbox 建成 detection view：

```bash
$PYTHON scripts/validate_ball_bat.py prepare \
  --source /home/uxin0/yolo/original/pose/dataset \
  --output /home/uxin0/yolo/original/pose/detect_dataset
```

`data.yaml` 是 ball=0／bat=1 的兩類資料；既有 COCO80 detector 必須使用
`coco80/data.yaml`，把來源映射到 sports ball=32／baseball bat=34。驗證單一 Bit-True checkpoint：

```bash
$PYTHON scripts/validate_ball_bat.py validate \
  --checkpoint /PATH/TO/BITTRUE.pt \
  --data /home/uxin0/yolo/original/pose/detect_dataset/coco80/data.yaml \
  --run-dir artifacts/validation/ball-bat-example \
  --batch 8 --workers 6
```

目前 valid 有 567 images、301 ball、484 bat instances；93 images 的原始 COCO ID 與 COCO train2017
重疊，因此只適合 checkpoint 間相對比較，不可稱為完全獨立泛化測試。

## 歷史 Full35 → Partial75 recovery queues（不可續接）

For the RTX 4080 SUPER recovery path, the completed Full35 Phase A1 can feed an autonomous, fail-closed queue. It
finishes Full35 before starting Partial75, and applies the same Bit-True validation and rollback gates to each
architecture's A1 → A2 → B → C trajectory. Every job fixes `batch=16` and `nbs=16`. A training child uses four
training DataLoader workers and zero simultaneously resident validation workers, preventing Ultralytics from turning
`workers=4` into 4 training plus 8 validation workers. A standalone formal validation uses four workers. Every job
runs in its own child process and waits for at least 3 GiB of available system RAM before launch. A completed child
exits before its successor starts, releasing DataLoader workers, shared memory, and CUDA state.

Phase C starts only when the detected GPU has that architecture's measured batch-16 peak plus one GiB of headroom.
If Phase C cannot fit, the queue records it as pending and continues the other architecture instead of launching a
known OOM configuration. After all currently safe work is complete, the final status is
`waiting_for_phase_c_gpu`.

```bash
$PYTHON scripts/run_architecture_recovery_queue.py \
  --run-tag rtx4080super-batch16-workers4-r1
```

Queue progress is written atomically to
`artifacts/queues/architecture-rtx4080super-batch16-workers4-r1/state.json`.

To continue after an already gated Full35 Phase B with at most eight resident workers, use a new hardware/settings
tag. This path requires at least 7 GiB of available RAM before every child process, retains the Phase-C VRAM gate,
and starts Partial75 without repeating Full35:

```bash
$PYTHON scripts/run_architecture_recovery_queue.py \
  --run-tag rtx4080super-batch16-workers8-r2 \
  --workers 8 \
  --continue-after-full35-state \
  artifacts/queues/architecture-rtx4080super-batch16-workers4-r1/state.json
```

The example below uses Full35. Repeat identically with `--variant partial75`, an `a2-partial75-*` run-id prefix,
and `artifacts/prepared/a2-partial75.pt`.

When the batch-16 queue reaches its Phase-C handoff, the recovery relay can test and run Phase C with a physical
microbatch of 8 and `nbs=16` (two-step gradient accumulation, effective batch 16). It keeps six DataLoader workers,
uses zero simultaneous in-training validation workers, and retains batch 16 for standalone Bit-True validation.
Before either architecture starts formal Phase-C epochs, a two-step real-loss profile must complete and its measured
peak plus 1 GiB must fit on the detected GPU. An OOM or insufficient margin is recorded as pending instead of being
retried blindly. The relay also recognizes the older workers=4 manifest-verification mismatch after a completed A1
and resumes from that completed checkpoint without interrupting or repeating the training job:

```bash
$PYTHON scripts/run_phase_c_recovery_after_queue.py \
  --source-state artifacts/queues/architecture-rtx4080super-batch16-workers6-r3/state.json \
  --source-run-tag rtx4080super-batch16-workers6-r3 \
  --source-full35-state artifacts/queues/architecture-rtx4080super-batch16-workers4-r1/state.json \
  --run-tag rtx4080super-batch8x2-workers6-r4 \
  --workers 6
```

## 目前契約的手動 staged workflow

### Phase A1

```bash
$PYTHON -m achitechure_1.cli train \
  --variant full35 --phase a1 \
  --weights artifacts/prepared/a1-full35.pt \
  --run-id a1-full35-phase-a1 --execute

$PYTHON -m achitechure_1.cli materialize-bittrue \
  --checkpoint artifacts/runs/a1-full35-phase-a1/ultralytics/weights/best.pt \
  --output artifacts/checkpoints/a1-full35-phase-a1-bittrue.pt --execute
$PYTHON -m achitechure_1.cli validate \
  --checkpoint artifacts/checkpoints/a1-full35-phase-a1-bittrue.pt \
  --run-id a1-full35-phase-a1-bittrue --execute
```

### Phase A2 and rollback gate

Phase A2 runs for at most ten epochs and stops after four consecutive Bit-True validation epochs without
improvement. `best.pt` remains the highest-scoring observation.

```bash
$PYTHON -m achitechure_1.cli train \
  --variant full35 --phase a2 \
  --weights artifacts/runs/a1-full35-phase-a1/ultralytics/weights/best.pt \
  --run-id a1-full35-phase-a2 --execute

$PYTHON -m achitechure_1.cli materialize-bittrue \
  --checkpoint artifacts/runs/a1-full35-phase-a2/ultralytics/weights/best.pt \
  --output artifacts/checkpoints/a1-full35-phase-a2-bittrue.pt --execute
$PYTHON -m achitechure_1.cli validate \
  --checkpoint artifacts/checkpoints/a1-full35-phase-a2-bittrue.pt \
  --run-id a1-full35-phase-a2-bittrue --execute

$PYTHON -m achitechure_1.cli gate \
  --parent-name phase-a1 --parent-map PARENT_MAP \
  --child-name phase-a2 --child-map CHILD_MAP
```

Use the selected phase's Float `best.pt` as Phase B input. A loss of exactly 0.001 is accepted; only a loss greater
than 0.001 rolls back.

### Phase B and rollback gate

Phase B uses the same patience-four rule with a maximum of ten epochs.

```bash
$PYTHON -m achitechure_1.cli train \
  --variant full35 --phase b --weights ACCEPTED_PHASE_A_FLOAT_BEST_PT \
  --run-id a1-full35-phase-b --execute
$PYTHON -m achitechure_1.cli materialize-bittrue \
  --checkpoint artifacts/runs/a1-full35-phase-b/ultralytics/weights/best.pt \
  --output artifacts/checkpoints/a1-full35-phase-b-bittrue.pt --execute
$PYTHON -m achitechure_1.cli validate \
  --checkpoint artifacts/checkpoints/a1-full35-phase-b-bittrue.pt \
  --run-id a1-full35-phase-b-bittrue --execute
$PYTHON -m achitechure_1.cli gate \
  --parent-name phase-a --parent-map PARENT_MAP \
  --child-name phase-b --child-map CHILD_MAP
```

Use the selected Float checkpoint as Phase C input.

### Phase C and strict parent retention

```bash
$PYTHON -m achitechure_1.cli train \
  --variant full35 --phase c --weights ACCEPTED_PHASE_B_PARENT_FLOAT_PT \
  --run-id a1-full35-phase-c --execute
$PYTHON -m achitechure_1.cli materialize-bittrue \
  --checkpoint artifacts/runs/a1-full35-phase-c/ultralytics/weights/best.pt \
  --output artifacts/checkpoints/a1-full35-phase-c-bittrue.pt --execute
$PYTHON -m achitechure_1.cli validate \
  --checkpoint artifacts/checkpoints/a1-full35-phase-c-bittrue.pt \
  --run-id a1-full35-phase-c-bittrue --execute
$PYTHON -m achitechure_1.cli gate \
  --policy phase-c-improve \
  --parent-name phase-b --parent-map PARENT_MAP \
  --child-name phase-c --child-map CHILD_MAP
```

Training itself converts a deep copy of the Float EMA to Bit-True every epoch, so `best.pt` and early stopping are
driven by Bit-True mAP50-95. Patience is four for A2/B and nine for Phase C. Phase C retains its parent unless the
child strictly improves.

## Architecture winner

Create one JSON file per A1/A2 candidate with these selection keys:

```json
{
  "name": "a1-full35",
  "map50_95": 0.0,
  "fp16_p50_ms": 0.0,
  "gflops": 0.0,
  "parameters": 0
}
```

Latency must be measured for both candidates on the same formal GPU（目前為 RTX 5060 Ti）with identical settings. An optional
`peak_vram_gib` field may be retained for diagnostics, but the selector deliberately ignores it.

Then select:

```bash
$PYTHON -m achitechure_1.cli select \
  --candidate results/a1-candidate.json --candidate results/a2-candidate.json
```

## T1/T2/T3 winner continuations

All three commands must reference the exact same accepted Float architecture-winner checkpoint:

```bash
$PYTHON -m achitechure_1.cli train --variant WINNER_VARIANT --phase tune --masf-lr 0.0002 \
  --weights WINNER_FLOAT_PT --run-id t1-lr-0002 --execute
$PYTHON -m achitechure_1.cli train --variant WINNER_VARIANT --phase tune --masf-lr 0.00038 \
  --weights WINNER_FLOAT_PT --run-id t2-lr-00038 --execute
$PYTHON -m achitechure_1.cli train --variant WINNER_VARIANT --phase tune --masf-lr 0.0006 \
  --weights WINNER_FLOAT_PT --run-id t3-lr-0006 --execute
```

Materialize and validate each best checkpoint using the same commands as above.

## Final profile and delivery

```bash
$PYTHON -m achitechure_1.cli profile \
  --checkpoint FINAL_BITTRUE_PT --kind inference \
  --output artifacts/profiles/final-inference.json --execute
$PYTHON -m achitechure_1.cli export-masf \
  --checkpoint FINAL_BITTRUE_PT --output results/masf-only.pt --execute
sha256sum FINAL_BITTRUE_PT results/masf-only.pt inputs/parent/best.pt
git status --short --branch
```

Copy, without reserializing, the selected reloadable Bit-True checkpoint to `results/best.pt`; record its SHA256 and
source Float checkpoint in the final results row. Update `results/results-template.csv`, including AP_S/M/L from
canonical COCO API, class 32/34 AP, recall, final alpha, latency, Params, GFLOPs, VRAM, total time, curves, best/stop
epochs, and stop reason. Explicitly retain the A0 causal-limitation statement from `EXPERIMENT_SPEC.md`.
