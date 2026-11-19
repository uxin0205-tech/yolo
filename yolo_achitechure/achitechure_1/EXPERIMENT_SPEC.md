# achitechure_1 experiment contract

Status: implementation complete。兩個 accepted A2 已從 commit `7f0bd61` 匯入並重新驗證；
`fraction=0.3` 的對稱 B／C 與 `fraction=1.0` Phase B 對照均完成，所有 child 均由 gate rollback 到 A2。
`fraction=1.0` Phase C 正以 batch 8 × accumulate 2、workers 6、patience 7 執行；完成前不得納入
2026-08-22 階段性結論。

## Scope and authority

This document is the formal contract for the YOLO26m Attention + P3 MASF experiment. It overrides conflicting
language in `references/plan.md`:

- `Partial75` means 25% context channels and 75% exact identity bypass.
- Both DW3 and DW5 outputs are summed and then passed through a required `C→C` 1×1 projection.
- This is a MASF-inspired P3 adaptation. It is not a claim of reproducing the complete MASF-YOLO paper.

No P2 feature or Detect input is added. P4/P5, Detect, loss, assignment, augmentation, and the two
hardware-friendly attention sites are not redesigned.

## Immutable parent

The immutable A0 parent is `inputs/parent/best.pt`, SHA256
`c989aeed09de7663ad093d32d098e5fc889cf04924fa1162efaf886869de0123`. The completed attention study ran all
19/19 queued jobs, including the seed-0 LR sweep and staged recovery. Its formal gate retained the verified
zero-train Bit-True checkpoint because the best trained observation improved mAP50-95 by only `0.0002017689` and no
three-seed mean was run. The completed final manifest, recipe, selector, queue state, and event history are preserved
under `inputs/parent/provenance/`. The unselected best-observed checkpoint is recorded by SHA256 but is not used as
A0.

The actual graph is end-to-end YOLO26m with Detect inputs `[16, 19, 22]`, strides `[8, 16, 32]`, and exactly two
hardware-friendly attention sites: `model.10.m.0.attn` and `model.22.m.0.1.attn`.

## Architecture

The P3 seam is discovered from the first actual Detect input and then validated. The existing C3k2 instance is
upgraded in place to a subclass, so graph index 16 and every original state-dict name remain unchanged.

For context tensor `x`:

```text
delta = Conv1x1(DWConv3x3(x) + DWConv5x5(x))
y = x + alpha * delta
alpha = learnable scalar initialized to 0.01
```

- A0: immutable parent, no MASF and no retraining.
- A1 Full35: all 256 P3 channels use the context path.
- A2 Partial75: the first 64 P3 channels use the context path; the final 192 channels are concatenated back without
  any operation and are bit-exact.

The three convolutions follow the installed Ultralytics Conv → BatchNorm → SiLU convention. DW3 and DW5 are true
depthwise convolutions; the projection is a dense 1×1 convolution.

## Common training contract

目前正式續跑以兩個架構各自「完成 Phase A2 gate 後所接受的 Float checkpoint」為共同邊界。舊工作站已跑完
的 Full35 Phase B 不直接沿用；Full35 與 Partial75 都在這台機器以相同新契約重跑 B，再依序跑 C，避免
兩個架構使用不同 fraction／workers 契約。若日後從 prepared checkpoint 重建 A1/A2，亦必須使用同一份
新契約。

基準設定使用 COCO2017 training split 的 30%（`fraction=0.3`）、image size 640、seed 0、六個 workers、
deterministic mode、AMP、相同 Ultralytics 8.4.90 augmentations 與 MuSGD。A1、A2、Phase B 固定 physical
`batch=16`、`nbs=16`，不使用梯度累積；只有 Phase C 固定 physical `batch=8`、`nbs=16`，每兩個
microbatches 更新一次（accumulate 2，effective batch 16）。Validation 與訓練批次分開，使用完整 5,000
張 COCO2017 validation split 且固定 batch 8。自動 OOM batch reduction 停用；若仍無法容納，會停止而不
偷偷改變 batch。

為隔離 30% 資料量的影響，Phase B 另有 `fraction=1.0` 對照；它從完全相同的 accepted A2 Float
checkpoints 開始，只覆寫 fraction，不修改 Phase B batch、workers、patience 或 optimizer 契約。後續
`fraction=1.0` Phase C 同樣從 gate 接受的 parent 開始，使用 batch 8 × accumulate 2；依使用者指定，此
queue 將 Phase C patience 明確覆寫為 7。這是有 manifest／queue state 的 run-specific override，不回寫
`phase-c.yaml` 的基準 patience 9。

Every phase creates a new trainer and therefore a fresh optimizer, scheduler, and EMA. Frozen BatchNorm running
buffers and frozen attention train/eval state are enforced immutable. Detect remains in training mode because
putting a frozen Detect module into eval changes its public output from training dictionaries to inference tuples.

## Resource safety and queue contract

所有 train、materialize、validate 與 profile job 都在獨立子程序執行。每個 GPU job 啟動前至少需要
8 GiB 可用系統 RAM；30% queue 的預設 VRAM 門檻為 12 GiB，完整資料 B／C 對照依 queue state 明確覆寫
為 9 GiB。資源不足時佇列每 30 秒重新檢查，不啟動工作。子程序退出後也要恢復到同一 queue 的門檻，
才可啟動下一個 job。訓練內每 100 batches 記錄一次 RAM、process RSS/PSS、VRAM
free/allocated/reserved/peak；可用 RAM 低於 1.5 GiB 或可用 VRAM 低於 1 GiB 時 fail closed。

每個 epoch 的 validation 與 checkpoint 完成後會執行 Python GC、`torch.cuda.empty_cache()` 與 Linux
`malloc_trim(0)`，並留下清理前後的 telemetry。這只釋放未使用的 allocator cache，不會卸載仍被模型或
DataLoader 使用的記憶體；完整的 CUDA context、workers 與 shared memory 由子程序結束負責回收。

## Fixed phases

| Phase | Epochs | Trainable scope | LR groups | Warmup | Schedule | Patience |
|---|---:|---|---|---:|---|---:|
| A1 | 5 | MASF only | MASF `1.0e-3` | 0.5 | constant | fixed phase |
| A2 | ≤10 | MASF only | MASF `3.8e-4` | 0 | cosine to 50% | 4 |
| B | ≤10 | MASF + actual Neck/Detect, attention frozen | MASF `3.8e-4`; Neck/Detect `1.9e-4` | 1 | cosine to 50% | 4 |
| C | ≤70 | full model | MASF `3.8e-4`; Neck/Detect `1.9e-4`; Backbone `3.8e-5`; attention `5e-6` | 1 | cosine to 10% | 9（完整資料對照覆寫為 7） |

Momentum is 0.948 and Conv weight decay is 0.00027. Alpha, biases, and BatchNorm parameters have no decay.
Every epoch's best/early-stop metric is measured on a deep-copied, actually converted Bit-True PWL EMA model;
Float-PWL remains the differentiable training model.

A2 and B stop after four consecutive epochs without a higher Bit-True mAP50-95, then roll back when the child loses
more than 0.001 to its accepted parent. Phase C retains its parent unless it strictly improves it. A1 remains a
fixed five-epoch phase.

## Winner tuning and selection

A1 versus A2 is selected by Bit-True COCO2017 mAP50-95. Within 0.001, lower same-GPU FP16 p50 latency wins,
followed by lower GFLOPs and Params. Peak VRAM is recorded only as a capacity diagnostic and never ranks candidates。
2026-08-19 的正式訓練與最終 comparable profiling 使用同一張 RTX 5060 Ti。The architecture winner then starts
T1/T2/T3 from the exact same Float winner checkpoint with MASF LR 0.0002, 0.00038, or 0.0006. Other LR groups
preserve Phase-C ratios. Each continuation runs at most ten epochs with patience five and a fresh optimizer.

Formal candidates are A0, accepted A1/A2 best, and T1/T2/T3 best. Reports must include mAP50-95, mAP50, mAP75,
AP_S/M/L, recall, class-32 sports-ball AP, class-34 baseball-bat AP, alpha, Params, GFLOPs, FP16 p50 latency, peak
VRAM, epoch time, curves, best/stop epoch, and stop reason.

## Causal limitation

A0 is not retrained. Any A1/A2 difference from A0 combines the MASF change with staged retraining and cannot be
claimed as a pure architectural causal gain.
