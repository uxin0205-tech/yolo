# Partial75 workspace

這是 Detect–Pose 融合的備案工作區。權威 Partial75-A2 Float/Bit-True checkpoint、
COCO 與 BBT5 都視為唯讀；所有新輸出固定進入本資料夾的 `artifacts/`。

## 現況

- COCO Detect：既有 Partial75-A2 已有完整 Bit-True 評估。
- 原始 BBT5 Pose：既有 YOLO11m 歷史 baseline 已登錄。
- 架構匹配 Pose26：正式 batch128 P1 17／P2 22／P3 最多100 均尚未執行。
- F1 1:1/2:1 與 accuracy gate 尚未開始。
- Float/Bit-True 的獨立模型 → 共享模型 CPU forward 均逐值相同。
- Float shared → 官方 Detect/Pose graph CPU forward 均逐值相同。
- synthetic 與真實 COCO/BBT5 的 2:1 CPU optimizer step 均通過。
- 參數 45,442,522 → 26,460,581，減少 18,981,941（41.771%）。
- 本輪沒有啟動任何 GPU 工作。

詳細數值見 [不融合基線](baselines/independent.md)，機器可讀資料見
[independent.json](baselines/independent.json)。

## CPU-only 驗證

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python variants/partial75/run.py audit --verify-hashes
CUDA_VISIBLE_DEVICES="" .venv/bin/python variants/partial75/run.py cpu-check
CUDA_VISIBLE_DEVICES="" .venv/bin/python variants/partial75/run.py joint-smoke --ratio 2:1
```

`run.py` 已鎖定 Partial75，不接受 `--architecture`。報告與本地 dataset/cache 位於：

```text
artifacts/cpu-validation.json
artifacts/joint-smoke.json
artifacts/cache-views/coco-detect-smoke/
artifacts/cache-views/bbat5-v1/
```

## GPU 空閒後的獨立 Pose 流程

以下與 Full35 對稱的正式 P1 命令目前未執行：

```bash
.venv/bin/python variants/partial75/run.py pose \
  --stage p1 --batch 128 --epochs 17 \
  --name p0-partial75-p1-b128-e17-seed0
```

輸出會自動固定到 `variants/partial75/artifacts/pose/`。P2/P3 必須提供前一階段的
`best.pt`，且 wrong-architecture checkpoint 會 fail closed。

[training.yaml](configs/training.yaml) 已鎖定 batch128、P1 17、P2 22、P3
最多100；P1/P2 early stopping 關閉，P3 patience 20。Full35 的舊 P1 結果
不得當作 Partial75 Pose 精度。
