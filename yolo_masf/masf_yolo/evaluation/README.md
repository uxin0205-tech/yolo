# 評估模組

`runner.py` 執行推論與 faster-coco-eval；`metrics.py` 計算 per-class/Ball 指標；`selection.py` 僅用 validation 選 M2/M3；`profiling.py` 量測成本與延遲；`galleries.py` 產生誤報證據；`reference.py` 驗證 B0。正式數字在 `artifacts/static-phase1/evaluation/`，硬體數字在 `profiles/`。
