# 實驗產物

本處保留歷史指標、配置與曲線來源。2026-10-19 依使用者指示，已刪除被取代的 run checkpoint 及訓練預覽／曲線圖片；必要選定權重由相容連結指向 `yolo_attention/weights/`。queue／events 保持原始歷史，不偽造新狀態，但舊 run 不再全部可 resume。逐檔 SHA 與刪除項目見 [清理紀錄](../../yolo_attention/artifacts/stage3/cleanup-final-delivery.json)，保留證據見 [索引](../../yolo_attention/artifacts/stage3/historical-evidence-index.json)。

- `lr-sweep-queue/`：正式完成的 queue，revision 95，19/19 個工作成功。
- `runs/lr-*/`：LR pilots、Bit-True 評估、staged recovery、gates 與最高觀測值選擇。
- `runs/s0-phase-b-bittrue/`：Git 唯一納入的 run 目錄；fresh clone 後執行 `final/run.py train --execute` 所需的不可變 parent。
- `runs/epoch0-*/`、`runs/pilot-*/`、`runs/s0-*/`：保留的 parent lineage 與 negative controls。
- `queue/`：已停止的 Attention-only 多 seed 歷史，只保留 provenance，不會自動啟動。

最終清理已移除初步的 `recovery-queue/`、重複 parent 評估與中斷的 block run。人類可讀報告位於 `../final/RESULTS.md`；`final-selection.json` 記錄為何 `../final/` 保留正式 checkpoint，而 block x1 只能列為 best-observed。
