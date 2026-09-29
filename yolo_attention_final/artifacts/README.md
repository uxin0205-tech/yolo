# 實驗產物

> **Cleanup-v1 發行範圍：** 本分支保留程式、設定、報告、統計圖表與 manifests；checkpoint、資料集影像／標註、逐筆預測、batch 預覽和冗長執行日誌已從此發行快照排除。下文的歷史權重路徑、數量、checksums 與實驗結果仍保留作研究紀錄；訓練、推論、完整交付驗證及資料重建需要另行提供原始資產，不代表 clone 後即可直接重跑。 詳見[清理範圍與資產需求](../../docs/cleanup-v1/README.md)。


所有 run 都視為不可變記錄，用來保存報告指標背後的證據。

- `lr-sweep-queue/`：正式完成的 queue，revision 95，19/19 個工作成功。
- `runs/lr-*/`：LR pilots、Bit-True 評估、staged recovery、gates 與最高觀測值選擇。
- `runs/s0-phase-b-bittrue/`：Git 唯一納入的 run 目錄；fresh clone 後執行 `final/run.py train --execute` 所需的不可變 parent。
- `runs/epoch0-*/`、`runs/pilot-*/`、`runs/s0-*/`：保留的 parent lineage 與 negative controls。
- `queue/`：已停止的 Attention-only 多 seed 歷史，只保留 provenance，不會自動啟動。

最終清理已移除初步的 `recovery-queue/`、重複 parent 評估與中斷的 block run。人類可讀報告位於 `../final/RESULTS.md`；`final-selection.json` 記錄為何 `../final/` 保留正式 checkpoint，而 block x1 只能列為 best-observed。
