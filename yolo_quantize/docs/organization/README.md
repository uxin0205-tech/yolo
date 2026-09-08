# 專案整理

本輪先讓「閱讀入口」清楚，同時保護訓練與證據路徑。

最新：[已核准報告清單](report-retention-2026-09-08.md)、[機讀盤點](report-retention-2026-09-08.json)與[執行證據](report-retention-execution-2026-09-08.json)。25 份報告整理完成：3 份備份後刪除、12 份移入歷史區、10 份保留原位。量化維持延後；日後從[恢復交接表](../RESUME.md)接續。

本次報告備份：`/tmp/yolo-quantize-report-backup-0908.rJlv9s/`，包含原 25 份報告與受連結調整影響的文件。備份 hash 已核對；/tmp 不是永久備份，必要原始實驗資料仍在原位。詳見[整理與發布工作紀錄](../worklogs/2026-09-08-report-cleanup-and-publication.md)。

- [精確盤點與待核准清除清單](cleanup-proposal-2026-09-08.md)：候選 ID、大小、保留原因、風險與恢復方式。
- [機讀檔案分類與現行引用閉包](inventory-2026-09-08.json)：一般檔明細；runtime 影像 symlink 只彙總計數，不重複展開成新資料版本。
- [整理工作紀錄](../worklogs/2026-09-08-project-organization.md)。

## 已做的整理

1. 主 README 只保留本輪目標、有效結果、固定規則與必要入口。
2. 現行計畫集中到 `docs/CURRENT_PLAN.md`；舊 v5、3epoch 與封存 queue 不再並列現行入口。
3. 報告／設定／腳本索引分成現行與歷史；整理前四份入口完整保留於 `docs/archive/`。
4. artifacts、程式、測試及可分享成品有各自閱讀說明，不要求使用者逐一理解所有舊版本檔名。

## 不做的事

不停止 GPU、不重命名或搬移既有 configs／runs／queue／資料集、不回寫已發表數字、不將「沒看到引用」當作可刪的證明。C01–C05 已核准並移出至 /tmp 保留恢復機會；C06 保留。精確位置、大小及驗證見工作紀錄；舊清單保留為執行前快照。

重建：`/home/uxin/yolo/.venv/bin/python scripts/inventory_project.py`。機讀清單是磁碟快照，訓練可繼續新增檔案；不把快照與即時大小要求逐位元相同。
