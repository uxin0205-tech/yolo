# 全模型量化盤點交付

> 本目錄是歷史發行快照，不是 live queue 的最新結果。1,332 筆 CPU 權重誤差與跨 parent PTQ 不得合併稱為逐層 mAP；現行比較條件見[血緣與證據層級釐清](../../docs/reports/2026-09-07-evidence-consistency-boundaries.md)，新 QAT 見[恢復結果](../../docs/reports/2026-09-07-continuous-qat-recovery-results.md)。

GitHub 0907發行包含本目錄輸出及CPU profile。`report_weight_evidence.py`可使用公開輸入重建；`report_full_model_audit.py`仍須本機未上傳的runs/checkpoint完成hash檢查，不能宣稱乾淨clone可直接重跑完整稽核。

新增：[SD4／三元詳細分析](../../docs/reports/2026-09-07-weight-distribution-detailed-analysis.md)／[數據附錄](weight-evidence-tables.md)。`weight-format-evidence.csv`為1332筆原權重摘要與量化誤差；`weight-format-nrmse.png/pdf/svg`是十區重建誤差圖，不是原權重histogram；`weight-evidence-provenance.json`記錄來源hash。重建：`/home/uxin/yolo/.venv/bin/python scripts/report_weight_evidence.py`，僅CPU。圖表、文檔、CSV可納入Git，checkpoint留本機。

主報告：[全模型盤點與四天計畫](../../docs/archive/reports/2026-09-07-full-model-audit-four-day-plan.md)。本目錄是本機證據快照，可直接複製分享。

- `qat-total-deltas.png/pdf/svg`：三個epoch相對accepted的最差總下降，圖中pp是百分點。
- `qat-epochs.csv`：每epoch兩族最差delta與deployment gate。
- `qat-metrics.csv`：每epoch16項原始值、accepted與total delta。
- `ptq-candidates.csv`：25個PTQ候選的total/incremental分欄；跨parent探索限制見主報告。
- `weight-sites.csv`：148個實際QAT權重sites、region、格式與elements。`fixed-sd4`為runtime格式名稱，QAT學習scale時解讀為LS-SD4。
- `audit.json`：queue完成狀態、CPU/PTQ/QAT parent、coverage與已驗證來源hash。
- `folder-inventory.csv`：目錄非symlink檔案數與bytes，為當次快照。
- `four-day-plan.json`：96小時規劃DAG；planned_not_enqueued，不是live executor輸入。

重建：在子專案執行 `/home/uxin/yolo/.venv/bin/python scripts/report_full_model_audit.py`。輸入為既有V36 artifacts；只使用CPU，輸出本目錄圖表/CSV/audit JSON。四天計畫另行版本維護；圖表數據可重算，未要求PDF/SVG逐位元穩定。Git僅交付文檔/表/圖/生成器，不攜帶checkpoint或資料。此次未提交/上傳。

## 清理盤點（尚未執行）

| ID | 精確路徑 | 類型/大小 | 依賴與原因 | 可恢復性/風險 | 建議 |
|---|---|---|---|---|---|
| C01 | /home/uxin/yolo/yolo_quantize/.pytest_cache | 測試cache，盤點時約60KiB | pytest可重建，不是研究metrics來源 | 可重建，低；失去lastfailed歷史 | 可清理，待指定ID授權 |
| C02 | /home/uxin/yolo/yolo_quantize/.ruff_cache | lint cache，盤點時約28KiB | ruff可重建，不是checkpoint依賴 | 可重建，低 | 可清理，待指定ID授權 |
| K01 | /home/uxin/yolo/yolo_quantize/artifacts/runs | 約45GiB | V35/V36 manifests、QAT與metric comparison引用；其餘依賴未逐一排除 | 不保證可重建，高 | 保留 |
| K02 | /home/uxin/yolo/yolo_quantize/artifacts/queues | 約2.2GiB | plan/hash、狀態、append-only events與失敗紀錄 | 丟失續跑/稽核，高 | 保留 |
| K03 | /home/uxin/yolo/yolo_quantize/artifacts/reports | 約55MiB | 研究數值原始證據 | 高 | 保留 |

未刪除任何cache、run或原始資料。finish-work第5節要求「Present a deletion manifest before requesting authorization」；若要清理，只需指定C01/C02，其他項目不在建議刪除範圍。
