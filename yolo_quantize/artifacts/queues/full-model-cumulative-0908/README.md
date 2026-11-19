# 累積 PTQ queue：已完成，量化延後

十二個候選已完成，supervisor 與 monitor 均自然結束。使用者依老師建議延後量化，見 [phase-hold.json](phase-hold.json) 與[階段收尾報告](../../../docs/reports/2026-09-08-quantization-phase-handoff.md)。原 `decision_required` 只保留為執行證據，**不接新增 QAT，不自行重啟以下命令**。hold 是交接政策，不是 CLI 強制鎖。

輸入：上一輪的 V36 parent、592 probe、40 PTQ、六組完成 QAT；輸出：十區逐步累積與 W6/W5 的最多 12 個搜尋驗證結果。此 queue 不訓練、不使用 formal。

- [機讀計畫](plan.json)：精確路徑、格式、提名理由、來源與 runner SHA-256。
- `generated/`：每步依接受前綴固定的新計畫與路由。
- `execution-status.json`：本批終點證據，completed_jobs=12、error=null；本批監測已結束。
- `*-report.json`：各階段原始 mAP、parent、quantized coverage 與估算成本。
- `cumulative-selection.json`：每步候選、雙门檻、接受／保留原配置及來源雜湊。

CPU 準備：`PYTHONPATH=src python scripts/run_cumulative_ptq_queue.py`。
GPU 執行：同命令加 `--execute-reviewed-plan`；既有執行器在跑時不得再開。異常續跑需診斷後明確加 `--resume`，不刪除既有結果。

```bash
PYTHONPATH=src /home/uxin/yolo/.venv/bin/python -m yolo_quantize.blocking_monitor artifacts/queues/full-model-cumulative-0908/execution-status.json
```

人工說明见[累積計畫](../../../docs/reports/2026-09-08-cumulative-ptq-plan.md)。JSON／報告可依明確發布授權納入 Git；log、run、checkpoint 不自動上傳。舊 queue 保留為完整證據，不回寫舊執行狀態。
