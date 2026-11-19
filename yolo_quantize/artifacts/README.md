# 實驗產物：日常不必逐個打開

看結論請用[階段收尾報告](../docs/reports/2026-09-08-quantization-phase-handoff.md)，看安排用[目前計畫](../docs/CURRENT_PLAN.md)。量化已延後，現有 queue 完成，不接新訓練。本目錄保存原始證據與執行依賴，不以版本號新舊決定是否可刪。

| 位置 | 內容與保留規則 |
| --- | --- |
| [完成的累積 queue](queues/full-model-cumulative-0908/README.md) | 十二個 PTQ 結果、接受前綴與 phase-hold 延後決策；monitor 已退出 |
| [前批完整證據](queues/full-model-continuous-0907/README.md) | 鎖定 parent、40 PTQ、592 probe 與六組完成 QAT；保留、不重啟 |
| `queues/` 其他版本 | 舊 handoff、plan、hash、封存狀態；不自行重啟或搬移 |
| `runs/` | 訓練、驗證、checkpoint、export、runtime dataset view；約 52 GiB 的主要用量，未完成逐 run 安全刪除證明 |
| `manifests/` | 資料、模型、方法與發行血緣；保留 |
| `reports/` | 歷史原始 JSON/CSV，包含負面結果；不能當成現行 148 層 mAP |
| `datasets/` | canonical assignment 的可重建 runtime View；不建立新 split、不改 labels |
| `archives/` | 歷史封存與 handoff 證據，不代表可刪原 checkpoint |

checkpoint、cache、run 不隨報告自動發布至 GitHub。清除只依[核准清單](../docs/organization/cleanup-proposal-2026-09-08.md)，不使用整目錄遞迴刪除來節省空間。
