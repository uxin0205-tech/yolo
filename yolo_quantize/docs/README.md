# 文件導航

日常只需：[階段收尾報告](reports/2026-09-08-quantization-phase-handoff.md) → [目前計畫（量化延後）](CURRENT_PLAN.md) → [比較限制](reports/2026-09-07-evidence-consistency-boundaries.md)。六組 QAT 及十二個累積 PTQ 已完成，不自動接新訓練。

日後恢復：[RESUME](RESUME.md)，先確認 parent、資料與同 benchmark，再另行核准下一批。

| 資料夾 | 角色 |
| --- | --- |
| [reports](reports/README.md) | 現行結果与必要解釋，其他報告視為歷史快照 |
| [worklogs](worklogs/README.md) | 修改、驗證、問題與決策的完整時間紀錄 |
| [archive](archive/README.md) | 歷史索引、整理前入口；不作現行訓練指令 |
| [organization](organization/README.md) | 全資料夾盤點、保留理由及已核准整理的執行紀錄 |
| research | 論文與方法來源，作研究參考，不等同實驗已執行 |
| superpowers | V36 的既有設計／實作規劃快照；目前以 CURRENT_PLAN 為準 |

有必要依賴的五份舊報告保留原位；十二份獨有歷史報告已移入 archive/reports 並修正連結。歷史 checkpoint、原始 artifact 與工作紀錄不因報告整理而刪除。
