# 保留的逐階段降 LR P3

依使用者要求保留本組結果。best.pt為第163輪綜合最佳，640統一評估Box mAP50–95=62.8808%、Pose=92.0616%。last與last-resumable對應第200輪；完整配置及results.csv一併保存。manifest.json記錄來源、大小與SHA256，副本已逐一核對。後續候選從這份best初始化，輸出至各自獨立run，不覆寫此目錄。

2026-09-18 清理後，本目錄為唯一保留的模型組；原 runs 來源已刪除，manifest 的 file 欄位仍有效，source 僅供追溯。正式評估見 evaluation.json，誤差診斷見 errors/report.md，三階段設定見 stage-configs/。
