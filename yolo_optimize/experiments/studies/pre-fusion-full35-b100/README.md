# 融合前 Full35-B100：方向 1

> 本目錄已實際移入 `experiments/`。內部 run 名與模型內容保留；操作請見[整理後指南](<../../../docs/OPERATIONS.md>)。

狀態：BinaryQK 梯度恢復、narrow／late 適應、HOG、RepConv17、P3 shared／fork／bridge、P2 最後對照均已完成或依門檻停止。使用者選 P3 bridge E8 接續後續融合；這不代表它已明顯勝過無 MASF 配對。

| 找什麼 | 位置 |
| --- | --- |
| 結果／架構／成本 | [方向 1 報告](<../../../reports/direction1/README.md>) |
| P3／P2 BBAT 比較 | [MASF 報告](<../../combine/pose-masf/RESULTS.md>) |
| 程式 | scripts/；保留固定目錄深度 |
| 每組訓練與診斷 | artifacts/；含 checkpoint、summary、失敗紀錄 |
| 已驗證候選 | artifacts/direction1-candidate-verification-v1/ |
| 模型選擇與完整清單 | [權重導航](<../../../reports/checkpoints/README.md>) |

P3 bridge COCO AP 0.508212／person 0.627664；仍低於 FP，額外模組收益未過方法 gate。PWL 保持 [-10,0]／20 段。資料是完整 COCO80；BBAT 觀察使用 canonical v1。沒有新增 P2 Detect head，也沒有本研究 P2 Pose 訓練成果。

原始分段計畫、當時的 queue 狀態與全部文字保留於[歷史入口](<../../../docs/history/README.md>)。新工作應另開 run，不直接重啟舊 queue。
