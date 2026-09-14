# 2026-09-13：硬體友善候選佇列狀態

## 狀態核對 2026-09-13T22:31:00+08:00

使用者詢問目前狀況；只讀 queue 狀態、已完成摘要／報告、Rep17 產物名稱與 heartbeat，並確認三個 queue／training 程序存活，未讀訓練 log、未查 GPU、未重啟工作。Rep17 heartbeat 距核對約 1.0 秒。

MASF B5 已完成。scale/bias 完成 5 epoch 與 Float／BitTrue 獨立驗證；對共同原生 QK B5，COCO +0.3034 pp、person +0.1582 pp、ball Pose +0.4441 pp，但 bat 框 -0.4642 pp、bat Pose -0.5228 pp，因此未通過全指標保護門檻，未取代正式模型。這是改方法加適應訓練的結果，不能單獨歸因 scale/bias，也不是舊 BinaryQK 的同預算配對比較。

佇列已自動接續 Rep17 train，尚未有第一個完整回合驗證產物；Rep20 等待，Rep17＋20 接續器正常等待原三組完成。困難／新錯誤：無。未解事項：RepConv 三組精度尚待訓練與驗證；不因候選未過採用門檻而重跑或延長回合。
