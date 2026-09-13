# 2026-09-13：教授版 YOLO Optimize 完整研究報告

## 需求

將 `yolo_optimize` 從 BinaryQK 起，依序整理 HOG、RepConv、MASF、Detect＋Pose 融合、activation、KD、推論與成本比較；同時保留失敗實驗、架構演進與研究限制，改寫成教授／論文審查可直接閱讀的研究敘事。

## 處理

新增 `reports/professor-overview/README.md`。報告不沿用單純時間軸，而以「研究問題 → 假說 → 對照 → 結果 → 決策」整理。數據只引用 Git 中既有正式完整驗證與 performance benchmark，不重算 AP。

2026-09-13 整理期間 `main` 又新增 `5090 Done 0913`：原生 QK＋PWL E2 已完成後暫停，Pose MASF priority 完成全量比較與成本，Pose MASF B-only 專項重訓只完成 CPU 前置。教授版報告已以此最新狀態更新，不將舊「尚未開始」文字冒充目前狀態。

## 限制

本文是教授版整合敘事，不取代 `reports/final/README.md` 的原始研究帳本。尚未完成的 native-QK 長期恢復、scale_bias、Pose MASF B 組 GPU 訓練、完整 weight PTQ/QAT、target hardware 均明確標示，不冒充成果。
