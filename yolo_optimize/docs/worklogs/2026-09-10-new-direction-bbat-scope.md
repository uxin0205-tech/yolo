# 新方向 checkpoint 與 BBAT5 Detect／Pose 範圍更正

## 使用者要求與更正

使用者明確指向 optimize 新方向重新訓練的模型；Pose 指 BBAT5 資料集，希望 Detect／Pose 都驗證。既有比較的七組 Detect checkpoint 全部來自 `studies/pre-fusion-full35-b100/artifacts/`，這部分來源符合要求；原舊 Pose checkpoint 的 MASF 開／關不屬此次新方向比較，必須排除於本次採用決策，不將其 box 改善套用到新 P2／P3。

## 唯讀核對與結果

本次不啟動 GPU，不重訓。直接載入 pickle checkpoint 的檢查被安全審核拒絕，因此改用 `zipfile`／`pickletools` 靜態解析，不反序列化或執行 pickle。新 P2 E5 與 P3 bridge E8 的序列化類別符號及研究契約皆指向 DetectionModel／Detect，沒有已訓練 Pose head。J0 雙任務驗證結果已存在，但僅無 MASF 一組，不能冒稱已完成 P2／P3 Pose 成對比較。

核對正式 `configs/detect.yaml` 與 `configs/pose.yaml`：val 各 683 張，逐檔影像內容一致；932 個 class／box 標註數值完全相同（Pose 標註前五欄與 Detect 五欄比較）。全部斷言通過。故新方向七個 checkpoint 已完成的 BBAT5 box AP，同時適用這兩個正式入口的同一組框，不需重複 GPU 驗證。

Detect-only 權重不能輸出 keypoints；若要比較新方向 P2／P3 的 Pose AP，需要各自有訓練完成的 Pose head，不能硬接舊 head 的結果當正式結論。目前沒有本次研究可用的成對權重，不擅自重訓或恢復 queue。

## 困難、未解事項與停止狀態

困難為 checkpoint 非安全 pickle 載入限制，已用不執行程式的靜態檢查完成必要核對，無需解除該限制。未解事項是新 P2／P3 權重沒有 Pose head，故沒有對應 keypoint AP。原始數字保留，但明確標記旧 Pose 消融不適用此次問題。停止狀態不變，等待使用者決定；未刪除、提交或推送。
