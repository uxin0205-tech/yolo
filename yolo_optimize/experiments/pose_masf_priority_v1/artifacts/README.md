# Pose MASF 比較產物

此資料夾保存本次實際結果；人讀完整結論見 [RESULTS.md](<../RESULTS.md>)。

- `parent/`：固定 E2 推論副本，SHA 由上層 source-pin.json 鎖定。
- `comparison-v1/`：baseline、Pose MASF、Pose α=0 的正式驗證結果；Pose MASF 子目錄含候選 inference-only 權重。
- `summary-v1.json`：三組精度與差值；不是新訓練完成宣告。
- `benchmark-v1.json`：本機 CPU／GPU 模型延遲、peak allocated memory 與有限 MAC 估算。
- `residual-diagnostic-v1.json`：全部 683 張 BBAT val 的 P3 相對殘差。
- `preflight-v1.json`、`final-audit-v1.json`：CPU 前置與交付稽核。
- `queue-v1/`：事件與原始 log；`datasets/` 是可重建 runtime View，不是新資料版本。

全部保留，沒有清理或發布。此處候選權重沒有新增 Pose 專項 optimizer，不可當作新的 full-resume；原 Attention E2 的完整續訓檔仍在原實驗且維持暫停。
