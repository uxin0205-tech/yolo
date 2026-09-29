# 實驗產物

> **Cleanup-v1 發行範圍：** 本分支保留程式、設定、報告、統計圖表與 manifests；checkpoint、資料集影像／標註、逐筆預測、batch 預覽和冗長執行日誌已從此發行快照排除。下文的歷史權重路徑、數量、checksums 與實驗結果仍保留作研究紀錄；訓練、推論、完整交付驗證及資料重建需要另行提供原始資產，不代表 clone 後即可直接重跑。 詳見[清理範圍與資產需求](../../docs/cleanup-v1/README.md)。


這裡是執行證據，不是 Python 套件內的 `masf_yolo/artifacts/`。

- `static-phase1/`：本次十個模型的正式訓練、評估、profile、選模與稽核。
- `official_baseline_cpu/`：舊版官方 detect baseline CPU 證據。
- `official_pose_cpu/`：`../original/pose/weight` 的 CPU 證據。
- `official_weight_metrics/`：來源權重逐一檢查指標。

正式權重與量測證據必須保留。只允許清理 `masf_yolo.cleanup` 白名單中的中間物，紀錄寫入 `static-phase1/cleanup_manifest.json`。
