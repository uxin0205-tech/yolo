# 實驗產物

這裡是執行證據，不是 Python 套件內的 `masf_yolo/artifacts/`。

- `static-phase1/`：本次十個模型的正式訓練、評估、profile、選模與稽核。
- `official_baseline_cpu/`：舊版官方 detect baseline CPU 證據。
- `official_pose_cpu/`：`../original/pose/weight` 的 CPU 證據。
- `official_weight_metrics/`：來源權重逐一檢查指標。

正式權重與量測證據必須保留。只允許清理 `masf_yolo.cleanup` 白名單中的中間物，紀錄寫入 `static-phase1/cleanup_manifest.json`。
