---
status: accepted
---

# 全專案統一使用 BBAT5 v1

所有新的 ball/bat Detect、Pose 與 Detect–Pose 融合實驗統一使用不可變的 `bbat5-v1`；原始 `dataset/`、`detect_dataset/` 只保留作來源稽核與歷史重建。各專案禁止建立 `artifacts/datasets/` 副本；需要隔離 Ultralytics cache 時，只能在 `artifacts/cache-views/` 建立 symlink-only View。這個選擇取代直接沿用 Roboflow basic split，因為 basic split 有 10 個同源群組跨 train/valid，而 `bbat5-v1` 使用共同 grouped assignment、零 leakage、固定四筆修補與完整 hash lineage。既有 checkpoint 與結果仍保留其舊 split 血緣，不回溯改寫。
