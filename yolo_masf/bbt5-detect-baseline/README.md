# BBT5 Detect Baseline

本目錄提供 MASF-YOLO 使用的 BBT5 兩類別 detection view。類別順序以 `data.yaml` 為準：Ball 與 Bat。正式 pipeline 只讀取這裡的資料與初始化權重，不會修改來源影像。

## 結構

| 路徑 | 用途 |
|---|---|
| `prepare_dataset.py` | 從 pose dataset 建立 detection view；影像用連結，11 欄 pose label 取前 5 欄 |
| `data.yaml` | Ultralytics 資料集入口 |
| `dataset/` | 正式使用的 images/labels 與 split view；不提交 Git |
| `weights/yolo11m_bat_detect_init.pt` | pose-derived YOLO11m detect initializer |
| `transfer_pose_to_detect.py` | 建 detect model、轉移相容權重、丟棄 pose-only keypoint branch |

初始化權重已在 BBT5 pose 訓練中看過這批資料。因此 B0 直接評估與 B1/MFAM 由它初始化都必須標示 data-exposed；不可把結果當成乾淨的 unseen-data baseline。

## 本次實際做法

正式 pipeline 使用固定 manifest 產生 leakage-aware group split，而不是任意重切資料。稽核輸出在 `../artifacts/static-phase1/dataset/`；共 2,578 unique frames，train/val/test 為 1,987/300/291。B1 由指定 initializer 開始，先凍結 backbone 0–10 訓練 10 epochs，再全解凍 90 epochs；其後變體由 repository-owned canonical checkpoint 轉移。

如需重建 detect view：

```bash
../../.venv/bin/python prepare_dataset.py
```

如需重新產生 pose-to-detect initializer：

```bash
../../.venv/bin/python transfer_pose_to_detect.py
```

不要覆寫 pose 來源權重或既有 initializer。正式結果不放在本目錄，請到 `../artifacts/static-phase1/`；總結見 `../EXPERIMENT_RESULTS_ZH.md`。
