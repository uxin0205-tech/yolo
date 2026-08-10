# 設定檔

| 檔案 | 用途 |
|---|---|
| `static-phase1.yaml` | Phase 1 唯一正式 pipeline 設定 |
| `b0-reference.yaml` | B0 來源權重參考評估 |
| `m7-priority.yaml` | M7 優先驗證設定 |
| `models/yolo11m-p2-slots.yaml` | YOLO11m P2 架構與 P2/P3 slots |

資料來源是 `../bbt5-detect-baseline/dataset`，初始化是 `../bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`。SHA-256 pin 防止誤換權重或基礎 YAML。修改正式設定會改變 pipeline identity；不要覆寫既有 `artifacts/static-phase1/`。
