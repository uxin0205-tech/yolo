# 交付稽核紀錄

日期：2026-08-24（Asia/Taipei）

## 完整性

- 11 個比較候選：A0 1 個、Full35 5 個、Partial75 5 個。
- `artifacts/checkpoints/*.pt` 18 個相關 Bit-True 檔案已全數映射；A2 的多個 queue 轉檔以 state-dict SHA256 證明為同一模型。
- 11 個 Bit-True checkpoint 全部實際載入，架構、兩個 attention backend、檔案 SHA256 與 state-dict SHA256 均通過。
- 10 個 Float best checkpoint 全部實際載入，架構、兩個 `piecewise_linear` backend 與檔案 SHA256 均通過。
- `code/achitechure_1/` 與 `code/yolo_attention/` 共 55 個 `.py`，與各自來源 snapshot byte-for-byte 相同。
- 8 次 B/C 訓練均保留 manifest、args、results.csv、results.png、F1／PR／P／R curves、confusion matrix、labels 與 telemetry，共 106 個訓練證據檔。
- 集中式權重索引有 21 筆：11 個 Bit-True evaluation candidates 加 10 個 Float best。

## Accuracy 與訓練證據

- COCO2017 val：11/11 候選具有 Ultralytics internal metrics。
- Canonical COCO API：11/11 候選均由各自 predictions 計算 overall、sports ball、baseball bat 的 AP50-95／AP50／AP75／AP_S／AP_M／AP_L／AR100。
- BBT5：11/11 候選均在 `/home/uxin0/yolo/original/pose/detect_dataset/coco80/data.yaml` 驗證；不是直接使用 pose dataset。
- 測試矩陣為 22/22 model-dataset 組合，所有 JSON 數值 finite，raw metrics 的 checkpoint SHA 與 registry 一致。
- 完整資料 Phase C queue 為 `completed`，沒有 pending job；Full35／Partial75 均 rollback 到 A2。
- 三張跨 run 圖已實際開啟檢查：訓練 mAP、BBT5 AP、RAM／VRAM telemetry 均可讀且內容非空。

## 自動檢查

```text
pytest: 49 passed in 6.68s
inspect_models.py: 11/11 Bit-True PASS；10/10 Float PASS
compileall: PASS
report audit: 11 models；18 source checkpoints；22 dataset validations；8 training runs；21 weight records
code snapshot: 55 Python files byte-for-byte PASS
```

`scripts/verify_bundle.py --write` 會先交叉檢查候選、metrics、權重 SHA、queue、訓練證據、圖檔與 Markdown link，再建立 `MANIFEST.json`；無參數模式用來反向驗證整包。

## 邊界

- AP 排名使用 Bit-True PWL；Float 只供接續訓練或重新 materialize。
- COCO internal 與 canonical COCO API 是不同 evaluator，報告分表呈現。
- BBT5 valid 有 93/567 張影像的 COCO ID 出現在 COCO train2017，只適合相對比較。
- 所有候選只有 seed 0，不能把極小差距解讀成統計顯著。
- 本次沒有刪除或搬移既有 runtime artifacts，也沒有建立 commit 或 push。
- 唯讀清理建議見 `CLEANUP_PROPOSAL.md`；未執行任何項目。
