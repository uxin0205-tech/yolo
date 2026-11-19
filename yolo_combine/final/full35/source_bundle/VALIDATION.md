# 交付稽核紀錄

日期：2026-08-22（Asia/Taipei）

## 完整性

- 9 個比較候選：A0 1 個、Full35 4 個、Partial75 4 個。
- 16 個 `artifacts/checkpoints/*.pt` 全部映射到 8 個不同的 Full35／Partial75 state-dict；A2 的多個 queue 轉檔已證實是同一模型。
- 9 個 Bit-True checkpoint 全部實際載入，架構、雙 attention backend、檔案 SHA256 與 state-dict SHA256 均通過。
- 8 個 Float checkpoint 全部實際載入，架構、雙 `piecewise_linear` backend 與檔案 SHA256 均通過。
- `code/achitechure_1/` 與 `code/yolo_attention/` 的 55 個 `.py` 和來源 snapshot byte-for-byte 相同。

## Accuracy 證據

- COCO2017 val：9/9 候選具有 Ultralytics internal metrics 與 `predictions.json`。
- Canonical COCO API：9/9 候選均由各自 predictions 重算 overall、sports ball、baseball bat 的 AP50-95／AP50／AP75／AP_S／AP_M／AP_L／AR100。
- BBT5：9/9 候選均在 `/home/uxin0/yolo/original/pose/detect_dataset/coco80/data.yaml` 驗證；不是直接使用 pose dataset。
- 後補實跑：Full35 B-100% BBT5、Partial75 B-100% BBT5、A0 RTX 5060 Ti COCO2017。
- 所有 JSON AP 欄位為 finite；整體測試矩陣為 18/18 model-dataset 組合。

## 自動檢查

```text
pytest: 48 passed in 4.51s
inspect_models.py: 9/9 Bit-True PASS；8/8 Float PASS
compileall: PASS
report audit: 9 models；16 source checkpoints；18 dataset validations
```

最終 `MANIFEST.json` 由 `scripts/verify_bundle.py --write` 建立，再以無參數模式重新驗證。

## 邊界

- AP 排名使用 Bit-True PWL；Float 只供接續訓練或重新 materialize。
- COCO internal 與 canonical COCO API 是不同 evaluator，報告分表呈現。
- BBT5 valid 有 93/567 張影像的 COCO ID 出現在 COCO train2017，只適合相對比較。
- `fraction=1.0` Phase C 嘗試未產生 checkpoint；它不是「已保存候選」，因此沒有虛構 AP 或空白列。
- 本次沒有刪除或整理既有 runtime artifacts，也沒有建立 commit／push。
