# 原始碼與資料來源

- `code/achitechure_1/`：2026-08-24 從本工作區 `src/achitechure_1/` 複製 18 個 `.py`，與當前 worktree byte-for-byte 相同。基底 revision 是 `ca36be5234f7ece1a762100dd8a21d8c0c0c1884`；正式 queue、資源保護與 ball／bat 驗證含該 revision 之後尚未 commit 的工作內容，因此不能只靠 revision 重建，應以此 snapshot 與 `MANIFEST.json` 為準。
- `code/yolo_attention/`：從 `/home/uxin0/yolo/yolo_attention_final/src/yolo_attention/` 複製 37 個 `.py`；來源 revision 是 `ca36be5234f7ece1a762100dd8a21d8c0c0c1884`。來源的 `README.md` 不是 Python runtime dependency，未納入 code snapshot。
- `configs/datasets/coco2017.yaml`：正式 COCO2017 validation 契約。
- `configs/datasets/bbt5-coco80.yaml`：從 `/home/uxin0/yolo/original/pose/detect_dataset/coco80/data.yaml` 複製。實際驗證使用 detection view，不是 pose `dataset/data.yaml`。
- `reports/raw/bbt5-dataset-manifest.json`：保存 pose 來源到 detect dataset 的類別映射、label tree SHA256 與 COCO train ID overlap 警告。
- `reports/raw/<model-id>/`：每個候選實際 Bit-True checkpoint 的 COCO internal、canonical COCO API 與 BBT5 internal metrics。
- `reports/training/<model-id>/`：八次正式 B/C run 的 manifest、args、results.csv、Ultralytics 訓練圖、曲線、confusion matrix 與原始 RAM／VRAM telemetry。
- `reports/raw/profiles/`：2026-08-24 在 RTX 5060 Ti 實測的 A0／Full35 A2／Partial75 A2 FP16 latency，以及完整資料 Phase C 的 AMP real-loss capacity probes。
- `reports/raw/queues/fraction10-phase-c-state.json`：完整資料 Phase C 最終 `completed` queue state。

所有 checkpoint、程式碼、設定與報表檔案的最終雜湊由 `MANIFEST.json` 統一列出；正式權重 metadata 另見 `weights/index.json`。
