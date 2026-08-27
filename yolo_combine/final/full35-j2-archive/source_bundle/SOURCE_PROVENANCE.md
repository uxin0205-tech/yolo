# 原始碼與資料來源

- `code/achitechure_1/`：從本工作區 `src/achitechure_1/` 逐一複製 `.py`。內容對應交付 commit `b9df408da1007c23f8c970d498b72d198e6f5e37`；`ball_bat_evaluation.py` 在目前 main worktree 是 untracked，但檔案 SHA256 與該 commit 完全相同。
- `code/yolo_attention/`：從 `/home/uxin0/yolo/yolo_attention_final/src/yolo_attention/` 複製 `.py`；來源 repository revision 為 `ca36be5234f7ece1a762100dd8a21d8c0c0c1884`，且 source tree 無修改。
- `configs/datasets/coco2017.yaml`：本次正式 COCO2017 validation 契約。
- `configs/datasets/bbt5-coco80.yaml`：從 `/home/uxin0/yolo/original/pose/detect_dataset/coco80/data.yaml` 複製。實際驗證使用的是這個 detection view，不是 pose `dataset/data.yaml`。
- `reports/raw/bbt5-dataset-manifest.json`：保存 pose 來源到 detect dataset 的類別映射、label tree SHA256 與 COCO train ID overlap 警告。

所有 checkpoint、程式碼、設定與報表檔案的最終雜湊由 `MANIFEST.json` 統一列出。
