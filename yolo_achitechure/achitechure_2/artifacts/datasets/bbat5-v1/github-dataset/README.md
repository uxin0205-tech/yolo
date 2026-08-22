# BBAT5 v1 GitHub 完整資料 Snapshot

此目錄是使用者於 2026-08-22 明確核准上傳的可攜 snapshot。它只改變儲存形式，不改變
`bbat5-v1` 的 grouped split、四筆座標修補或 2.0.0 資料 lineage，也不會啟動 Pose 訓練。

## 內容

- 唯一影像：6,647 張。
- Pose／Detect 各有 formal train 5,964、val 683。
- Pose／Detect 各自保存 labels；Detect labels 仍是 Pose 每列前五欄。
- `data.yaml` 是 formal split；`data-search.yaml` 的搜尋範圍只在 formal train 內。
- 所有 split 路徑均為相對路徑；影像是一般檔案，不含指向 `/home/uxin/...` 的 symlink。
- 不含 test split、weights、checkpoint、cache、runs、`.pt`、`.pth`、`.onnx` 或 engine。

## 使用

Detection 診斷 view：

    data=artifacts/datasets/bbat5-v1/github-dataset/detect/data.yaml

Pose view（正式執行仍需使用者另外決定）：

    data=artifacts/datasets/bbat5-v1/github-dataset/pose/data.yaml

搜尋設定分別使用同一目錄下的 `data-search.yaml`。完整來源、split、patch 與 COCO 排除證據在
上一層 `../manifests/`；本 snapshot 的檔案樹雜湊與數量在 `publication-manifest.json`。
