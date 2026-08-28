# BBAT5 v1 衍生資料

此目錄是 architecture_2 的不可覆寫資料版本。建立資料不會啟動 Pose 訓練；是否執行 Pose 仍由使用者另外決定。

## 我們做了什麼

- 唯讀 Pose 來源：`/home/uxin/yolo/original/pose/dataset`。
- 唯讀 Detect audit 來源：`/home/uxin/yolo/original/pose/detect_dataset`。
- Pose labels 是唯一權威；Detect labels 由修補後每列前五欄產生。
- 依 `.rf.` 前 prefix、seed 0 做 grouped 90%/10% formal train/val。
- search split 僅從 formal train 內再分組，formal val 完全不參與搜尋。
- 將 4 個負座標在衍生 label clamp 成 0；原始檔不變。
- COCO train 重疊群組數：331；排除狀態：`passed`。
- Pose/Detect 共用相同 assignment；影像為 symlink；沒有建立 test split。

## 目錄

- `pose/`：正式 Pose view 與 search 清單。
- `detect/`：ball/bat 2-class 診斷 view，不取代 COCO80 Detect。
- `configs/`：可直接交給 Ultralytics 的 formal/search dataset YAML。
- `manifests/`：來源稽核、split、patch、COCO 排除與重建 lineage。
- `github-dataset/`：使用者核准上傳的完整可攜 snapshot；一般影像檔、兩種 labels、relative splits
  與 publication manifest，不含 symlink、test、cache 或 weights。

## 重建

```bash
python -m achitechure_2.cli prepare-pose-data --execute
```

此 v1 不可覆寫；規則或資料有任何變更時，請建立 `bbat5-v2`。

完整 snapshot 的建立與驗證命令：

```bash
python -m achitechure_2 export-github-dataset --execute
python -m achitechure_2 validate-github-dataset
```
