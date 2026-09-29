# BBAT5 v1 衍生資料

> **Cleanup-v1 發行範圍：** 本分支保留程式、設定、報告、統計圖表與 manifests；checkpoint、資料集影像／標註、逐筆預測、batch 預覽和冗長執行日誌已從此發行快照排除。下文的歷史權重路徑、數量、checksums 與實驗結果仍保留作研究紀錄；訓練、推論、完整交付驗證及資料重建需要另行提供原始資產，不代表 clone 後即可直接重跑。 詳見[清理範圍與資產需求](../../../../../docs/cleanup-v1/README.md)。


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
