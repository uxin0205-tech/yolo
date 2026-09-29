# BBAT5 v1 正式共用資料版本（Pose + Detect）

> **Cleanup-v1 發行範圍：** 本分支保留程式、設定、報告、統計圖表與 manifests；checkpoint、資料集影像／標註、逐筆預測、batch 預覽和冗長執行日誌已從此發行快照排除。下文的歷史權重路徑、數量、checksums 與實驗結果仍保留作研究紀錄；訓練、推論、完整交付驗證及資料重建需要另行提供原始資產，不代表 clone 後即可直接重跑。 詳見[清理範圍與資產需求](../../../../docs/cleanup-v1/README.md)。


這不是只供 Pose 或只供 Detect 的資料夾，而是全專案共同使用、不可覆寫的版本容器。
兩個 Task View 共用相同 6,647 張影像身分與 grouped split，但 labels 格式不同。此根目錄
不能直接交給 `YOLO(...).train(data=...)`，必須依任務選擇 `configs/` 內的 YAML。

## 要使用哪一個 YAML

| 工作 | 正式 YAML | 標註語意 |
| --- | --- | --- |
| 正式 ball/bat Pose 訓練／驗證 | `configs/pose.yaml` | bbox + 2 個 keypoints |
| 正式 ball/bat 二類 Detect 診斷 | `configs/detect.yaml` | bbox only，classes 是 ball／bat |
| Pose 超參數搜尋 | `configs/pose-search.yaml` | 只在 formal train 內搜尋 |
| 二類 Detect 超參數搜尋 | `configs/detect-search.yaml` | 只在 formal train 內搜尋 |

COCO80／person Detect 不使用本目錄的 `detect.yaml`，而使用
`/home/uxin/yolo/coco2017.yaml`。融合程式應讀取全域
`/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`，再依 route 取得 BBAT5 Task View；
若模型保留 COCO80 Detect head，仍另外讀取 COCO 設定。

建立或驗證資料不會啟動 Pose 訓練；是否執行 Pose 仍由使用者另外決定。

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

## 重建

```bash
python -m achitechure_2.cli prepare-pose-data --execute
```

此 v1 不可覆寫；規則或資料有任何變更時，請建立 `bbat5-v2`。
