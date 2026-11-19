# BBAT5 v1 YOLO26m Pose

目前僅保留逐階段降低學習率訓練的 P3 模型，其餘實驗權重與大型輸出已依使用者要求刪除。

## 保留模型

目錄：[artifacts/preserved/stagewise_lr_p3_640](artifacts/preserved/stagewise_lr_p3_640/README.md)。

- [best.pt](artifacts/preserved/stagewise_lr_p3_640/weights/best.pt)：第 163 輪最佳模型。
- [last.pt](artifacts/preserved/stagewise_lr_p3_640/weights/last.pt)：第 200 輪模型。
- [last-resumable.pt](artifacts/preserved/stagewise_lr_p3_640/weights/last-resumable.pt)：保留 optimizer 的最後 checkpoint。
- [完整設定](artifacts/preserved/stagewise_lr_p3_640/args.yaml)、[逐輪指標](artifacts/preserved/stagewise_lr_p3_640/results.csv)、[正式評估](artifacts/preserved/stagewise_lr_p3_640/evaluation.json)、[誤差診斷](artifacts/preserved/stagewise_lr_p3_640/errors/report.md)。

formal val 683 張、imgsz 640：Box mAP50–95 **62.8808%**，Pose mAP50–95 **92.0616%**。單一 seed，沒有獨立 test split。

## 資料集規範

唯一正式資料來源為 `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml`；registry 為 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。沿用 train 5,964 張、val 683 張，不重新切分、抽樣或更動標註。`configs/data.runtime.yaml` 與 `artifacts/datasets/` 僅提供可重建 runtime View。資料集與稽核紀錄未刪除。

## 歷史與重現

程式碼、設定、精簡報告與 `artifacts/historical-run-metadata/` 的逐輪指標保留。歷史設定內的舊 checkpoint 路徑可能已刪除，不能直接啟動舊佇列；官方初始權重需重新下載。保留目錄 manifest 的 source 欄位表示歷史來源，file 欄位是仍有效的相對路徑。

[工作紀錄索引](docs/worklogs/README.md) · [本次清理紀錄](docs/worklogs/2026-09-18-cleanup.md) · [精確刪除清單](docs/worklogs/2026-09-18-cleanup-manifest.json)
