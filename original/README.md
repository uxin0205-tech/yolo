# original 資料發行範圍

> **Cleanup-v1 發行範圍：** 本分支保留程式、設定、報告、統計圖表與 manifests；checkpoint、資料集影像／標註、逐筆預測、batch 預覽和冗長執行日誌已從此發行快照排除。下文的歷史權重路徑、數量、checksums 與實驗結果仍保留作研究紀錄；訓練、推論、完整交付驗證及資料重建需要另行提供原始資產，不代表 clone 後即可直接重跑。 詳見[清理範圍與資產需求](../docs/cleanup-v1/README.md)。


此目錄在 GitHub 保存可追溯的原始／歷史資料內容，不是所有新訓練都可直接使用的入口。
BBAT5 新實驗仍以 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml` 為唯一 registry。
`original/pose/` 是目前 Git tree 與本機共用的唯一 BBAT5 資料資產庫；各子專案不得再提交
`artifacts/datasets/` 副本。


## 會上傳

- `pose/dataset/`：唯讀原始 Pose 來源。
- `pose/detect_dataset/`：唯讀歷史 ball/bat／COCO80 Detect View。
- `pose/derived/bbat5-v1/`：正式 Pose／二類 Detect Task Views、YAML 與 manifests。
- README、license、data YAML、split lists、影像與 labels。

## 不會上傳

- `weight/`、`pose/weight/`、任何 `.pt`／`.pth`／checkpoint。
- `*.cache`、run、profile、deployment artifact。
- `pose/detect_dataset.zip`：726,351,400 bytes，超過 GitHub 一般 Git 100 MB 限制，且內容已由
  完整 `pose/detect_dataset/` 目錄提供。

GitHub 發行樹中的 canonical image symlink 使用 repository-relative target，clone 到其他位置後
仍能解析；本機原始資料及權重不會因發布而被刪除或改寫。資料角色與任務入口見
[`pose/README.md`](pose/README.md)。

2026-08-22 的 architecture_2 可攜快照已由 0823 統一決策取代，只留在 Git 歷史；現行資料、
YAML、split、patch 與 lineage 均以本目錄內容為準。
