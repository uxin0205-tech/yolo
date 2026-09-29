# 2026-08-22：統一全專案 BBAT5 資料集

> 0823 後續決策：唯一資料資產庫進一步固定為 `/home/uxin/yolo/original/pose/`；
> `bbat5-v1-runtime` 現改名為 `artifacts/cache-views/bbat5-v1`，只保存 symlink/cache，
> `artifacts/datasets/` 禁止再作現行入口。下文舊名稱保留作當時紀錄。

## 任務與範圍

盤點 `/home/uxin/yolo/original/pose/` 與
`/home/uxin/yolo/yolo_combine/artifacts/datasets/`，選出唯一正式資料版本，整理
`yolo_combine` 的資料入口，並確保未來 Detect、Pose 與融合實驗使用相同樣本與 split。
本輪只執行 CPU 資料與程式驗證，沒有啟動 GPU 或正式訓練。

## 盤點結果

- 原始 Pose `dataset/`：6,080 train／567 valid，共 6,647 images；有 10 個 `.rf.`
  同源群組跨 split。
- 歷史 Detect `detect_dataset/`：由 Pose 前五欄衍生，沿用相同 basic split；
  `coco80/` 只是舊 COCO80 class-ID 評估 View。
- `yolo_combine/artifacts/datasets/bbt5_pose_basic/`：從原始 basic split 建立的 runtime
  View，包含 6,647 image symlinks、6,647 labels、2 個 cache 與 2 個 metadata。
- `derived/bbat5-v1/`：6,647 images；grouped formal train/val 5,964/683；零
  source-group leakage；Detect/Pose 共用 assignment；四筆負座標修補有 manifest。

逐檔比較 legacy View 與 `bbat5-v1`：missing images 0、missing labels 0、label content
mismatch 0、broken symlink 0。兩者差異是 split assignment 與 lineage，不是遺失資料。

## 變更內容與原因

1. 選定 `/home/uxin/yolo/original/pose/derived/bbat5-v1/` 為所有新實驗唯一正式資料。
2. 新增 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`，鎖定 dataset ID、正式與
   search YAML、split、樣本統計、五份 manifests 與 SHA256。
3. 新增根層 `CONTEXT.md` 與 ADR 0001，區分原始來源、正式資料、Task View、runtime
   View、GitHub snapshot 與 legacy basic split。
4. `original/pose/dataset/`、`detect_dataset/` 與 `weight/` 保持不變，新增
   `original/pose/README.md` 說明角色；raw 只供稽核、重建與歷史解釋。
5. `yolo_combine` variant schema 升為 2，Full35/Partial75 都引用同一 registry 與
   canonical Pose/Detect YAML。
6. `yolo_combine.data.CanonicalBBAT5` 統一驗證 YAML／manifest hash、實際 split
   assignment、Pose/Detect labels、broken links、負座標與 leakage；任何漂移 fail closed。
7. 新 runtime View 改名 `bbat5-v1-runtime`，image 與 label 都使用 symlink，只有 cache
   與 runtime manifest 留在 workspace。
8. 在保存 legacy manifest/data YAML SHA、10 個 overlap groups 與逐檔核對結果後，刪除
   `/home/uxin/yolo/yolo_combine/artifacts/datasets/bbt5_pose_basic/`。刪除內容約 54 MB，
   只有可重建 View/cache，沒有 checkpoint 或權重；可由原始來源與歷史程式重建。
9. 更新根層、`architecture_2` 與 `yolo_combine` README/RUNBOOK/AGENTS，禁止新 run
   直接使用 raw/basic split。歷史 checkpoint 與 metrics 不回溯改寫。

## 資料改動聲明

- 正式 `bbat5-v1`：影像、labels、split 與 manifests 全部未修改。
- 原始 `dataset/`、`detect_dataset/`：未修改、未移動、未刪除。
- 沒有新增 test split，沒有重新隨機切分，沒有啟動 Pose 訓練。
- 唯一刪除的是 `yolo_combine` 可重建的 legacy runtime View；其血緣證據保存在
  `yolo_combine/docs/data/2026-08-22-bbat5-unification.md`。

## 驗證方式與結果

- architecture_2 `validate-pose-data`：`valid=true`、`source_untouched=true`、
  5,964/683、4 patches、無 test。
- canonical paired audit：6,647 Pose／Detect images、0 derivation mismatch、0 broken
  links、0 group overlap、0 negative keypoint rows。
- legacy vs canonical 逐檔核對：6,647/6,647 全部涵蓋，label mismatch 0。
- `yolo_combine` focused tests：14 passed。
- `yolo_combine` 完整 CPU suite：38 passed、3 CUDA-only skipped。
- Detect 與 Pose 完整 runtime dry-run：每個 task 各有 6,647 image links 與 6,647
  label links，合計 26,588 symlinks、4 個 regular metadata files、0 broken links、
  0 regular image/label copies；Ultralytics 8.4.90 正確解析 train/val、names 與
  `kpt_shape=[2,3]`。
- Full35／Partial75 workspace audit：都解析到同一個 `bbat5-v1` registry，
  `missing_paths=[]`、`hash_mismatches=[]`。
- 本輪修改的 10 個 Python 檔案：Ruff lint 與 format check 全部通過；compileall 通過。
- 沒有使用 GPU，沒有執行正式 Detect/Pose/fusion 訓練。

## 困難與解法

- 困難：兩邊目錄名稱都像資料集，但 `bbt5_pose_basic` 實際是有 leakage 的 runtime View。
  解法：讀取 manifest、程式與 symlink，並逐檔比對 canonical 內容後才移除。
- 困難：原始 Detect README 宣稱影像是 symlink，但現場檔案是獨立 regular files。
  解法：保留整個 raw/historical 目錄不動，只禁止新 run 使用，避免破壞歷史報告。
- 困難：本機根層 `main` 原本已 dirty 且與 GitHub 分岔。
  解法：沒有 reset、pull 或清理其他 dirty 內容；本次只修改明確列出的資料政策、
  `architecture_2` 文件與 `yolo_combine` 程式／文件。

## 未解事項或風險

- 舊 Full35 P1 與其他 basic-split metrics 只能保留為 provisional；正式比較需以
  `bbat5-v1` 從頭重跑，但本輪未獲授權啟動 GPU 訓練。
- `original/pose/detect_dataset/` 內有歷史影像實體副本，約占額外空間；為保護舊結果，
  本輪沒有刪除或改成 symlink。
- 本機 `main` 仍 ahead 1／behind remote，後續 commit/push 應繼續使用乾淨 worktree，
  不可直接 reset 或 pull 覆蓋本機資料。
