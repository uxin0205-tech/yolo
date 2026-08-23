# BBAT5／BBT5 棒球資料集使用規範

本文件是 `/home/uxin/yolo` 下所有子專案共用的資料集契約。使用者所稱的 BBAT5 與現有報告中的 BBT5 均指以下固定棒球資料集。

## 唯一正式版本

所有 2026-08-22 之後開始的新實驗固定使用 `bbat5-v1`。全域 machine-readable
registry 是 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。

| 任務類型 | 唯一正式 YAML | 資料 View |
| --- | --- | --- |
| ball/bat 物件偵測（detect） | `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml` | `derived/bbat5-v1/detect/` |
| ball/bat 姿態估計（pose） | `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml` | `derived/bbat5-v1/pose/` |

凡任務提到 BBAT5、BBT5、ball／bat 或棒球專用資料集，若未另有明確指示，依任務
類型使用上表 YAML。Pose 與 Detect 是同一版本的兩種 Task View，不是兩套可各自切分的
資料集。

重要邊界：上表 `detect.yaml` 是 `nc=2` 的 ball/bat 診斷資料，不是 COCO80 Detect
主線。COCO80／person Detect 固定使用 `/home/uxin/yolo/coco2017.yaml`；不得把 80 類 head
指向 BBAT5 二類 YAML，也不得因融合需要而把兩者稱為同一資料集。

## 唯讀來源與歷史資料

| 目錄 | 角色 | 新實驗可直接使用 |
| --- | --- | --- |
| `/home/uxin/yolo/original/pose/dataset/` | 原始 Pose 來源與 legacy basic split | 否 |
| `/home/uxin/yolo/original/pose/detect_dataset/` | 原始 Pose 的歷史 Detect/COCO80 衍生 View | 否 |
| 各專案 `artifacts/datasets/` | 禁止使用；避免被誤認為第二份資料集 | 否 |
| 各專案 `artifacts/cache-views/` | symlink-only、可重建的 cache 隔離 View | 否 |

原始來源保留是為了重建、hash audit 與解釋既有 checkpoint，不代表它仍是正式訓練入口。

## 控制變因規則

1. 直接沿用 `bbat5-v1` 的 formal train／val assignment；不存在 test split。
2. 不得自行隨機切分、重新分層、抽樣、合併、補造 test split，或使用不同 seed 產生新 split。
3. 專案 runtime View 只能放在 `artifacts/cache-views/`，以 symlink 與本地 cache 重現相同 assignment；不得擁有另一份 labels、split、YAML 或被稱為新來源，也不得以其他 Roboflow、COCO 或自建資料替換。
4. 不得自行新增、刪除、重標、過濾或轉換影像與標註。任何例外都需要使用者明確授權。
5. 每次訓練與評估報告必須記錄 registry、dataset ID、正式 YAML、任務類型與 split，並明記是否維持不變。
6. 若設定檔只有路徑引用錯誤，可做最小限度的路徑修正；修正不得改變樣本、標註或 split，且必須寫入工作紀錄。

## Legacy basic split 的已知狀態

- 原始 Pose basic split 是 6,080 train／567 valid，存在 10 個 `.rf.` 同源群組跨 split。
- 原始 Detect labels 與 Pose 前五欄逐筆相符，但它沿用同一個有 leakage 的 basic split。
- `detect_dataset/data.yaml` 與歷史 manifest 仍含舊機器路徑 `/home/uxin0/...`；這些檔案只保留作歷史證據，不再修成新訓練入口。
- `detect_dataset/coco80/` 是供既有 COCO80 head 解讀 ball/bat class ID 的歷史評估 View，不是另一個正式 BBAT5 split。
- `yolo_combine/artifacts/datasets/bbt5_pose_basic/` 是從 legacy basic split 建出的歷史 runtime View；新 run 禁止使用。

## 正式不可變版本

2026-08-22 使用者明確核准 architecture_2 建立獨立衍生目錄、grouped split 與四筆座標修補。
此版本現為全專案唯一正式 BBAT5 資料，不代表可再建立另一套 split。

正式版本固定為：

- 實際資料：/home/uxin/yolo/original/pose/derived/bbat5-v1/
- 建立工具：/home/uxin/yolo/yolo_achitechure/achitechure_2/src/achitechure_2/pose_data.py
- 操作說明：/home/uxin/yolo/yolo_achitechure/achitechure_2/README.md
- YAML／manifests：/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/ 與 manifests/

bbat5-v1 的固定契約：

1. 原始 Pose／Detect 目錄維持唯讀；Pose labels 是 ball／bat 的權威來源，Detect view 由修補後每列前五欄衍生，原始 Detect labels 只作一對一 audit。
2. 依檔名 .rf. 前 prefix 分組，以 seed 0 建立 90%／10% formal train／val；同源 augmentation 不得跨 split。
3. search train／val 只在 formal train 內再次分組；formal val 不參與搜尋。
4. 共有 6,647 images、2,578 source groups；formal train／val 是 5,964／683，search train／val 是 5,364／600。
5. 與 COCO train 重疊的 331 groups 固定留在 train，formal／search val overlap 都是 0。
6. 四個極小負座標只在衍生 Pose labels clamp 為 0，逐筆保存在 patch manifest；原始 label tree hash 維持不變。
7. Pose／Detect 共用相同 assignment；不建立不存在的 test split。
8. v1 不可覆寫。資料或規則需要改變時，必須由使用者核准並建立新版本，例如 bbat5-v2。

其他子專案必須直接引用 bbat5-v1 registry 與正式 YAML。若 Ultralytics cache 必須隔離，
可從正式 Task View 建立 symlink-only runtime View；不得從原始目錄再切一次或再次修補 labels。
重建命令只會建立資料，不會啟動 Pose 訓練：

    cd /home/uxin/yolo/yolo_achitechure/achitechure_2
    export PYTHONPATH="$PWD/src"
    /home/uxin/yolo/.venv/bin/python -m achitechure_2 prepare-pose-data --execute

## GitHub 發布與唯一位置

2026-08-23 使用者明確核准發布 `original/` 並將它定為唯一 BBAT5 資料位置：

- 包含 `original/pose/dataset/`、`original/pose/detect_dataset/`、
  `original/pose/derived/bbat5-v1/` 及其 README／YAML／manifests。
- raw Pose、歷史 Detect 與 legacy basic split 即使出現在 GitHub，仍只供稽核與重建；
  所有新訓練仍以 `bbat5-v1` registry 為唯一入口。
- canonical image links 在 Git 發行樹使用 repository-relative symlink；連到相同 raw image，
  不建立新 split 或新資料版本。
- 永久排除兩個 `original/**/weight/`、所有 `.pt`／checkpoint、cache、run 與 deployment artifact；
  Git 歷史曾追蹤的兩個 original 權重從 0823 tree 移除，本機檔案不動。
- `original/pose/detect_dataset.zip` 是解壓目錄的重複 archive 且超過 GitHub 100 MB 單檔限制；
  發布完整解壓目錄但不提交此 zip。
- 2026-08-22 的 architecture_2 portable snapshot 只保留於 Git 歷史；目前 tree 已移除。
  `artifacts/datasets/` 不得再保存資料、YAML、labels、split 或 manifests 的正式副本。
- architecture_2 的正式 snapshot 匯出／驗證 CLI 已停用；資料驗證只使用 `validate-pose-data`。
  是否跑 Pose 仍是獨立的使用者決定。

## 可追溯性

所有相關修改、實驗、驗證、困難與解法，依[工作紀錄規範](../worklogs/README.md)以中文留存。全域入口見[根層 README](../../README.md)。
