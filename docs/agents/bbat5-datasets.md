# BBAT5／BBT5 棒球資料集使用規範

本文件是 `/home/uxin/yolo` 下所有子專案共用的資料集契約。使用者所稱的 BBAT5 與現有報告中的 BBT5 均指以下固定棒球資料集。

## 固定來源

| 任務類型 | 固定目錄 | 既有設定 |
| --- | --- | --- |
| 物件偵測（detect） | `/home/uxin/yolo/original/pose/detect_dataset/` | `data.yaml` |
| 姿態估計（pose） | `/home/uxin/yolo/original/pose/dataset/` | `data.yaml` |

凡任務提到 BBAT5、BBT5、ball／bat 或棒球專用資料集，若未另有明確指示，依任務類型使用上表來源。

## 控制變因規則

1. 直接沿用固定目錄內既有的 train／valid／test 定義與樣本歸屬。
2. 不得自行隨機切分、重新分層、抽樣、合併、補造 test split，或使用不同 seed 產生新 split。
3. 不得把資料集複製到實驗目錄後當成新來源，也不得以其他 Roboflow、COCO 或自建資料替換。
4. 不得自行新增、刪除、重標、過濾或轉換影像與標註。任何例外都需要使用者明確授權。
5. 每次訓練與評估報告必須記錄實際資料集絕對路徑、任務類型與使用的既有 split，並明記是否維持不變。
6. 若設定檔只有路徑引用錯誤，可做最小限度的路徑修正；修正不得改變樣本、標註或 split，且必須寫入工作紀錄。

## 已知狀態

- Detection 資料集既有 README 說明其標註由相鄰 pose 資料衍生，且目前只提供 train／valid；缺少 test 不構成自行新增 split 的授權。
- `detect_dataset/data.yaml` 目前的 `path` 值仍指向舊位置 `/home/uxin0/yolo/original/pose/detect_dataset`，與本機固定來源不一致。正式執行前只能修正路徑引用，不得藉此重建或重切資料集，並須留下紀錄。

## 使用者已核准的不可變衍生版本

2026-08-22 使用者明確核准 architecture_2 建立獨立衍生目錄、grouped split 與四筆座標修補。
這是上方「除非使用者明確授權」的已記錄例外，不代表可再建立另一套 split。

正式版本固定為：

- 實際資料：/home/uxin/yolo/original/pose/derived/bbat5-v1/
- 建立工具：/home/uxin/yolo/yolo_achitechure/achitechure_2/src/achitechure_2/pose_data.py
- 操作說明：/home/uxin/yolo/yolo_achitechure/achitechure_2/README.md
- Git metadata：/home/uxin/yolo/yolo_achitechure/achitechure_2/artifacts/datasets/bbat5-v1/

bbat5-v1 的固定契約：

1. 原始 Pose／Detect 目錄維持唯讀；Pose labels 是 ball／bat 的權威來源，Detect view 由修補後每列前五欄衍生，原始 Detect labels 只作一對一 audit。
2. 依檔名 .rf. 前 prefix 分組，以 seed 0 建立 90%／10% formal train／val；同源 augmentation 不得跨 split。
3. search train／val 只在 formal train 內再次分組；formal val 不參與搜尋。
4. 共有 6,647 images、2,578 source groups；formal train／val 是 5,964／683，search train／val 是 5,364／600。
5. 與 COCO train 重疊的 331 groups 固定留在 train，formal／search val overlap 都是 0。
6. 四個極小負座標只在衍生 Pose labels clamp 為 0，逐筆保存在 patch manifest；原始 label tree hash 維持不變。
7. Pose／Detect 共用相同 assignment；不建立不存在的 test split。
8. v1 不可覆寫。資料或規則需要改變時，必須由使用者核准並建立新版本，例如 bbat5-v2。

其他子專案若需要相同 grouped split，應直接引用 bbat5-v1 或其 manifests，不得從原始目錄再自行
切一次。重建命令只會建立資料，不會啟動 Pose 訓練：

    cd /home/uxin/yolo/yolo_achitechure/achitechure_2
    export PYTHONPATH="$PWD/src"
    /home/uxin/yolo/.venv/bin/python -m achitechure_2 prepare-pose-data --execute

預設 Git 只提交 README、dataset YAML 與 split／patch／source-audit／COCO-exclusion／rebuild
manifests。2026-08-22 使用者另明確核准 architecture_2 上傳同一份 bbat5-v1 的完整可攜 snapshot：

- 唯一位置：`/home/uxin/yolo/yolo_achitechure/achitechure_2/artifacts/datasets/bbat5-v1/github-dataset/`。
- 允許物化 Pose／Detect 影像與 labels，但不得改變既有 assignment、四筆 patch 或 2.0.0 lineage。
- snapshot 必須使用一般檔案與相對 split path，不得提交會在 GitHub clone 後失效的 symlink。
- 仍禁止 checkpoint、weights、cache、runs、`.pt`、`.pth`、`.onnx` 與 deployment engine。
- 建立與驗證使用 `export-github-dataset --execute`、`validate-github-dataset`；是否跑 Pose 仍是
  獨立的使用者決定，資料上傳不代表授權訓練。

## 可追溯性

所有相關修改、實驗、驗證、困難與解法，依[工作紀錄規範](../worklogs/README.md)以中文留存。全域入口見[根層 README](../../README.md)。
