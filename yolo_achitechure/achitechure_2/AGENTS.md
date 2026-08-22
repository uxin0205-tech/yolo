# architecture_2 工作規範

## 規格與語言

- [`EXPERIMENT_SPEC.md`](EXPERIMENT_SPEC.md) 是本目錄唯一正式規格。
- `plan.md` 只作為規格入口，不得複製另一套規則。
- 使用者文件、進度、報告與工作紀錄使用繁體中文；程式識別字與必要原文可保留英文。
- 修改術語前先讀 [`CONTEXT.md`](CONTEXT.md)；重大且難以逆轉的決策記錄於 `docs/adr/`。

## 工作邊界

- `yolo_combine` 負責融合實驗與 winner 選擇；本目錄只承接其不可變 handoff revision。
- 本目錄只評估 C0、C1、C2、C3，不得重新實作雙權重切換、融合選型或自動挑選 C_best。
- 正式長訓練、CUDA 測試、QAT 與 GPU profiling 必須先取得使用者明確授權。
- GPU 未授權時，一律以 `CUDA_VISIBLE_DEVICES=""` 執行 CPU 驗證。

## 資料

- BBAT5 Detect 原始資料：`/home/uxin/yolo/original/pose/detect_dataset/`。
- BBAT5 Pose 原始資料：`/home/uxin/yolo/original/pose/dataset/`。
- 原始資料唯讀；任何分割或修補只能建立於版本化衍生目錄，且必須附中文 README 與 manifests。
- Git 只提交程式、YAML、README、split／patch manifests，不提交影像、labels 複本、checkpoint 或 runs。

## 驗證與提交

- 公開測試 seam 是設定檢查、資料準備、handoff 驗收、候選建構、checkpoint 重載與 CPU dry-run。
- 使用 pytest；每次修改留下中文工作紀錄並更新索引。
- GitHub Issue 規範見 `/home/uxin/yolo/docs/agents/issue-tracker.md`；若 `gh` 未登入，必須在交付說明中明記。
