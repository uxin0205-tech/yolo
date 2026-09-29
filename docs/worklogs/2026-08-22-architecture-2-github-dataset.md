# 2026-08-22：發布 architecture_2 完整 BBAT5 GitHub Dataset

> 歷史狀態：本紀錄描述 0822 當時的發行方式。0823 已將 `/home/uxin/yolo/original/pose/`
> 定為唯一資料位置，並從目前 tree 移除此 snapshot；詳見
> [0823 original 資料發布紀錄](2026-08-23-original-data-publication.md)。

## 任務與範圍

依使用者最新明確授權，將既有不可變 `bbat5-v1` 除 metadata 外，也以完整可攜 dataset snapshot
上傳 GitHub。程式、規格與文件提交固定命名 `5090 PRE 0822`；物化資料提交固定命名
`5090 CLEAR WEIGHT 0822`。資料上傳不代表授權 Pose 正式訓練、GPU 測試或 QAT。

## 變更內容與原因

- 將 authoritative spec 升至 2.0.2，SHA256 為
  `a09a1a659c231d0ea7dac42471f67d3071e760d6828846866a3a69cc388197df`。
- 新增 `export-github-dataset`：預設只規劃，只有 `--execute` 才在
  `artifacts/datasets/bbat5-v1/github-dataset/` 建立不可覆寫 snapshot。
- 新增 `validate-github-dataset`：驗證 Pose／Detect assignment、兩 view 影像內容、label 前五欄、
  formal/search split、relative paths、tree SHA256、零 symlink、零 test 與零禁止權重檔。
- 本機 `/home/uxin/yolo/original/pose/derived/bbat5-v1/` 維持原狀：影像仍是唯讀來源 symlink，
  不覆寫或修改 labels。GitHub snapshot 只改變儲存形式，不建立新 split。
- GitHub snapshot 在匯出 worktree 內先複製一份一般影像，再以 Pose／Detect view 內部 hardlink
  避免本機重複占用；Git commit 以相同 blob 去重，clone 後可呈現兩個一般檔案路徑。
- portable `data.yaml`／`data-search.yaml` 不含機器限定 `path`；split lists 只含 `./../images/...`
  相對路徑。
- `.gitignore` 只開放上述固定 snapshot；`.pt`、`.pth`、`.onnx`、engine、cache、runs 與
  checkpoint 仍禁止。

## 資料結果

| 項目 | 結果 |
|---|---:|
| 唯一影像 | 6,647 |
| Pose／Detect 影像路徑 | 13,294 |
| Pose／Detect label 路徑 | 13,294 |
| formal train／val（每個 view） | 5,964／683 |
| search train／val（每個 view） | 5,364／600 |
| 唯一影像 bytes | 360,488,995 |
| 兩 view 邏輯影像 bytes | 720,977,990 |
| labels bytes | 1,480,347 |
| snapshot 檔案數 | 26,602 |
| symlink／test／禁止權重檔 | 0／0／0 |
| 最大單檔 | 475,100 bytes |
| publication tree SHA256 | `344ab62b9fb12b6f6a65c11abf96a987ec8415e53465bdc66bafcf336300cb4f` |

資料 lineage 仍是 spec 2.0.0／
`75db239262de75998171a05a85c8755dea10c2bc76920dd2aabe8ff0dabb7a3b`；publication lineage 才是
spec 2.0.2。四筆座標 patch、331 個 COCO overlap groups 與 formal/search assignment 均未改寫。

## 驗證方式與結果

- `python -m pytest -q`：53 passed、1 skipped；唯一 skip 仍是尚未提供正式 Fusion Winner handoff。
- focused dataset／CLI tests：11 passed。
- `python -m ruff check .`：通過。
- `python -m achitechure_2 config-check`：valid=true，spec 2.0.2 與 C0–C3 全部一致。
- `python -m achitechure_2 validate-pose-data`：canonical v1 valid=true、source_untouched=true。
- `python -m achitechure_2 validate-github-dataset`：valid=true、6,647 unique images、零 symlink、
  零禁止權重、tree SHA256 與 manifest 一致。
- Ultralytics 8.4.90 `check_det_dataset(..., autodownload=False)` 已成功解析 Pose／Detect 的
  formal/search 四份 portable YAML。
- 未執行 Pose 正式訓練、GPU smoke、長訓練、latency／VRAM 或 QAT。

## 困難與解法

1. canonical v1 有 13,294 個指向 `/home/uxin/...` 的影像 symlink，直接提交後在 GitHub clone
   會斷鏈。解法是獨立物化一般檔案，並由 validator 禁止任何 symlink。
2. 兩個 task view 的影像邏輯容量約 721 MB。解法是匯出時以 snapshot 內部 hardlink、Git 以
   相同 blob 去重；唯一影像內容約 360 MB。最大單檔僅 475,100 bytes，因此不使用 Git LFS。
3. 本機 `main` 有獨有提交、缺少遠端提交且含大量未追蹤工作。解法是從最新 `origin/main`
   建立暫存 worktree，只複製本次核准範圍，不 reset、pull 或混入其他專案。
4. 發布期間遠端新增 `333c4a2 Initial 0822`。解法是以該最新 commit 為基底，保留其新增工作紀錄，
   不覆蓋或重寫歷史。
5. 先前已公開 commit 名為 `5090 PRE 0819: ...`。為避免 force-push 改寫共享 main，本次新增的
   程式／規格 commit 使用使用者指定的精確名稱 `5090 PRE 0822`，舊 commit 保留。
6. `gh auth status` 顯示尚未登入任何 GitHub host，因此無法建立或連結 Issue；Git push 仍使用
   已配置的 SSH remote。此限制在交付說明中明記。

## 未解事項或風險

- 完整 snapshot 會讓 Git 歷史增加約 360 MB JPEG 內容；clone checkout 因兩 view 路徑可能占用
  約 721 MB 影像空間。這是使用者明確核准的容量取捨。
- 正式 Pose 執行仍需使用者 opt-in；上傳資料不會自動啟動任何訓練。
- 尚待 `yolo_combine` 提供正式 Fusion Winner handoff，故仍沒有 C0–C3 Float 精度或 C_best。
