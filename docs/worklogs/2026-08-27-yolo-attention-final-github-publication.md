# 2026-08-27：YOLO26m Attention final GitHub 發布

## 任務與範圍

依使用者明確要求，將 `yolo_attention_final/final/` 完整發布到
`uxin0205-tech/yolo` 的 `main`，包含正式權重與訓練 parent checkpoints；
commit subject 固定為 `5060ti ForVic 0827`。

發布主體只包含可攜 final workspace；另依 repository 規範同步本工作紀錄、
`docs/worklogs/README.md` 索引與根 `README.md` 入口。沒有加入主工作區其他
未提交內容，沒有修改或發布 COCO2017／BBAT5 v1 的影像、標註或 split。

## 變更內容與原因

- 將遠端既有的精簡 `yolo_attention_final/final/` 更新為完整 82-file workspace，
  總大小 160,563,975 bytes。
- 發布內容包含完整 README、Binary Q/K＋PWL 演算法文件、訓練／結果文件、
  37 個 production Python 模組、12 份 training YAML、9 個功能測試檔、
  CPU smoke、CLI、queue／evaluation code 與 machine-readable results。
- 依使用者明確要求保留三個權重：
  - `pwl-final-best.pt`：44,305,659 bytes。
  - `weights/v1-br-best.pt`：71,574,771 bytes。
  - `artifacts/runs/s0-phase-b-bittrue/checkpoints/evaluated-variant.pt`：
    44,306,619 bytes。
- 三個權重都低於 GitHub 100 MB 單檔限制。根 `.gitignore` 原本忽略
  `*.pt`／`weights/`，因此只在授權的 final scope 使用 force-add。

## 驗證方式與結果

- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q`
  - 結果：`21 passed`。
- `ruff check --no-cache .`
  - 結果：通過。
- `ruff format --check --no-cache .`
  - 結果：55 個 Python 檔案格式正確。
- `python run.py check`
  - 結果：正式 SHA-256、YOLO26m scale m、80 classes、兩個 Attention sites
    與 Bit-True PWL contract 全部通過。
- 從 final 外部目錄執行 `scripts/smoke_cpu.py`
  - 結果：CPU forward 成功並回報兩個固定 Attention sites。
- 複製到最新 `origin/main` 的隔離 worktree 後重新稽核：
  - 82 個 regular files、總大小 160,563,975 bytes。
  - 三個權重 SHA-256 分別為
    `c989aeed09de7663ad093d32d098e5fc889cf04924fa1162efaf886869de0123`、
    `9e2ca0d93793785f0f7e514d876580204070bac57b4163d7c5f0c5ca352b9c3f`、
    `2f5a18e37eca6f780e1aa52c9aeb50306e4aba7f14e1c22a8b499bcf8a40d8ac`。
  - 沒有 `.orig`、`.rej`、`.pyc` 或 cache 目錄。

## 困難與解法

- 主工作區 `main` 落後遠端且含大量無關未提交內容。先 fetch 最新
  `origin/main`，再建立隔離 temporary worktree，只複製授權範圍，避免把
  其他使用者內容混入 commit。
- 根 ignore policy 會排除所有 `.pt`；以限定路徑的 `git add -f`
  明確加入三個使用者要求的權重，沒有放寬全 repository ignore 規則。

## 未解事項或風險

- final 不包含 COCO2017 images／labels；新環境仍需執行
  `python run.py configure --data-root /絕對路徑/coco2017`。
- 本次是整理與發布，不是新的 GPU 訓練或完整 COCO accuracy evaluation。
- 大型權重進入一般 Git 歷史後，即使未來從最新 tree 刪除，舊 blob 仍會留在歷史；
  本次是依使用者明確要求直接發布，未改用 Git LFS。
