# 2026-08-27 GitHub yolo_combine 完整專案發布

## 變更內容與原因

使用者在 `final/` 已發布後，進一步要求將整個 `yolo_combine/` 上傳到
`uxin0205-tech/yolo` 的 `main`。本次發布範圍採 repository 的 source-managed 邊界：

- 納入根層 README、AGENTS、CONTEXT、pyproject、資料設定、`src/`、`tests/`、
  `scripts/`、`docs/`、`reports/` 與 Full35／Partial75 variant 定義；
- 保留上一個 commit 已完整發布的 `final/`，包括程式、報告、圖表與 Git LFS 權重；
- 依 `.gitignore` 排除 `.venv/`、Python／pytest／ruff cache、`artifacts/`、`runs/`、
  final 外的 `.pt/.onnx/.engine`。這些是本機環境、可重建 runtime View、原始 run 或
  中間 checkpoint，不屬可 clone 的 source-managed 專案；本機內容不刪除。

父 repository 的本地 `main` 有未提交內容且與遠端分歧，因此仍從最新
`origin/main` 建立隔離 worktree，只同步 `yolo_combine/`，不 stage 或改寫父工作樹。
commit 訊息沿用使用者指定的 `5090 Full35 0827`。

## 驗證方式與結果

- 發布前盤點本機 `yolo_combine/` 約 42 GB：`.venv` 7.2 GB、raw `artifacts/runs`
  約 28 GB、`final` 6.4 GB，其餘為 source-managed 專案內容。
- 隔離 worktree 的 `final/full35/verify.py`：`valid=true`，407 個 manifest 檔案；
  `final/full35-j2-archive/verify.py`：`valid=true`，308 個 manifest 檔案。
- 隔離 worktree 的 CUDA 隱藏完整測試：`121 passed, 3 skipped`；三項只因需要 CUDA，
  本次沒有占用 GPU。
- staged diff 為 151 個檔案，且全部在 `yolo_combine/`；`final/`、raw artifacts、
  cache、patch backup 與 editable-install metadata 都沒有被本次重寫或加入。
- `git diff --check` 只回報既有報告 CSV 的 CRLF、Markdown hard-break 與 EOF 空行；
  沒有程式碼行尾空白。為保持生成報告格式，本次不機械改寫研究輸出。
- 發布前後都會確認 staged path 只在 `yolo_combine/`、LFS dry-run 無漏傳，以及
  遠端 `main` hash／subject／tree 與本地發布 commit 一致。
- 全程不啟動 GPU、不重新訓練，也不重算 AP。

## 遇到的困難及解法

- 「整個資料夾」含 42 GB 本機環境與 raw run：依專案既有 `.gitignore` 區分
  source-managed project 與本機實驗儲存；正式交付成果已由 `final/` 保存。
- parent worktree 髒且分歧：使用最新遠端建立暫時 worktree，避免混入無關變更。
- `final/` 權重受全域 `*.pt` ignore：沿用已提交的 `final/.gitattributes` 與 LFS
  pointers，不重新上傳重複 checkpoint。
- 系統提供的 `apply_patch` 因 bwrap loopback helper 錯誤無法讀寫；改用等價且可稽核的
  `git apply` patch，沒有使用直接覆寫檔案的 shell 指令。

## 未解事項或風險

- raw `artifacts/`、runtime dataset symlink、歷史 smoke／diagnostic checkpoint 沒有發布；
  若日後確實需要逐 epoch 原始 run，應另建有配額規劃的 LFS release，不應混入主 branch。
- 目前只有 seed0；seed1、Partial75 正式訓練、FPGA/HLS 與真實 latency／energy 仍未執行。
- 困難：如上；其餘無。
