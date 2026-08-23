# 2026-08-23：發布 original 資料並清除 Git 權重

## 任務與範圍

依使用者要求確認全專案未來 BBAT5 資料入口，發布本輪更新及 `original/` 的可用資料內容，
並以 `5090 CLEAR WEIGHT 0823` 提交。發布以最新 `origin/main` 的隔離 worktree 準備，不修改、
reset、pull 或清理原本 dirty 且分岔的本機 main。

## 變更內容與原因

- 上傳 raw Pose、歷史 Detect 與 canonical bbat5-v1 的影像、labels、設定、split 與 manifests。
- Git canonical image links 改為 repository-relative symlink，保留相同 image target 與 assignment。
- 將 `/home/uxin/yolo/original/pose/` 定為唯一 BBAT5 資料資產庫；新 run 只使用 canonical registry／YAML。
- 從目前 tree 移除 architecture_2 的 0822 重複 snapshot：26,612 個 tracked paths、
  724,846,884 aggregate blob bytes；歷史 commit 仍可追溯。
- 停用三個正式 snapshot／metadata CLI；保留的低階相容函式必須指定專案外目的地，否則 fail closed。
- 將 yolo_combine 的 symlink-only runtime View 從 `artifacts/datasets/` 改名為
  `artifacts/cache-views/`，避免被誤認為第二個資料集。
- 從 Git tree 移除 `original/weight/yolo11m.pt` 與 `original/pose/weight/yolo11m_bat.pt`；本機檔案保留。
- 排除兩個 Ultralytics `labels.cache` 與所有執行 cache／run。
- 排除 726,351,400-byte `detect_dataset.zip`：它通過 archive integrity 檢查，但超過 GitHub
  100 MB 一般 Git 單檔限制，且完整解壓 `detect_dataset/` 已納入。
- 同步正式 spec、AGENTS、README、資料規範、ADR 與 `.gitignore`。

## 驗證方式與結果

- `validate-pose-data`：valid／formal_ready=true；Pose／Detect 各 6,647、formal train/val
  5,964／683、4 patches、331 COCO-overlap groups 固定在 train、test=null、source untouched。
- `config-check`：valid=true；Ultralytics 8.4.90、C0–C3、spec 2.0.3 與
  `8b07bdb4195e88c8c9bf4a95dab782c75a77854ef5022ede4757290348a95471` 一致。
- architecture_2 CPU pytest：57 passed、1 skipped；skip 是未提供正式 Fusion Winner handoff 的預期 gate。
- yolo_combine CPU pytest：38 passed、3 skipped；三項皆為 CUDA-only integration tests。
- architecture_2 與 yolo_combine 完整 Ruff：全部通過。
- 最終 staged paths：66,635；Git rename detection 顯示 39,975 added、46 modified、17 deleted、
  26,597 renamed（多數是 0822 snapshot blob 在唯一 original tree 的內容重用）。
- 發行 tree：53,204 original regular files、13,294 repository-relative canonical symlinks、
  0 absolute／broken symlinks、0 architecture dataset paths、0 新增權重／cache、0 超過 99 MB 檔案。
- staged diff secret pattern scan：0；`detect_dataset.zip` archive integrity：passed。
  commit／remote hash 由最終交付訊息記錄，避免在同一 commit 內形成自我雜湊循環。

## 困難與解法

- 本機 main ahead 1／behind 43 且有大量 unrelated untracked files：從最新 `origin/main` 建立
  隔離 worktree，只複製授權範圍。
- `gh` 未登入：無法建立／更新 GitHub Issue；Git 發布改由既有 SSH remote 執行，狀態在交付說明揭露。
- raw tree 含超限 zip、cache、權重與本機絕對 symlink：分別以解壓目錄、忽略規則、Git tree
  刪除與 repository-relative symlink 解決，不改本機原始資料。

## 未解事項或風險

正式訓練、Pose validation、C0–C3 GPU 實驗仍未執行；資料發布不代表取得 GPU 訓練授權。
