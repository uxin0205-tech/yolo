# 2026-08-23：發布 original 資料並清除 Git 權重

## 任務與範圍

依使用者要求確認全專案未來 BBAT5 資料入口，發布本輪更新及 `original/` 的可用資料內容，
並以 `5090 CLEAR WEIGHT 0823` 提交。發布以最新 `origin/main` 的隔離 worktree 準備，不修改、
reset、pull 或清理原本 dirty 且分岔的本機 main。

## 變更內容與原因

- 上傳 raw Pose、歷史 Detect 與 canonical bbat5-v1 的影像、labels、設定、split 與 manifests。
- Git canonical image links 改為 repository-relative symlink，保留相同 image target 與 assignment。
- 從 Git tree 移除 `original/weight/yolo11m.pt` 與 `original/pose/weight/yolo11m_bat.pt`；本機檔案保留。
- 排除兩個 Ultralytics `labels.cache` 與所有執行 cache／run。
- 排除 726,351,400-byte `detect_dataset.zip`：它通過 archive integrity 檢查，但超過 GitHub
  100 MB 一般 Git 單檔限制，且完整解壓 `detect_dataset/` 已納入。
- 同步正式 spec、AGENTS、README、資料規範、ADR 與 `.gitignore`。

## 驗證方式與結果

發布前已確認：canonical Pose／Detect 各 6,647 張、formal train/val 5,964／683、4 個 patch、
0 group leakage、0 derivation mismatch、0 broken image link；`source_untouched=true`。最終 staged
file counts、tree links、large-file scan、pytest、config-check、commit 及 remote hash 會在完成後補記。

## 困難與解法

- 本機 main ahead 1／behind 43 且有大量 unrelated untracked files：從最新 `origin/main` 建立
  隔離 worktree，只複製授權範圍。
- `gh` 未登入：無法建立／更新 GitHub Issue；Git 發布改由既有 SSH remote 執行，狀態在交付說明揭露。
- raw tree 含超限 zip、cache、權重與本機絕對 symlink：分別以解壓目錄、忽略規則、Git tree
  刪除與 repository-relative symlink 解決，不改本機原始資料。

## 未解事項或風險

正式訓練、Pose validation、C0–C3 GPU 實驗仍未執行；資料發布不代表取得 GPU 訓練授權。
