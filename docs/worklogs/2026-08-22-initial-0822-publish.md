# 2026-08-22：以 Initial 0822 發布全域基本規則

## 任務與範圍

把 BBAT5／BBT5 固定資料集、中文報告與完整工作紀錄等全域基本規則發布至 GitHub `main`，commit 主旨固定為 `Initial 0822`。

## 變更內容與原因

- 保留最新 `origin/main` 已有的根層 `AGENTS.md`、`README.md`、資料集規範、工作紀錄與 `achitechure_1` README 連結。
- 保留遠端後續加入且已獲使用者核准的 `bbat5-v1` 不可變衍生版本契約，不以本機較舊文件覆寫。
- 新增本發布紀錄並更新 `docs/worklogs/README.md` 索引，使本次 commit／push 也符合全域可追溯規則。
- 本次沒有修改、複製、重切或提交任何資料集影像、標註、cache、symlink、checkpoint 或實驗產物。

## 驗證方式與結果

- 已先執行 `git fetch origin main`，並以最新 `origin/main` 建立隔離 worktree。
- 已確認遠端基準包含全域代理規則、根層 README、BBAT5 資料集規範、工作紀錄索引與 `achitechure_1` README 連結。
- 提交前已檢查本次精確差異、Markdown 相對連結、`git diff --check` 與未追蹤檔案清單，結果均通過。
- 本次只新增與更新 Markdown 紀錄，未改動程式、模型或設定，因此不執行 `pytest`。
- 推送後核對 GitHub `main` 的 commit hash 與主旨；實際 hash 由交付回報保存。

## 困難與解法

1. 本機 `main` 相對遠端為 ahead 1、behind 40，且含大量無關未追蹤工作，直接 commit／pull 可能混入內容或造成衝突。解法是從最新 `origin/main` 建立臨時 worktree，只製作本次兩個文件差異，再以 fast-forward 推送。
2. 要發布的基本規則已存在於最新遠端，且遠端版本另有較新的 `bbat5-v1` 授權契約。解法是不重複覆寫，改以本紀錄確認與封存發布狀態，保留完整遠端內容。
3. 原工作樹有 `.orig` 備份及其他使用者檔案。解法是全部排除在提交外，也不在未取得刪除授權時清理。

## 未解事項或風險

- 原工作樹的本機 `main` 仍維持原有分歧與未追蹤內容；本次透過隔離 worktree 發布，未擅自合併、重設或清理該工作樹。
