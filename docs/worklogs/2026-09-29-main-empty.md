# 2026-09-29：清空 GitHub main 內容並保留分支

## 任務與範圍

使用者明確選擇「清空內容但保留分支」。本次只清空 GitHub `uxin0205-tech/yolo` 的 `main` 最新檔案樹，保留分支及預設分支設定。已發布的 `Cleanup-v1`、其他遠端分支，以及本機程式、checkpoint 與資料集均維持現況。

## 變更內容與原因

- 在最新 main `1d409c4fe21d3287d6f19be48f4457407a6bb50e` 上建立一般提交 `4d63cf6db6ab4580e3c759a1774d4f375ee33144`，訊息為 `chore: clear main contents for future updates`。
- 新提交的檔案樹為 Git 標準空樹 `4b825dc642cb6eb9a060e54bf8d69288fbee4904`，追蹤檔案數為 0。
- 使用一般 fast-forward push，保留既有提交歷史；未使用 force push 或刪除分支。
- 使用 `git mktree`／`git commit-tree` 建立提交，沒有檢出或刪除本機工作檔案，也沒有更動本機 HEAD 與暫存區。
- main 必須保持完全空白，因此本次工作紀錄保存在本機 `docs/worklogs/`，同步更新本機索引、README 與先前清理紀錄。

## 驗證方式與結果

- 發布前 SSH fetch 取得最新 main，與預期來源 SHA 一致。
- `git ls-tree -r` 確認清空提交沒有任何檔案；parent SHA 等於原 main。
- 建立提交前後比對本機 HEAD 與 `.git/index` 的 SHA-256，均未改變。
- SSH push 成功，遠端 main 從 `1d409c4f` 更新為 `4d63cf6d`。
- GitHub 連接器確認 main 提交 SHA 與標準空樹 SHA 完全符合本機驗證結果；儲存庫預設分支仍為 `main`。
- GitHub 連接器確認 `Cleanup-v1` 仍是 `7d2cc18d75e7f06f251bb1a6d01512325e6367e8`，程式與報告基線未變。
- 本次只變更遠端檔案樹與本機說明文件，不執行模型測試、GPU、PTQ、QAT 或訓練。

## 困難與解法

- 沙箱命令仍受 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted` 影響，透過工具權限審查後執行必要操作。
- GitHub `git/trees/main` 對空樹回傳 404；改以 branch API 內的 commit SHA 與 `commit.tree.sha` 核對。遠端明確指向標準空樹，且提交 SHA 與本機已驗證的 0 檔案提交相同。

## 未解事項或風險

- main 的最新內容已清空，但既有 Git 歷史保留，舊 checkpoint 與 Git LFS 物件仍可能占用儲存空間；此次操作不代表遠端配額已釋放。
- 本機權重、資料集及未提交研究內容均保留。BBAT5 canonical `/home/uxin/yolo/original/pose/derived/bbat5-v1/` 的影像、labels 與 split 未改變。
- `yolo_optimize-cleanup` 未在本次刪除範圍內，維持原狀。

遠端：[main](https://github.com/uxin0205-tech/yolo/tree/main)、[Cleanup-v1](https://github.com/uxin0205-tech/yolo/tree/Cleanup-v1)。先前清理數據見 [Cleanup-v1 工作紀錄](2026-09-29-cleanup-v1.md)。
