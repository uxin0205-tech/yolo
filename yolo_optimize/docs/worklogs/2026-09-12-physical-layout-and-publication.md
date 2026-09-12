# 2026-09-12：實體資料夾重整、詳細報告與 GitHub 更新

## 需求與處理範圍

使用者要求實際整理資料夾，不只建立入口，並要求詳細報告、沿用 `5090 Done 0912` 作為 commit 名稱，以整理版取代舊發布內容。依 finish-work 技能進行實體遷移、歷史保存、驗證及隔離發布；全程只使用主代理。

「覆蓋」限定為本次授權的 optimize 發布樹及相應根層入口，不刪除本機權重、不覆蓋其他專案，也不 force push 或重寫遠端提交歷史。

## 實際變更與原因

共 14 筆目錄搬移：九個研究相關根目錄移入 `experiments/`，維護工具移至 `tools/`，原始提案移至 `proposals/`，三個日期型報告目錄改為 `reports/final`、`reports/direction1`、`reports/publication`。checkpoint 隨所屬實驗搬移；沒有建立根層舊路徑 symlink。

原始搬移盤點 5,130 個一般檔案，第一階段更新 395 份文字檔的路徑，4,735 個檔案在該步驟的大小、mtime_ns、inode 不變。後續另重寫導航、補充總報告並修正模型索引。精確搬移及路徑修改清單見[manifest](<../history/layout-v2-20260912/manifest.json>)，修改前原文保留在同目錄 originals。

共盤點 704,592 個既有 symlink，未更換任何 target；主要是 runtime 資料 View，沒有複製、抽樣或改標註。完整清單無損壓縮留在本機 `local-full-manifest.json.gz`，GitHub 發布精簡 manifest，避免把大量重複路徑塞入報告。

實驗目錄內部布局保留，只有外部 YOLO_ROOT 等少數根語意需調整。設定檔與原始指標中必要的絕對路徑已更新，數值不重新計算；模型 bytes 及其內嵌 metadata 不修改。已完成封存包不動，只更新 archives 的入口連結。

## 報告交付

[詳細總報告](<../../reports/final/README.md>)涵蓋 BinaryQK、HOG、RepConv、MASF、融合、activation、KD 及推論，保留各階段數據、架構圖、MAC 算式、超參數、負結果及限制。新增最終採用／未採用判斷、實體遷移與復現流程；將已完成的 one2many 推論從待辦改為已完成。

另提供[目錄地圖](<../WORKSPACE.md>)、[操作指南](<../OPERATIONS.md>)及[模型索引](<../../reports/checkpoints/README.md>)。不建立同內容的另一個「final-final」，也不將檔名 best 視為跨階段最佳證明。

## 已完成驗證

- `CUDA_VISIBLE_DEVICES='' .../python -m pytest experiments/tests -q -p no:cacheprovider`：83 項全部通過，9.85 秒。
- 以 `SelectedSource`、固定 SHA 與 `weights_only=True` 在 CPU 重建現行 qSiLU 模型；Detect／Pose 的 BitTrue 160×160 基本 forward 均有有限值，不寫入新 run。
- 六個重要 checkpoint 的 SHA-256 與搬移前索引一致。
- 177 份修改的 JSON 逐份比較，除路徑替換外內容完全相同；未更動 AP 數字。
- 167 份 Python 的 AST 通過；858 個現行文件連結均存在。這是當時完整檢查數，後續新增工作紀錄及發布清單另做範圍驗證。

這些是路徑遷移與 CPU 行為驗證，不是新 AP 評估、全歷史 exact resume、ONNX／板端效能驗證。本次沒有 GPU 訓練或推論工作。

## 發布方式

以最新 `origin/main` 建立隔離 worktree，保護本機原 checkout 的不相關修改。舊 optimize 發布樹先移出隔離 worktree 保存，再建立新樹；新提交只包含 optimize 與根層必要入口更新。commit 名稱固定 `5090 Done 0912`。

發布報告、JSON／CSV 指標、設定及研究程式；排除權重、runtime 影像／labels、cache、PDF、大封存及遷移原文快照。快照及全部模型仍留本機，線上明確標示本機限定來源。最終發布 hash 與檢查結果見 reports/publication 的交付紀錄。

## 困難與解法

只搬目錄會造成絕對路徑、外部父目錄解析及發布固定輸出位置失效；依舊新對照修正，再以數值等價與 CPU 模型重建確認。大量 symlink 的逐筆清單體積大，改為本機無損壓縮與可發布摘要。其他困難：無。

## 未解事項與風險

舊命令及外部使用者保存的絕對路徑必須依對照更新；checkpoint 內嵌的歷史 metadata 保留原貌，續訓前需解析新位置。舊 queue 不可直接重新啟動。未把全部歷史寬鬆 pickle 載入重構為新安全介面，正式入口使用已核對的安全載入。

本次未刪除失敗紀錄、cache 或舊模型，不宣稱回收空間；同磁碟封存仍不是異機備份。模型精度及推論取捨維持原實測結論，沒有新訓練成果。


## 提交後交付狀態（本機追加，未包含於該 commit）

新提交 `8786b58fbc9f203d953f27e19580b9a1b39d1a38` 已建立，名稱 `5090 Done 0912`。發布檢查 1,373 份來源檔、906 個連結、897 份 JSON、72 份 CSV 通過，舊發布內容依新路徑映射後遺漏 0。另確認本機 364 個模型檔的大小／mtime_ns／inode 不變，六個重要模型 SHA 一致。

push 被平台 auto-review 拒絕，理由是需要使用者明確確認 repository、分支及研究程式／設定／內部結果的發布範圍，避免不當對外揭露。未採取替代通道繞過，已保留本機分支與隔離 worktree；需確認 `uxin0205-tech/yolo`、`main` 及上述文字發布內容後才能續傳。目前不得宣稱整理版已上傳。

完整狀態見 PUBLISHED.md（本機／歷史參照：`../../reports/publication/PUBLISHED.md`；未隨本次報告發布）。其他未解事項維持前述；模型及數據沒有刪除。
