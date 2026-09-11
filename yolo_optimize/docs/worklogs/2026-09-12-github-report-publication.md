# 2026-09-12：BinaryQK 起始至推論的完整報告發布

## 任務與變更

使用者要求將從頭開始優化 BinaryQK 的各種報告統整並上傳 GitHub，commit 名稱指定為 `5090 Done 0912`。依 finish-work 技能，由主代理完成，不使用子代理。新增按研究階段排序的發布入口，保留既有詳細總報告、156 份左右的歷史／方法／結果文件及可追溯指標，不將 YOLO11m 歷史消融誤稱 YOLO26M 新結果。

本機 HEAD 與遠端 main 不同，且工作區有大量未提交內容，因此從遠端最新 main 建立獨立暫存 worktree；只複製 optimize 報告、數據、設定、必要研究程式及直接引用的上層 MASF 稽核。遠端根 README 與工作紀錄入口作小幅導航更新，不混入本機其他專案修改。

## 驗證與發布邊界

發布前檢查所有新增 Markdown 檔案連結、檔案大小與副檔名、JSON 可解析性、關鍵指標來源及 staged diff。發布轉換記錄於 `reports/5090-done-0912/publication-manifest.json`；已發布檔案改為相對連結，本機限定權重／影像／PDF不偽裝成可下載連結。實際驗證數量與結果保存於同目錄發布檢查檔。

不包含 checkpoint bytes、113 GB 封存、資料集影像／labels、cache、逐步大 log 或第三方 PDF；checkpoint CSV 的 SHA／來源只作索引，不表示權重上傳。原始權重、封存、失敗紀錄與使用者 dirty worktree 全部保留，沒有刪除提案或清理行為。未新增 GPU 工作，也不重跑既有正常訓練或 AP 驗證。

使用非強制 fast-forward push；若遠端在此期間前進，先檢查並整合本次限定報告，不覆蓋其他提交。上傳後以遠端分支 hash 與完整 commit subject 確認，不只依本機 commit 成功宣稱上傳完成。

## 困難與解法

原工作區較遠端舊且有大量無關修改，使用隔離 worktree 解決。原報告包含本機絕對路徑與未發布權重連結，發布版轉成相對連結或明確的本機參照，原始報告保留。資料集不變，沒有模型／指標修改。

## 未解事項

發布前實測：158 份 Markdown、853 個本機檔案連結可解析、166 個 Python AST、892 份 JSON 與來源內容一致；73 份 CSV 另核對欄位內容。没有模型／資料集／封存檔案或已知 credential 格式誤入。首次 diff check 發現新增 MASF CSV 的 CRLF 被判為行尾空白，只在發布副本轉成 LF、保留原始檔與數值，修正後再驗證。最終清單與 checksum 以 publication-check／manifest 為準。

這是報告與方法追溯發布，不是完全自足的模型部署包；重跑需要本機外部權重、客製來源與固定資料。研究仍未全面恢復精度，class routing、共享特徵後續及板端效能仍未驗證，不因 commit 名稱含 Done 就標記研究目標全部達成。
