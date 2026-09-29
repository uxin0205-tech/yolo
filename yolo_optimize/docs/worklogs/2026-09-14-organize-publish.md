# 2026-09-14：本輪整理與 GitHub 發布

## 範圍與授權

使用者要求整理並上傳 GitHub，commit 名稱逐字採用 `5090 Uodate 0914`，不修正為 Update。依 finish-work 技能完成結果／文件／驗證／發布分開核對，採單一主代理。

發布 MASF B5、BinaryQK scale/bias E5、從其出發的 Rep17／Rep20／Rep17＋20 程式、設定、結果與圖；不新增 GPU、不改正式模型、不刪除權重、cache 或舊方案。大型權重留本機，53 份 .pt 共 10,354,210,367 bytes 全部重新計算 SHA-256，索引放 reports/update-0914/checkpoint-manifest.json。

## 整理內容

新增統一 0914 報告入口、48 列比較 CSV、架構 SVG、checkpoint manifest 與保留清單；root README 按最新結果／正式選用／歷史資料區分，更新 Pose MASF 文件中過時的 GPU 未啟動敘述，保留研究推導。舊方案標示已被 BinaryQK E5 後接 Rep 取代。

## 驗證

既有 queue ALL_DONE、三組 summary completed／E5 BitTrue 与 Float、各五回合 state 已在完成稽核確認。本次 53 份 SHA 全部計算完成；沒有重新執行已通過的 GPU／CPU 模型測試。發布副本另做 AST、JSON 有限值、CSV、SVG／PNG、Markdown 路徑與 git whitespace 檢查，精確結果由 publication-check.json 與 Git 稽核記錄。

## 困難與處理

原主工作樹有大量與本次無關的未追蹤資料及上層 AGENTS.md 修改，不能直接 git add 全部。先 fetch origin/main（31e6b16a），新建 /tmp/yolo-publish-0914.r98z7m detached worktree，只覆入明確白名單的 optimize 內容。原工作樹、上層修改、既有其他臨時工作樹不碰。工作樹 checkout 尚未完成時的狀態查詢曾列出大量暫態差異，已等待 checkout 成功後才進行任何 staging，沒有據此刪檔或復原。

## 發布與風險

只推送一般 Git 報告／程式／必要小型結果；本機 checkpoint 連結在 GitHub 副本明記未上傳，不冒充可下載權重。推送後會核對遠端 main commit hash／subject，再保存本機成功收據。只收束本次專用且乾淨的發布 worktree。

沒有新訓練錯誤；模型限制為無獨立 test／多 seed、硬體 latency／energy 未完成實測、Rep 三組未過採用閘。是否再訓練或採用仍未變更。

## 提交前格式問題修正

git diff --cached --check 發現 Matplotlib 產生的 b-training-curves.svg 有行尾空白，第一次 staged 檢查未通過，未提交／未推送。僅移除 SVG 行尾空白，XML 重新解析通過；圖中數據與路徑座標不變。重新建立發布 manifest 及 staging 後再驗證 whitespace，沒有放寬檢查規則。
