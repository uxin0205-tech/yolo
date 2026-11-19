# 2026-09-08：報告整理、恢復交接與 Stop 發行

## 變更與原因

使用者核准精確清單，要求保留日後恢復能力並以 `5090 Stop 0908` 同步 GitHub 同位置。使用 finish-work 進行交接、備份、驗證與隔離發布；量化仍依最新決定延後，不啟動 GPU／queue／monitor。

- 原 25 份報告：10 份原位保留、12 份移至 docs/archive/reports、3 份重複報告備份後刪除。精確檔案與 SHA 見[執行紀錄](../organization/report-retention-execution-2026-09-08.json)。
- 22 份文件共 76 處歷史連結修正；額外更新現行導航、整理狀態及工作紀錄索引。封存報告的非連結文字不變，不改實測數字。
- 新增[恢復交接表](../RESUME.md)，保留同 benchmark 契約、parent／checkpoint 區別、已完成／未完成、量化方法、下次接續順序及需另行移交的本機資產。
- [PUBLICATION_0908.yaml](../../PUBLICATION_0908.yaml) 明確區分可發布快照與本機完整資料。本次納入凍結 queue metadata，不代表 live GPU 狀態；不改舊發行契約。

## 驗證方式與結果

- CPU 全套：`CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/uxin/yolo/.venv/bin/python -m pytest -q -p no:cacheprovider`，**351 passed in 50.33s**。
- 25 份原報告的備份 SHA 核對通過；3 個刪除來源不存在、12 份封存目的地存在、報告根目錄保留 10 份（不含 README）。
- 封存 12 份的非連結文字與備份一致；當前機讀依賴保留，不更改 Q3 report pin、parent、累積 queue plan/status 與原始結果。
- 全域 `ruff check .`：6 個既存問題，位於 full_coverage_successor.py 與其測試（import、例外型別、未用變數及 __all__ 排序）。本次不改已綁定實驗血緣的程式，不宣稱全域 lint 通過。
- 發布前核對現行／封存入口連結、關鍵 SHA、候選檔案安全與可發布檔案清單；提交時逐檔核對 Git blob 與本機 bytes，且 diff 只允許 yolo_quantize/。完整檔名與 SHA 見[發布清單](../organization/publication-files-0908.json)。
- 實際核對：631 個發布檔案的 Git index／暫存副本與本機 bytes、mode 一致，範圍外變更 0；現行／封存入口斷鏈 0，4 個關鍵血緣檔 SHA 通過。清單本身不計入自身雜湊，另由 Git blob 核對。
- 推送採一般 fast-forward，不 force push；推送後核對遠端 main commit 與本次 commit 相同。最終 commit hash 由 Git commit 與交付訊息提供，避免把自身 hash 寫回同一 commit 形成循環。

## 困難與解法

- 原工作區包含其他專案的未提交／未追蹤內容：使用專用暫存 sparse worktree，基於 origin/main 建立發布，不把其他專案納入、不重設原工作區。
- 暫存 worktree 最初使用 no-checkout，索引尚未初始化；在同步前以 read-tree 初始化，確認索引與 HEAD 一致後才處理子樹，未提交或推送該暫態。
- 首次文件補丁缺少 Begin Patch 標記，工具拒絕且沒有寫入；補齊標記後重試成功。
- diff whitespace 檢查發現 V30 歷史 archive-handoff.json 原有 EOF 空行；只為該精確路徑補 .gitattributes 規則，不為格式美化改寫歷史 metadata bytes。
- 首次 commit＋push 請求被自動審核攔下，當時未提交／上傳。其要求確認目的地與所有權；GitHub CLI 未登入，後以唯讀 SSH 驗證確認帳號為 uxin0205-tech，公開 API 確認 uxin0205-tech/yolo 的 owner 同名、公開庫且預設 main。補足證據後才重新提交推送審核，不繞過拒絕。
- 舊報告與工作紀錄可能同名：解析實際 Markdown 目標，避免全域替換；保留機讀／hash 依賴報告原位。
- 刪除備份在 `/tmp/yolo-quantize-report-backup-0908.rJlv9s/`，可恢復但不是永久保管；歷史獨有成果保留於版本庫歷史區。

## 未解事項與風險

量化、額外 QAT、逐層 mAP 擴展、finalists／formal／純整數部署繼續延後。GitHub 不是完整訓練環境備份：本機權重、完整 runs、資料集、log 與大型 predictions 未上傳，也未刪除。換機恢復需另行移交資產並核 SHA。benchmark 一致目前為書面契約，未新增自動 gate。沒有處理本輪範圍外的既存 lint 問題。
