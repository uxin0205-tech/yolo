# 補充相同比較條件的 benchmark 契約

## 變更與原因

使用者強調 benchmark 必須相同，重點是公平比較。於 CURRENT_PLAN 增補恢復量化前必要的評估／控制變因契約，未啟動實驗、未改結果或機讀執行計畫。

固定評估資料、split、輸入、類別與 evaluator；權重方法比較固定 parent、activation、替換路徑及校正預算；累積比較固定同一步驟的前綴；QAT 固定起點、訓練預算與選 checkpoint 規則；硬體量測固定裝置與計時口徑。只允許明列的研究變因不同，不要求各方法訓練後產生相同 checkpoint。

歷史不匹配結果保留為探索，不能因 benchmark 同名就混入隔離方法排名。PTQ／QAT 額外訓練成本、fake quant／整數部署、search／formal 分開標示。沿用不可變 BBAT5 v1，不重切／抽樣。

## 驗證方式與結果

對照現行封存決策及比較限制，確認量化仍延後；本次僅兩份文件及索引變更。CPU 文件檢查：三份 Markdown、44 個本機連結、尾端空白與必要契約文字均通過，0 errors；交接狀態仍為 deferred_by_user、new_gpu_jobs=false。不跑 GPU、訓練或程式回歸。

## 困難與未解事項

困難：無。benchmark 同名不保證控制變因一致，因此明列必要條件與不匹配時的標籤。新增內容是規劃契約，尚未實作自動身分驗證 gate；未宣稱所有歷史比較已補成同條件。日後恢復需使用者明確要求，目前不接 queue 或 monitor。
