# 六組短 QAT 接續執行

## 變更與原因

依使用者確認，不把 PTQ 未達標等同格式無效。40 組區域 PTQ、148 路徑 × 4 格式共 592 組單層探測已完成，狀態無錯誤。探測僅為固定樣本輸出敏感度，不是逐層 mAP 驗證。

保留原三組計畫，新增獨立 `selected-qat-jobs-v2.json`，共六組各最多 5 epochs、patience 5、AdamW，資料 assignment 與 parent 不變：

1. Pose tower Fixed SD4：固定所選權重尺度。
2. Pose tower LS-SD4：相同區域學習尺度，與第一組配對。
3. MASF exact ternary。
4. MASF Paper-TWN。
5. MASF filterwise TWN：與前兩組比較同區域三元格式。
6. Detect predictor LS-SD4：PTQ 總 mAP50 下降 1.575 pp、mAP50–95 下降 5.339 pp，列為有限恢復試驗，不冒稱達標。

每次僅替換指定區域，其餘沿用已量化 parent；148 權重路徑、124 activation quantizers 的覆蓋保持。純整數部署尚未完成，保護算子不因此宣稱整數化。

## 執行與驗證

新 supervisor 每組先 CPU 真實圖預檢，再等待 GPU 空閒、串行 QAT；完成需核對 plan hash、五回合完成證據及 checkpoint hash。已存在未完成 run 時停止要求明確 resume，不自動覆寫。訓練等待由子程序 wait(600) 負責；模型端 blocking monitor 僅讀狀態 JSON。

六份機讀計畫已通過 Full35QATPlan 解析。新增測試涵蓋完成證據、回合數、血緣、checkpoint 雜湊與子程序錯誤；連同 schedule、plan、runtime、blocking monitor 測試共 43 passed（23.82 秒），ruff 通過。串行 supervisor 已啟動；當下進入第一組 CPU 預檢，尚不宣稱 QAT 已完成。

## 困難與解法

系統無 `python` 別名且 sandbox bwrap 失敗；使用專案虛擬環境及經核准的工作目錄命令。原雙門檻報告將恢復候選列為 reject；另建 admission 檔明記 recover 與理由，保留原 PTQ 數字及原報告不變。

## 未解事項與風險

### 第一組完成事件

blocking monitor 回傳 completed_jobs 633、current_index 1、pose-ls-sd4 CPU 預檢、error null；證實第一組完成後 queue 自動接下一組。Fixed-SD4 完成 5 epochs／2315 macro steps，未觸發提早停止。五回合 16 項搜尋指標均通過雙門檻。第 5 回合最差總 mAP50／mAP50–95 下降分別 0.834／1.008 pp；BBAT box mAP50–95 由 PTQ 81.3896% 回升至 83.3527%，增加 1.9631 pp。COCO Person box mAP50／mAP50–95 則較 PTQ 下降約 0.076／0.072 pp，非全項改善。

新增可重建 `scripts/summarize_continuous_qat.py` 與逐項報告，來源保留 SHA-256；不操作 GPU、不停止第二組。困難：多組彙整必須先列完摘要表再列逐項表，已調整迴圈避免混入不同欄位。未解：LS-SD4 配對、三元與恢復候選待完成，export 重驗與 formal 尚未進行。

彙整程式 ruff 通過；新增逐項回升、雙門檻與非法數值拒絕測試 5 passed（0.01 秒），報告重新生成成功，顯示 1/6 completed。

### 第二組完成事件

blocking monitor 回傳 completed_jobs 634、current_index 2、masf-exact-ternary running_qat、error null。LS-SD4 已完成 5 epochs，五回合的 16 項搜尋指標均通過。第 5 回合最差總 mAP50／mAP50–95 下降為 0.896／1.043 pp；BBAT box mAP50–95 為 83.4421%，較同 parent PTQ 增加 2.0525 pp。第 4 回合的最差總下降為 0.764／0.975 pp，提醒不可把 last 一律當作最優 checkpoint。

同第 5 回合比較，LS-SD4 較 Fixed-SD4 的 BBAT box mAP50–95 高 0.089 pp、COCO Person box mAP50 高 0.105 pp，但 COCO overall box mAP50 低 0.062 pp；無單一方案全項支配，先保留配對結果，不宣稱學習尺度必然較好。報告生成器已核對兩組 plan／accepted 雜湊及每組五份完整指標，重建成功（2/6 completed）。困難：無。未解：三元三組與 Detect predictor 恢復候選仍待完成；未使用 formal，也未中止第三組。

QAT 精度是否回升尚待實測；本批次非六組全達標保證。三元的整區敏感案例後續可再縮小子層，但不一次全部訓練。六組完成後依 16 指標、量化成本與耗時選擇 backbone→neck→head 累積驗證，不能直接拼接各自最佳。四天為軟目標，尚未宣稱全模型最終驗證完成。
