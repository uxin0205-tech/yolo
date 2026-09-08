# 持續量化P0實作與共同parent診斷

## 變更與原因

依使用者明確執行授權開始P0。起算與96小時軟目標記錄於 `artifacts/queues/full-model-continuous-0907/implementation-start.json`。新增CPU parent preflight、同parent PTQ載入入口、600秒靜默monitor、5epoch日程測試、EMA投影重現、原權重分布與40格十區特殊格式矩陣生成器；不改舊V36結果。

PTQ使用鎖定QAT export且保留學得A8，不重新校正；incremental改對鎖定parent，total仍對accepted。舊parent候選不混入新排名。部分替換的未變更權重繼承原量化成本，不誤算FP32。新矩陣每格只變更一區，其餘已量化權重不重新投影。

## 驗證與結果

- 初次CPU嚴格重載通過：V36 best_joint為epoch index1、148權重路徑、124activation；16指標metadata與metrics一致，worst total mAP50 −0.0083473081、mAP50–95 −0.0115671260。
- monitor、同parent入口、日程及相關runtime回歸40項通過；追加繼承成本測試後其對應15項測試通過。投影回歸另2項通過；測試集合重疊，不能相加當獨立總數。
- 真實parent繼承成本驗證148路，weight codes＋FP32 scales估20,290,368 bytes；不含bias、protected及alignment，非實際packed大小。
- 分布分析已完成148路×原EMA／export兩view，共296列，含近零比、幅值直方圖及三元歸零能量。檔案：`weight-distribution.json`；不等於mAP sensitivity。

## 困難與診斷

獨立從EMA在CPU投影，不逐位元等同GPU export：145層存在差異，7層超過rtol1e-6/atol1e-7。非權重狀態與EMA完全相同。將回饋迴圈從完整圖重建縮到1秒左右的逐層投影後，檢查GPU無compute程序，再以同GPU投影：148層全部逐位元相同。故為CPU/GPU投影数值路徑差異，不能誤判成checkpoint血緣錯誤，也不能直接寬鬆容差略過。

正式preflight新增明確 `--projection-device cuda:0` 選項，使用真實EMA尺度重放；原export不修改。GPU僅用小型投影診斷，沒有啟動訓練。完整export前向parity仍需等待當次preflight完成，以狀態檔為準。一般sandbox helper失敗時改用獲准的限定apply_patch。

## 未解事項與風險

依使用者補充，新增PTQ→QAT恢復判讀報告：明確說明PTQ是未針對替換格式適應訓練的結果，QAT可能恢復但不保證。Detect predictor SD4列為待單層probe及預算審查的恢復候選，不把PTQ gate的reject當作永久淘汰。驗證為核對40格dual-summary原始delta與文件一致；困難：無。現有單層probe不停止，也沒有因此更改精度門檻或宣稱額外QAT已完成。

監測第一次error事件後的修復：parent GPU評估已完成，但舊 `_validate_role` 只回傳8項mAP50摘要，新驗收取mAP50–95時KeyError。原始metrics.json保有40項數值，16項要求指標與V36 parent逐項差異全為0。新增hash-pinned雙指標讀取helper；PTQ保留8指標歷史gate，另外保存all_search_metrics供16指標總gate。parent重驗使用明確recover-completed-report入口重用本次已觀測完成產物，未再次做GPU評估。也修正新parent的matched_metrics須以8鍵傳入舊gate介面，避免混用40鍵原始字典。

最後一輪修復前相關整合回歸45項通過（72.83秒）；新增雙指標介面測試後另重跑相關子集合，精確結果以工具輸出與後續紀錄為準，不混加測試數。

後續實測更新：同GPU EMA逐層重放及CPU Detect/Pose固定64x64前向parity均通過；40格矩陣已由真實parser驗證；ATen精度追蹤完成136筆operator/dtype紀錄，明列仍為浮點算子執行的fake-quant研究圖，不宣稱全整數部署。新增supervisor測試2項通過，確認GPU被他人使用時600秒等待、前置失敗不啟動、40格完成轉decision_required。

已啟動第一批supervisor：先做一次export搜尋精度重驗（非baseline訓練），16指標總gate及相對原parent的1e-4再現容差通過，才自動接40格特殊格式PTQ。新QAT尚未啟動，後續必須依本批結果選擇5epoch候選。低Token監測入口：`PYTHONPATH=src python -m yolo_quantize.blocking_monitor artifacts/queues/full-model-continuous-0907/execution-status.json`；status起始terminal立即返回，無變化只在程序內每600秒等待。

目前尚缺148路各特殊格式的完整單層probe矩陣、實際5epoch候選日程與QAT生成接線、後續累積／uniform及formal／實際整數部署。不得把第一批queue已啟動解釋為全部四天工作已完成。

全算子functional precision trace、40格矩陣實際schema檢查／GPU執行、完整5epoch QAT配置與持續queue接線尚未全完成；monitor已有程式與測試，不代表訓練監測已啟動。沒有新正式訓練、資料重切、停止他人GPU或Git上傳。工作進度從planned變implementation_in_progress_preflight。
