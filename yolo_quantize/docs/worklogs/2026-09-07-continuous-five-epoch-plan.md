# 持續5epoch實驗規劃修訂

使用者後續澄清：四天仍為軟限制，不是取消時程。新增soft_target_hours=96、從實作開始計時、最後8小時保留驗證／報告與必要延長說明；每組5epochs及精度gate不變。JSON解析與96小時／5epoch／600秒不變量檢查通過。困難：無。新queue仍未啟動。

變更與原因：依使用者新要求，每組新QAT改5epochs，取消96小時截止；保留初批最多6組，再每批最多2組按證據追加。明列全模型precision、雙總gate、AdamW選擇依據與blocking monitor契約。既有V36完成結果不改寫。

驗證：讀取V36 queue為complete/error=null；檢查新的機讀JSON可解析，5epochs、600秒、148權重路徑、16指標、無硬截止及planned_not_enqueued狀態一致。這是規劃驗證，不是執行器測試、QAT或GPU驗證。

困難與解法：舊四天計畫未實作共同parent及live queue，不能只改epochs就宣稱開始訓練。新增獨立版本並標註取代範圍、要求5epoch scheduler preflight。其餘困難：無。

未解事項：P0 parent/export/全算子盤點、5epoch日程實作與queue接線尚待完成；本次不啟動GPU、不宣稱monitor已啟用、不自動上傳GitHub。
