# v2 GPU 功能短測

三個初篩分支及 B3 全模型恢復各觀測 512 張、8 次 optimizer step，全數通過梯度、固定狀態、重載、optimizer／scaler resume 及 phase 交接。不是完整 epoch，不是正式 AP。

結果與 recipe 保留供查核；驗證後的可重建臨時 checkpoint 已直接清除，沒有封存，不作正式 parent。需重做時用新的短測輸出目錄。摘要見 [recovery-v2-smoke-summary.json](../recovery-v2-smoke-summary.json)。
