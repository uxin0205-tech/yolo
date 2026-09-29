# 2026-09-13：scale／bias 硬體設計確認

變更與原因：使用者要求確認硬體友善性；核對實際 import 路徑、StaticDyadicScore、相對 bias、XNOR、Hadamard 與 PWL，新增 [硬體核對報告](<../../experiments/post_binary_rep_v1/SCALE-BIAS-HARDWARE.md>)。未改模型、GPU 排程或訓練來源。

驗證方式與結果：CPU import 確認來源為 achitechure_1/final/code；解析 E5 constants，16 個 scale、1008 個 bias，以 uint16/int16 保存共 2048 bytes。尺度範圍包含 1024，需至少 11-bit unsigned；score 小數位不能以單純丟棄 10 bits 代替。PWL 定點 exp 後仍浮點正規化，BinaryQK 是 bool sum 軟體參考，不是 packed kernel。未執行新的 GPU／精度測試，也未做目標硬體測量。

困難及解法：無執行錯誤；容易混淆數值定點與實際整數實作，報告分開表達且標出仍有浮點／展開表的位置。

未解事項／風險：完整 Attention 尚非純整數；匯出表、位寬溢位、捨入對齊、reciprocal 與目標板端成本待驗證。未推論 scale/bias 必然提速，不改正在訓練的來源 hash。
