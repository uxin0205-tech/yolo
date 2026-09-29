# scale／bias 硬體友善性核對

## 結論

目前是「固定尺度／相對位置小表、可映射定點的算法與軟體參考」，不是完整純整數 Attention，也沒有 RTL／FPGA／ASIC 延遲或功耗的驗證。這次只核對並記錄，沒有修改模型、訓練設定或 GPU 排程。

## 固定 scale

實作在 [StaticDyadicScore](<../attention_recovery_v1/modeling.py>)：每 site 4 heads×2 basis，兩 site 共 16 個常數。float master 存在 checkpoint，前向係數 round(clamp(c,1/1024,1)×1024)/1024；訓練以 STE 傳梯度，eval 時刷新非 persistent 常數快取。没有逐圖片幅值均值、動態 selector 或推論校準。

兩個 binary dot 分別是 sign(Q/K)、sign(Hadamard(Q/K))，不是多跑一條浮點 QK 作推論補償。signed dot = 2×popcount(XNOR)−d。每個 query/key 的 score 分子是 m0×Z0+m1×Z1+bx[Δx]+by[Δy]，帶 10 個小數位。乘數固定，可映射固定乘加或 shift-add；例如 188×Z=(Z<<7)+(Z<<6)−(Z<<2)。並非所有係數只需一個 shift，也不保證比目標硬體現成 DSP 更省。

重要：/1024 表示小數點位置，不可把分子直接右移 10 位後只留整數，否則會丟掉全部小數精度。連接目前 PWL Q8.8 入口需要依其四捨五入規則轉換；對理想 Q10 分子 S，在 row-max 相減後，Q8 code 為 floor((S−rowmax(S))/4+0.5)，再照契約 clamp。此是格式推導，不是已執行全模型逐位驗證。需要檢查 signed 位移、捨入、飽和及中間累加溢位。

E5 scale 整數碼：layer10 為 [[173,188],[191,186],[206,214],[237,237]]；layer22 為 [[129,204],[233,242],[228,242],[192,220]]。由 [constants.json](<../post_masf_hardware_v1/artifacts/scale_bias/train/epochs/e5/constants.json>) 直接核對。範圍含 1024，最少須 11-bit unsigned，現有報告以 uint16 儲存估算，不誤稱完整 10-bit 乘數。

## 相對 bias

實作在 [DyadicRelativeBias](<../post_masf_hardware_v1/models.py>)：bx、by 是依相對位移索引的兩張表，每個 head 各自有值，不依影像內容重新產生。所有值量化為 signed16 整數／1024，範圍 [-32,32−1/1024]。

實際兩 site 各 504 個值，合計 1008×2=2016 bytes；scale 16×2=32 bytes，邏輯常數總計 2048 bytes。這不含 PWL 表、QKV、特徵 buffer、索引／查表 port，也不是 checkpoint／GPU tensor 實際大小。理論硬體可按 Δx/Δy 串流查表，避免保存完整 N×N bias；PyTorch 現行則仍做 clamp/round 並展開 N×N，所以「值固定」不等於軟體已無每次前向的量化工作。真正匯出需預先打包整數表。

signed16 是因舊試驗用約 ±2 的範圍會截斷既有 bias，甚至讓部分梯度為 0；最新 E5 的整數碼範圍 site10 [-5108,6665]、site22 [-4312,8464]。不再盲目縮位而破壞表示能力。相對位置 bias 隨 key 改變；全 row 相同常數會被 row-max 消去，沒有相同的表達效果。

## 訓練與推論分離

[ExactDotSurrogate](<../studies/pre-fusion-full35-b100/scripts/qk_challenger.py>) 只在反向用浮點代理點積傳回 Q/K；eval 不執行反向／optimizer／STE。這項梯度修復不增加推論的反向電路。Hadamard 在推論用加減蝶形且省略正規化，因正的統一倍率不改變 sign；但仍增加加減、暫存與資料移動，不是零成本。硬體若有限位寬，也須避免蝶形累加溢位。

## 已核對的軟體邊界

實際 import 來源為 `/home/uxin/yolo/yolo_achitechure/achitechure_1/final/code/yolo_attention/`，不是僅依另一份同名程式推論。

- [binary_basis.py](<../../../yolo_achitechure/achitechure_1/final/code/yolo_attention/binary_basis.py>)：XNOR/popcount reference 使用 bool 比較、張量展開與 sum，未做 packed-bit kernel；速度／記憶體不能直接當硬體 popcount 性能。
- [normalization.py](<../../../yolo_achitechure/achitechure_1/final/code/yolo_attention/normalization.py>)：BitTrue PWL 為 Q8.8 score、UQ1.15 endpoints，範圍 [-10,0]20 段；分段 exp 有定點參考，但最後 weights/sum(weights) 仍是軟體浮點除法。固定 PWL 表不是本次訓練參數。
- [attention.py](<../../../yolo_achitechure/achitechure_1/final/code/yolo_attention/attention.py>)：QKV 投影、V 加權與輸出投影仍在；不能把 BinaryQK 的局部簡化說成全 Attention 無乘法。
- [pwl_contract.py](<../studies/pre-fusion-full35-b100/scripts/pwl_contract.py>)：兩個 PWL site、[-10,0]、20 段、沒有可訓 PWL 參數。

## 尚待完成，而非本次擅自修改

要宣稱純定點部署，還需整數表與乘數匯出、累加位寬界定、Q10→Q8.8 捨入對齊、PWL reciprocal 方案、V 加權格式與端到端 reference 對比。接著量測目標硬體資源／頻率／latency／energy。這些可能改變精度，不能只靠目前 BitTrue PWL 結果視為全部完成；本次沒有變更正在訓練的版本。
