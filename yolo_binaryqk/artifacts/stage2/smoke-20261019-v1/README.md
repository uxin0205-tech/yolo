# 第 2 階段 GPU 短測

S1／S2 均完成：每組 512 張訓練、兩個 macro steps，評估僅 16 張。這是前向、梯度、實際參數更新、固定 BN、EMA、checkpoint 及 resume 的驗證，不是正式 COCO AP，也不是正式模型 parent。

兩組各十項必要參數均實際更新，AMP retries 為 0，峰值 allocated memory 均為 6240560128 bytes。兩組首批影像一致；已視覺檢查 `S2/train-targets.jpg` 的影像／bbox 對齊。

短測發生在 B2 搬到 `result/stage1/` 前；其 recipe 保存當時 reference-review SHA。後續只更新 parent 位置，權重及方法沒有改變，短測不續跑；正式候選重新從搬移後的原始 B2 開始。小型驗證可提交，短測 checkpoint／resume 不提交。
