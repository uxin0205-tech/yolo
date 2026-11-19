# KD 訓練：五輪完成

`result.json`／`metrics-00.json`～`metrics-05.json` 保存完整每輪八項 AP。`resume.pt` 是第五輪 live／EMA、optimizer、scaler 的唯一完整保存，不能當成普通 inference checkpoint。`epoch-00.pt` 是原 D1 的 KD lineage 回退點，並非已訓 KD 候選。正式選型未採用 KD，原 D1 仍在 result/stage3。checkpoint 不提交 Git；來源與小型結果可提交。
