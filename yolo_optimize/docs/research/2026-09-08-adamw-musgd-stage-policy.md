# AdamW→MuSGD 策略

## Ultralytics YOLO26

來源：[docs](https://docs.ultralytics.com/modes/train)、[trainer](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/engine/trainer.py)、[muon](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/optim/muon.py)；2026-09-08 查閱 live/main，8.4.90 待另核。

支持：docs 稱 MuSGD 是 SGD+Muon 正交化，適合長訓／大資料；`auto` 以 iterations>10000 選 MuSGD，否則 AdamW。main auto 設 MuSGD `lr=.01,m=.9`，AdamW `lr_fit=.002*5/(4+nc),m=.9`，且忽略使用者 `lr0/momentum`。這是二選一長度規則，非 AdamW→MuSGD 證據。

分流：main 對 `param.ndim in {2,4}` 用 Muon；bias、normalization/BN 不衰減，其餘衰減。類別預設 `muon=.5,sgd=.5`，main 覆寫 `(0.2,1.0)`；Muon 組為正交化+SGD momentum，其他組純 SGD，部分 `cv3/one2one_cv3` head lr×3；hybrid 衰減只作用 SGD。

限制：BN 僅是參數分流的 BatchNorm；「BN 較大」若指架構／接續改動，不是 batch size 或 BN 觸發。官方無架構切換依據。

## Muon 作者實作（2024）

來源／年：[KellerJordan/Muon](https://github.com/KellerJordan/muon)，README citation 2024。支持：hidden 2D 用 Muon、embedding/head/gain/bias 用 AdamW；例示 Muon lr=.02、AdamW lr=3e-4、betas=(.9,.95)、wd=.01，且 lr/wd 要調。這是同一步分流（MuonWithAuxAdam），非階段切換；本次未找到獨立 MuSGD 論文或 staged recipe。

不可外推：該例為 hidden-layer／LLM，lr、分流與收益不可直接搬到 YOLO26 Detect+Pose，也不能證明大架構適應期必須 AdamW。

## PyTorch AdamW／切換狀態

來源／版本：[AdamW stable API](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)，stable 頁導向 2.14。支持：AdamW 解耦 weight decay，預設 `lr=1e-3`、`betas=(.9,.999)`，state=`exp_avg/exp_avg_sq`；MuSGD state=`momentum_buffer`，hybrid 另有 `momentum_buffer_SGD`，無直接對應。

切換限制：保留模型權重，重建 MuSGD、初始化 momentum，不載 AdamW state。Ultralytics `resume=True` 會恢復 optimizer/scheduler/epoch；切換須避免誤恢復，重定 scheduler/warmup，並先建 scheduler 再載 state。

## 對提案的可驗證邊界

官方僅支持長訓可選 MuSGD／短訓 auto 選 AdamW，沒有支持已訓練模型先 AdamW 適應、長訓必換；也非禁止 MuSGD。lr 須分調（main .01 對 AdamW `lr_fit`；作者 .02 對 3e-4）。實驗建議而非收益結論：先固定口徑比較候選 BEST、選 parent，再同 parent／seed／資料／總 step 比較 AdamW-only、MuSGD-only、固定切換 step；記錄 lr、warmup、scheduler、state reset。不可用 batch size／BatchNorm 解釋「BN 較大」。

## 子任務 worklog

- 變更原因：核對已訓練 YOLO26M Detect+Pose 的階段策略與切換風險。
- 驗證／結果：查閱上述一手來源，確認混合權重、分流、lr、state、auto 閾值；未執行 GPU、torch、模型、資料或 checkpoint 操作。
- 困難／解法：MuSGD 以 Ultralytics 實作為主，並列 live／`main` 差異；區分架構改動、batch size、BatchNorm。
- 未解風險：本地 8.4.90、staged warmup、實際收益待主代理於候選 parent 公平 A/B；本文不代決排名。
