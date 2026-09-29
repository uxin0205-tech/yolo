# 2026-09-09：融合前單點 RepConv 對照

## 變更與原因

HOG 與原生完整 Neck＋head 階段均未改善 overall，head BN 統計恢復也沒有幫助。因此本輪 RepConv 不沿用退化的完整 Neck 配方，將兩組可訓練範圍縮到同一個 layer 17，檢查參數化本身是否有用。這是原方向 1 的單點實驗，不擴到 layer 20、不疊加 HOG。

共同起點是 late E8 EMA，SHA256 `27c09c6685a003e9ff106898fd898199e184ee195cce643dd668079abadb9342`，仍為探索 parent，不是正式 winner。A0、B100、融合後來源與全部歷史結果保留。

## 架構與訓練範圍

```text
原生：layer16 P3 → layer17：3×3 Conv + BN → P4／P5

候選：layer16 P3 → layer17 ┬─ 3×3 Conv + BN ─┐
                          └─ 1×1 Conv + BN ─┤→ 相加 → 原 activation → P4／P5
                                           │
部署：上述兩支折回原格式的一個 3×3 Conv + BN ┘
```

stride 為 2，沒有 identity branch。原 3×3 支完整載入；新 1×1 支的 BN gamma／bias 為零，插入時輸出應逐值一致。新支權重的首步梯度為零是合理現象，必須確認 BN gamma 梯度非零、可打開分支，不誤判為斷梯度。

兩組都只訓練 layer 17 的參數，head、Backbone、其他 Neck 與 attention 參數全部固定，所有 BN running statistics 固定。原始 P3 特徵路徑與 head 權重不改；但最後全尺度 top-k 的選擇仍可能受 P4／P5 變動影響，不能稱所有最終 P3 偵測完全不變。

## 超參數

兩組各 5 epochs、同一個 10-epoch criterion／cosine horizon，warmup 1、AdamW LR `2e-6`、betas `(0.948, 0.999)`、eps `1e-8`、weight decay `0.00027`、clip `10`。新 optimizer 階段，不假稱無中斷 resume。physical batch `32`、累積 `4`、logical `128`、AMP FP16、完整 COCO train118287／val5000。

EMA 從共同起點初始化、age `7400`，沿 ModelEMA 原 decay／算術，只更新可訓練 layer 17 參數；固定 state 不反覆做浮點平均，live 與 EMA 皆逐 epoch 核對固定 state 不漂移。此共用策略讓兩組的固定 P3 路徑維持一致，不把未訓練參數的微小 EMA 舍入算成方法收益。

主要 gate 仍為 overall／person；相對起點任一項下降超過 `0.005` 停止分析。ball／bat 數據保留；使用者明確要求的 HOG／MASF 單類觀察不自動變成 RepConv 硬性 gate。

## 前置驗證與實作

`rep17.py` 只重用現成 `from_conv`／`to_eval_conv`，不呼叫融合後 runner；匯入後恢復 `sys.path`，避免共用模組的 runtime 匯入改變融合前來源優先序。原來源程式不修改。新增單點 trainer、preflight 與串行 queue，並擴充本研究匯出入口，將 RepConv 折回標準 Conv。

啟動正式訓練前依序檢查：實際整圖 CPU160 的零分支初始化逐值一致；折疊後輸出容差 `atol=1e-3, rtol=1e-4`；推論 state keys 與原圖相同；折疊初始模型完整 COCO5000 的四項 AP 對起點各差不超過 `1e-4`。以上為事前工程容差，不是精度驗收門檻。

接著兩組各用真實 128 張 macro 校準，檢查同資料 trace、更新前 loss 完全一致、新支 BN gamma 梯度、首步相對更新在 `(0, 0.001)`、live／EMA 固定 state，以及匯出重載 CPU160 一致。smoke 更新丟棄，通過後才正式各 5 epochs；每回合只用折疊後的 BitTrue 圖評分。

## 困難與未解事項

已知共用 RepConv 模組有 runtime 的路徑副作用，採局部匯入並還原搜尋路徑，不更動融合後流程。其餘新增執行困難：尚未執行，不能預先稱無。前置與 GPU 更新驗證、AP 結果皆待完成，不保證單點重參數化有效；無收益則停止，不全面替换 Conv。MASF 後續仍加看 ball／bat。

## 前置完成與正式啟動

新入口 AST 與既有 RepConv 2 項 CPU 測試通過。2026-09-09 17:46（Asia/Taipei）實際整圖初始化逐值相同；折疊 CPU160 最大絕對差約 `0.000110626`，在事前容差內；折疊初始模型完整 COCO5000 的 overall／person／ball／bat 對共同起點差值均為 0，推論 state keys 相同。layer 17 訓練參數由 590336 增至 656384，部署折回原格式。

两組真實 128 張 macro 校準通過：資料 trace 同為 `a0dba80a7ab1f1b4c63cc04ae4d229fbb792a2f7d7f8f17f828fe8ec621057bc`，更新前 loss_sum 都是 `226.00542831420898`。原 Conv／RepConv 首步相對更新約 `7.74e-6`／`7.03e-6`；新支 BN gamma 有梯度；live／EMA 固定 state 檢查及匯出重載 CPU160 一致性通過。證據為 `artifacts/rep17-preflight-v1/summary.json` 與 `artifacts/rep17-smoke-proof-v1.json`。

17:46:35 啟動 `rep17-control-v1`，正常完成後 queue 接續 `rep17-rep-v1`；monitor session34916，每次 shell wait 最多 600 秒。新增執行困難：無。正式 AP 尚待完成，沒有把工程等價當成精度增益。
