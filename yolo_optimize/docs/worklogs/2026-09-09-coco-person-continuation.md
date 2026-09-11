# 2026-09-09：改以 COCO overall／person 決策並續訓

## 變更與原因

使用者明確表示本分支不處理融合後任務，ball／bat 不應主導實驗，主要目標是避免 COCO2017 精度下降，允許依趨勢多訓練。原 v1 中 ball／bat 是 COCO 類別 AP，不是 Pose；但其 gate 與本次優先順序不符，因此後續取消該 gate，只用 overall／person 作停止與選模。歷史指標與停止結果保留，不事後重寫為成功。

## E1 結果

兩臂均完成完整 COCO train118287、925 macros，沒有 runtime error。舊 bat gate 導致 E1 暫停；overall／person 都未達 -0.005 的安全停止幅度。

| 指標 | A0 | control E1 EMA | QK E1 EMA | QK−control |
| --- | ---: | ---: | ---: | ---: |
| overall AP | 0.506738574 | 0.506281422 | 0.506240041 | -0.000041381 |
| person AP | 0.626805274 | 0.625640127 | 0.625996009 | +0.000355882 |

差距太小，不能宣稱 QK 已增準。control 每 epoch 含驗證約495.7秒，QK約577.0秒；皆是真實完整資料訓練，不是 smoke。

## 續訓設計

另開 `continue_a0.py` 與 v2 產物，不修改已執行的 v1 程式。保留兩臂 E1 的 model、optimizer、scaler、EMA 和 age；使用相同的新 deterministic 資料流程從 E2 接續。這不是 exact uninterrupted dataloader replay，因為舊快照未保存 worker／prefetch 狀態，必須明記限制。

native E2E criterion 仍用10-epoch horizon，E1快照在 epoch validation／criterion.update 前保存，所以續訓建立相同 criterion 並前進一次。warmup 只佔原 E1，續訓不再重設；cosine沿原 global step，10epoch後若延長則保持最後LR，不回彈。

先續至共同 E3 再分析，最多20epoch只是可用上限，不代表已排滿。patience6看實際run內overall／person改善，而非要求每一回合先超過原parent才能續訓；正式驗收仍要求兩項對parent／control不退超過0.001，且至少一項改善0.001。任一主要指標相對parent下降超過0.005仍停止分析。ball／bat不作gate或追加診斷。

## 驗證與狀態

已新增成對 continuation smoke 與檢查入口：應確認相同128張增強影像／標註trace、optimizer step925→926、EMA925→926、criterion updates=1。正式v2必須等兩臂smoke及檢查通過才啟動。實際結果見 `artifacts/continuation-proof.json` 與 `a0-pair-v2-state.json`，本紀錄不把預期檢查當成已通過。

## 困難與解法

舊gate比使用者目前需求更廣，已明確版本化變更；不因其他COCO單類波動阻止主要任務恢復。舊snapshot缺worker狀態，採相同的epoch邊界新資料流程，保留optimizer／EMA而不宣稱exact replay。未修改原始資料、B100或融合後權重。

## 未解事項／風險

## 續訓驗證與啟動補記

初版 continuation smoke 接上 optimizer／EMA，但新 seed 未影響 native `build_dataloader` 的固定 generator，首 macro 重播 E1。未啟動正式 v2 即發現；保留初版 smoke，改用 native dataset／collate 配上明確 seeded DataLoader。未抽樣、未重切、未改標註。

修正後兩臂 smoke 已通過；首128張影像／標註 trace 都是 `709302ef5eb7ed9c03ca2487d2ed4042101be068aed5acd55ff17fdc74407c4d`，不同於 E1 的 `1d20be...`。optimizer step與EMA皆由925接到926，criterion updates=1。證據是 `artifacts/continuation-proof-v2.json`；較早 `continuation-proof.json` 只記錄舊 native replay，不作新流程證據。

2026-09-09 12:37（Asia/Taipei）啟動 `a0-control-recovery-v2`，同 queue 接續 `a0-qk-recovery-v2`；兩臂都從自己的完整 E1 開始，計畫終點為 E3。GPU blocking monitor session35788；正常只等待，每次最多600秒，沒有額外讀取進度 log。正式結果尚待工作退出後核對。

尚無新的驗收winner。繼續訓練可能改善也可能退化，仍依主要AP與推論驗證判斷。真實影片案例尚待使用者提供；無獨立場景改善或硬體效能宣稱。
