# MASF E5 結果與 P3 head 配對適應續訓

## 已完成結果

UTC 2026-09-09 15:47:33，session 70915 收到 fork JOB_DONE／exit 0 與三組 ALL_DONE。control／shared／fork 均完成相同 5 epochs，每回合完整 118287 張與 925 次更新；未重跑、未覆寫。

| Epoch | control overall | shared overall | fork overall | control person | shared person | fork person |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.508156300 | 0.508242593 | 0.508132190 | 0.627462316 | 0.627397040 | 0.627398014 |
| 2 | 0.507948227 | 0.507817290 | 0.507943458 | 0.627612181 | 0.627608987 | 0.627654329 |
| 3 | 0.507981396 | 0.507760420 | 0.508004005 | 0.627582779 | 0.627658107 | 0.627549811 |
| 4 | 0.508055082 | 0.507750883 | 0.508071314 | 0.627485543 | 0.627558638 | 0.627589170 |
| 5 | 0.507973760 | 0.507614304 | 0.507963067 | 0.627688255 | 0.627747652 | 0.627627189 |

都是 EMA、完整 COCO internal AP50–95。Parent overall／person=0.508153635／0.627520517。fork 在 E2–E5 的 overall 比 shared 好，但未取得相對 control 的 +0.001 主要改善；不能宣布移位已成功增準，也不能推論 MASF 在所有訓練範圍下一律無效。

fork E4 相對同回合 control 的 overall／person／ball／bat 差值為 +0.000016232／+0.000103627／+0.000071945／+0.000204228；方向一致但幅度不足。E5 則為 −0.000010694／−0.000061066／−0.000884382／+0.000230315，不穩定。所有候選保留，尚未升格 winner。

## 最小診斷與推導界線

只彙整完成的 `progress.jsonl` 每 epoch 平均 loss／AMP 重試，再以 CPU 讀 E5 EMA 與初始化參數比較，没有讀進行中 log 或擴大 GPU 檢查。

- 三組 native batch-sum loss 的平均值大約從 213.44 增至 230.79，但實際 `criterion.update()` 每回合會改變 one-to-many／one-to-one 權重；資料增強也改變。因此跨回合數值不能直接判定發散。相同 epoch 三組 loss 差異很小，亦沒有證明 MASF 已改善泛化。
- 三組 E5 各遇到 1 次 AMP retry，現有縮放重試後正常完成；沒有未恢復的浮點錯誤，不重跑。
- E5 EMA 的 P3 head 相對 L2 更新：control=0.000421843、shared=0.000422549、fork=0.000421155，即約 0.042%；44 個 parameter tensors 都有更新。
- shared／fork context 相對 L2 更新分別為 0.011643688／0.009696072，約 1.16%／0.97%，9 個 tensors 都有更新。
- shared／fork E5 EMA alpha=0.020212874／0.016123101，沒有停在 0。

以上排除了「完全沒更新」，但更新範數小不等於已證明欠訓練，參數範數與輸出影響也不能直接比較。可驗證的下一個假設是：原 P3 head LR=2e-6 過於保守，限制它適應新特徵。先只改這一項，不把失敗直接歸因於位置或增加所有模組。

## 接續 E6–E10 的設定

新增 `continue_masf_head.py` 與 `run_masf_head.py`，不修改原始 `train_masf_p3.py`。

- control 與 fork 各從自己的 E5 live／optimizer／EMA／scaler 續訓，前 5 epochs 已為同起點、同配方差一個 MASF 的配對歷史。兩組 E6 起不要求 loss 精確相同，因模型已學到不同權重；仍要求配對資料 trace 相同。
- 唯一配方變更：P3 head base LR 從 2e-6 提高至 1e-5（5 倍）。MASF context=1e-5、alpha=1e-4 不變，仍為 AdamW、相同 scope、全部 BN running statistics 固定，拓撲不變。
- 總預算延長至 E10；原 10-epoch cosine 與 native loss horizon 不變，不重新 warmup。E6 初始 cosine 係數 0.75，因此實際 P3 head LR=7.5e-6。
- 保留 logical128=32×4、imgsz640、完整 COCO80。E6 起使用共同新資料流 seed20260927，不宣稱重現原 DataLoader 未中斷時的完整順序。
- E5 optimizer steps=4625、EMA updates=12025。舊快照在 epoch 末 criterion.update() 前保存，因此 saved criterion age=4；E6 需重建至 age=5，不能照抄成 4 或重設 0。
- 沿用 overall／person 安全門檻與同預算驗收，ball／bat 繼續觀察。shared 的主要指標趨勢沒有支持直接加訓，暫不延長；不拿 shared E5 與其他 E10 宣稱同預算優劣。

不可變來源：

| Arm | E5 SHA256 |
| --- | --- |
| control | `432832f3c4d410e97ebd96f3d12a9fdc6a7b421b6a0747ae1f70065462fc2b67` |
| fork | `132c74576946cdaa4ef606dfddaa2b56fddf219a0987b9ddfe12ca051d40c33e` |

## 驗證方式與目前狀態

兩個新程式 AST 通過。正式訓練前各做 128 張／1 次更新 smoke：先檢查 live／EMA state 精確恢復、optimizer 全部 age4625、scaler 與 criterion 排程；更新後需 optimizer age4626、EMA age12026、criterion age5、P3 head／context 更新比例在 (0,0.001)，固定 state 在 live／EMA 中均保持不變。fork context 已有非零 gate，這次梯度應非零。

再檢查兩臂新資料 trace 相同且不同於 E1 舊 trace，並做 Bit-True 推論匯出 CPU160 精確重載。所有 smoke 通過才自動串行 E6–E10；GPU 結果尚待完成，不預先宣稱改善。

## 困難與風險

### 2026-09-10：Control E10 完成

UTC 2026-09-09 16:37:57（Asia/Taipei 2026-09-10 00:37:57），control 正常完成 E6–E10，每回合 118287 張／925 次更新。最佳 EMA E8 overall／person=0.508267092／0.627699455；E10=0.508172720／0.627605819。相對原 late E8 parent 只是小幅改善，未達 +0.001 門檻。已核對完成 summary；queue 自動接續 fork，沒有重跑或修改 control，亦未讀 fork 執行中 log。困難：無。MASF 額外收益仍待配對 fork 結果。

### 實際續訓驗證補記

兩個 smoke 均正常完成，optimizer／EMA／criterion 年齡為 4626／12026／5，live／EMA 初始 state 精確恢復，固定 state 與 Bit-True 匯出 CPU160 重載通過。新配對資料 trace 同為 `e2adc77c5e398c4213900fcfca1eb868981e081dfeaac27dc8e8260e3f530299`，與 E1 不同。

control／fork 的 P3 head 首更新相對比例為 4.73385e-6／4.71598e-6；fork context 更新比例 1.02036e-5、梯度範數 0.150393，證實續訓時 context 確有任務梯度。alpha 的相對更新約 0.000749，沒有碰到 clamp 邊界。以上均為 smoke，不作 AP 結論。

正式 control E6–E10 於 UTC 2026-09-09 15:59:07（Asia/Taipei 23:59:07）開始，session 62736，接續工作跨入 2026-09-10。Queue 後續自動接 fork；兩者均從原 E5 正式快照開始，沒有拿 smoke 當續訓來源。啟動與驗證困難：無。

困難：前一配方未取得足夠增準；已保留負面結果，從實際更新幅度建立下一個單變量假設。實作中特別處理 criterion snapshot 的 off-by-one 邊界，避免續訓排程重設。其他困難：無。提高 LR 仍可能無效或退化，應以同回合完整驗證判斷；不將使用者允許加訓當作可以無限延長的證據。
