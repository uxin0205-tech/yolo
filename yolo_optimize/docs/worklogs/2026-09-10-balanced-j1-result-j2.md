# 2026-09-10：校準 J1 結果與 J2 接續

## J1 正常完成

`balanced-j1-v2` 完成14 epoch／6482 macro，patience8停止，沒有觸發safety stop。最佳Pose是第6 epoch（zero-based5），SHA256 `7ef02ce42ab9dfec984f60a79cc3b52002e6630dd35665ecf7c0001269902c50`。

| 指標 | 最佳Pose候選 | 原独立baseline差值 |
| --- | ---: | ---: |
| COCO overall | .504954396 | -.003257559 |
| COCO person | .625437323 | -.002226804 |
| BBAT box | .600208253 | -.030755360 |
| BBAT pose | .888377841 | -.023782707 |
| ball box | .496680169 | -.014066731 |
| ball pose | .867074961 | -.009188119 |
| bat box | .703736338 | -.047443988 |
| bat pose | .909680721 | -.038377295 |

COCO已過0.005保護，但box／pose總體及bat未過0.02最終門檻，所以沒有`best_joint`，不能宣稱融合完成。相較merge J0 Pose AP .866087已改善；原独立完整Pose候選 .897997 仍保留。最後epoch14的COCO .506108／.625908，Pose .880952，未優於所選E6。

## J2來源與範圍

新增 `j2_stage.py` 與 `j2_train.py`，安全載入已完成J1 E6的全部trunk與兩heads，不重組回舊初始化。嚴格逐key載入、鎖定SHA，PWL[-10,0]20段與硬體固定狀態保留。fresh optimizer及新loss horizon，不稱exact resume。

J2按報告的後段解凍思路：backbone layer9+ LR1.5e-6、Neck7.5e-6、兩heads各2e-5；MASF／α保持J1的1e-6，不同時放大15倍。attention固定、shared與MASF BN統計固定、shared BN affine固定。AdamW其他參數與J1相同、warmup1。

最多40 epoch；plateau沿原final的patience17、recovery_after8、最多一次LR×0.5。這是明確折衷配置，不宣稱與PDF40／10或final80／17完全相同。

## J2重新校準與驗證

新增backbone9+使共享梯度scope改變，因此在固定J1 E6 checkpoint重跑16＋8 train-only macro，沒有使用validation。以Pose weight0.25測得校準中位數3.4821587，推導固定weight0.07，後8個確認中位數1.7683307，在0.5–2帶內。新參數只影響訓練，推論不添加動態控制。

`balanced-j2-smoke-v1`真實兩macro通過：layer9參數、MASFα確實更新，context／α梯度有限非零，固定參數／EMA／MASF BN／硬體契約通過。共同smoke改以實際stage與horizon建loss/scheduler，J1預設20不變。正式entry在run前實際序列化resolved config，避免重演super整合錯誤。

## 啟動與安全

使用者詢問進度時的唯讀快照：J2已保存17epoch／7871macro，程序PID480038仍為Rsl、執行約2小時51分；無ERROR/JOB_DONE事件。E17 COCO .504888904／person .626835486、box .601567855／pose .882493136；ball box／pose .485806805／.850675071，bat box／pose .717328905／.914311202。最佳Pose checkpoint仍為E6，box .599041609、pose .888960061；尚無best_joint。最近plateau stale_epochs11、should_stop0。此為使用者要求的單次進度檢查，未修改或中斷訓練，之後回到blocking monitor。

正式 `balanced-j2-v1` 已啟動，事件檔 `combine/artifacts/logs/balanced-j2-v1.events.jsonl`，600秒blocking monitor。最終COCO相對原Detect下降<=0.005、六項BBAT相對原独立Pose下降<=0.02；與訓練期暫時適應安全門檻分開。Pose災難性下降比較起點改為J1 E6，不是舊merge J0。完整Float／BitTrue每epoch驗證。

困難：J1已保住COCO但bat仍未恢復，可能需要更多共享特徵自由度；J2是待驗證推進，不預設一定改善。仍沒有最終可升格融合模型；activation／方向2未啟動。沒有刪除、覆寫來源、commit或push。
