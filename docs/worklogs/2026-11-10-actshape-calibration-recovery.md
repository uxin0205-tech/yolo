# Stage9 降 LR 校準中斷與恢復：2026-11-10

查核時間：2026-11-10T10:41:16.384324+08:00。使用者詢問目前狀況，確認 09:56 擴充 queue 因梯度校準未過門檻退出；本次在已完成 50 個完整 epoch 的安全段落邊界診斷與恢復。沒有重新訓練完成的段落，總上限仍為 100E。

## 問題與事前參考

256×512 第一段 10E 已完成，最佳為 E6。接續 `low-lr-calibrate` 原設定為量測 3 macros、每次確認 2 macros、最多兩次。第一次確認 median 為 0.1523284651，第二次為 1.3350652297，均超出原 band [0.2,0.8]，因此主動停止。這次錯誤是 `ValueError: 任務梯度比例未通過確認`，沒有本次 CUDA SIGSEGV 證據。

動手前唯讀核對 optimize `experiments/combine/bridge_v1/calibrate_task_weight.py`、`j3_stage.py`、實際 J3 校準 log 與 `artifacts/j3-task-weight-calibration-v1/summary.json`。optimize 採固定 checkpoint、train-only 16 macros 量測＋8 macros 確認、median 梯度比例與逐 macro 還原模型；該 run weight=0.215、確認 median=0.9618735993，target=1、band=[0.5,2]。它是不同 parent／範圍，僅採量測方式；本輪原 target=0.5、band=[0.2,0.8]、bounds=[0.01,1]、step=0.01、attempts=2 均保持。

事前核對、來源 SHA、父模型、重現規格、預算見 `yolo_actshape/artifacts/calibration-recovery-20261110/review.json`。新政策在 `policy.json`，執行程式為 `yolo_actshape/operations/recover_calibration.py`。

## 重現與修復

以原始 worker、原 parent、原 seed 與配方，在獨立診斷輸出重跑 7 macros：

```bash
./yolo_actshape/run_extension.sh worker --spec /home/uxin/yolo/yolo_actshape/artifacts/calibration-recovery-20261110/reproduce/job.json
```

exit code=1，完全重現同一 `ValueError`；逐批次 records 與原失敗完全一致，parent SHA 相符、模型已還原。優先假設依序為少量確認受批次差異影響、權重步長／範圍限制，以及來源或模型改變。來源／parent 未變，增加量測後能在相同權重步長與 band 內通過，證據支持取樣變異是這次未通過的主要原因；不宣稱已證明所有批次穩定。

補充只改量測數為 16、每次確認數為 8，沿用原 `stage9.training.calibrate` 函數、原迭代修正與最多兩次確認。實際使用 32 個 probe macros；沒有 optimizer 權重更新、沒有用 validation 挑權重，parent 完整還原。量測 median=1.6308987348；weight=0.08 的第一次確認 median=1.0250433355 未過，原公式修正為 weight=0.04，第二次確認 median=0.2766813849 通過原門檻。

保留原失敗 `job.json`／`result.json`，新結果獨立保存；只有真實 GPU 校準通過後才新增原 job 的 `done.json`，其中明列新來源／新結果 SHA 及被接續的原證據。後續正式訓練 config 已攜帶完整 `calibration_recovery`，不是把原失敗覆寫成成功。原程式與 protocol 未改。

原擴充 source SHA：`d8212a644a473d3bdd0e9d1f15cd38c14e8601d48d3b62dfd379f1a8f3da168b`。

補充 source SHA：`bfe92ac19712f66ecf3bd3a7111ccbc362b9bf0267120edc4a0c60a1c6112ac1`。補充鎖包含自身程式、policy、事前 review、原兩份 source lock，以及唯讀 optimize 方法／結果 SHA。

## 有界背景恢復與驗證

背景補充 supervisor PID 45438 只在原 queue 因相同 `needs_review` 校準失敗且退出時處理。可用範圍僅為本次 256×512 low-LR 校準與後兩組尚未執行的四個校準；每 job 最多一次、每次最多 32 macros。其他錯誤、外部 GPU 衝突、STOP 或逾時均停止，不自動放寬門檻或無限重試。正式 queue 保持每 600 秒監測、每 30 秒資源檢查，全部 Stage9 完成後停止。

驗證：原失敗與重現結果均被拒絕，原 3＋2 的成功案例也不能冒充 16＋8 通過；實際新 GPU 結果通過。另驗證非有限梯度、未還原 parent、使用 validation、超出 band、確認數不足皆被拒絕。新舊來源鎖及 published result SHA 通過，見 `validation-before.json`／`validation-after.json`。未重跑既有 smoke 或 AP；正式 AP 依原排程每 epoch 執行。

## 最新進度與結果

約 10:38 恢復原 queue。supervisor PID 45516、worker PID 45521；目前為 256×512 降 LR 第 1/10 輪、step 60/463。EMA updates=60，三個梯度群均存在，AMP overflow retries=0。GPU：NVIDIA GeForce RTX 5090, 64 %, 2772 MiB, 46。目前 worker 首次啟動，沒有新增 SIGSEGV retry 或資源中斷。

已驗證此段以第一段 E6、SHA `61018101da486a724419bfcf5897bacc9e038423bd9159cc1c09867380324587` 出發；neck LR=9.5e-7、兩 head LR=2.5e-6，均為前段一半，預算 10E。配置確實帶入本次確認的 Pose weight=0.04 與校準來源。硬體友善部署契約、資料、可訓範圍、BN／EMA 均沿用原定義。

當前最佳 AP50–95 仍來自第一段 E6：COCO 44.8663%、person 56.1822%、BBAT box 55.3245%、Pose 85.9081%。降 LR 段尚無完整 epoch AP。原主線 qSiLU Rep17＋20 仍保留；384×512、512×768 後續各 20E，既有七組 GPU replay 已通過。

256×512 依先前每輪約 18.68 分鐘，估計 2026-11-10T13:47:33.002568+08:00 完成，對外約今天 14:00；整體仍粗估 11 月 11 日。後兩組實際速度與新中斷可能改變時間。

## 困難與未解事項

困難：原少量校準未通過；已增加 train-only 量測並通過原門檻。建鎖時首次引用了不存在的 optimize `artifacts/fusion/` 摘要路徑，CPU check 明確失敗，已改為實際 `artifacts/j3-task-weight-calibration-v1/summary.json`，未啟動 GPU 或修改 optimize 以規避。

未解事項：這次恢復不代表先前 CUDA 原生崩潰根因已修復。精度尚未恢復至原主線；仍須等待本段及後兩組完整 AP。硬體 replay 不代表 RTL 或板端實測加速。小型即時查核：`yolo_actshape/artifacts/status-20261110-1040.json`。
