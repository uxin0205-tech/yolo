# Stage9 進度查核：2026-11-11

後續狀態：本階段已於今天 06:25 完成全部 100 輪與交付，詳見 [完成紀錄](2026-11-11-actshape-stage9-results.md)。以下保留 00:46 的歷史進度。

查核時間：2026-11-11T00:46:15.655478+08:00。使用者詢問目前狀況；本次核對實際程序、GPU、監測、完整 epoch AP 與最新 checkpoint，未修改配方、來源鎖或資料，未增加 GPU 工作。

## 執行與驗證

前四組各 20E 已完成；最後一組 512×768 已完成第一段 8E，目前第 9/10 輪、step 380/463。總計完成 88/100 個完整 epoch。第一段結束後仍有半 LR 10E、完整重載驗證與交接；整個 Stage9 尚未完成。

supervisor 45516、補充校準 supervisor 45438 均持續存在約 14 小時，worker 58614 正常。當前 attempts：starts=1、SIGSEGV retries=0、resource interruptions=0，無新增中斷。GPU 即時取樣 75%、57°C、總記憶體5640 MiB；最近背景 monitor 約 212 秒前，符合每600秒政策。三組梯度都存在、AMP overflow retries=0。

最新 resume、config、source 與最佳 checkpoint 的實際 SHA 均相符，next_epoch=9。最佳 checkpoint SHA `a7526f060612fdada757679b6b32d408b760d4c53f99e6adcb9923f16636b457`。只做與新 checkpoint 相關的完整性核對，未重跑已通過的 smoke／單元或 GPU 評估。

## 精度

綜合最佳仍為第一段 E2：COCO AP 51.9632%、person AP 64.5865%、COCO 球棒 AP 45.3351%、BBAT box AP 62.4501%、Pose AP 89.9026%。這些數值全部來自同一份 checkpoint。

最新完成的 E8 COCO AP 51.9729% 略高，但 BBAT box AP 61.5455%、Pose AP 89.3092% 較差，未超過 E2 綜合選型。原 qSiLU Rep17＋20 主線繼續保留；目前候選的球棒／Pose 仍有退化，不因單項 COCO 小幅增加就更換交接模型。

## 預估與未解事項

近期每個完整 epoch 約 30.02 分鐘，第一段估計 2026-11-11T01:22:46.759434+08:00 完成，半 LR 段與驗證估計 2026-11-11T06:26:57.103350+08:00；對外仍估計今天 06:30～07:00，假設無新中斷且後段同速，最後交接整理時間可能浮動。背景 queue 持續，全部 Stage9 完成後停止。

困難與解法：無新增困難。先前 CUDA 原生崩潰根因仍未確認，穩定執行不等於根因修復。此輪只核對既有實驗，未新增硬體綜合／板端量測。小型查核資料：`yolo_actshape/artifacts/status-20261111-0045.json`。
