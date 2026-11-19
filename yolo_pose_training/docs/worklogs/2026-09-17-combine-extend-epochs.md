# 2026-09-17 combine 配方增加訓練輪數

變更與原因：使用者要求更多 epoch。P1／P2／P3 上限由17／22／100改為40／60／200，patience由10／12／20改為20／25／40。640、batch32、nbs128、MuSGD lr0=0.00038等設定不變。

保留進度：P1已完成6輪，完整optimizer checkpoint與results.csv已備份至artifacts/combine_recipe/before-epoch-extension-20260917-125537。停止本實驗程序後載入last-resumable.pt接續，未完成的部分epoch會重做。

實作：官方8.4.90的check_resume不允許epochs覆寫，因此以輕量PoseTrainer子類在官方check_resume後套用明確傳入的較大epoch上限；不修改套件、不編輯checkpoint。optimizer／EMA／scaler仍由官方resume還原。cosine與原生loss schedule依新總輪數計算，與原17輪排程不中斷跑完不逐值等價。

驗證：三份YAML通過get_cfg，腳本語法檢查通過；備份checkpoint具有optimizer狀態。GPU接續檢查見下方。

困難與解法：官方resume忽略epochs覆寫，改用上述子類明確支援延長；其他困難：無。

未解事項：最多300輪，不代表一定全部跑完；仍依patience提前停止。更多輪数不保證精度提升，最終依固定驗證比較選模。背景session為bbat5_combine_long_20260917，狀態仍為artifacts/combine_recipe/status.json。

實際接續驗證：args.yaml確認epochs40、patience20與resume路徑；已完成第7輪，新輪次所有CSV數值為有限值，GPU訓練持續。

使用者再次要求開始時確認既有佇列仍運作，未重複啟動。P1／P2均已有completed標記，實跑輪數{"combine_recipe_p1_640": "40", "combine_recipe_p2_640": "60"}；P3已完成13輪、正在14/200，GPU使用率100%、19329 MiB。困難：無；最終結果待完成。

精度查詢：P3已完成25輪，綜合最佳為第19輪；640訓練中驗證Box mAP50／mAP50–95=0.88488／0.60021，Pose=0.90699／0.90331。P1／P2各自綜合最佳Box／Pose mAP50–95=0.23237／0.56872、0.59054／0.89480。同為640的舊Stage2訓練中最佳=0.62314／0.91343，新方案目前尚未超越。尚未完成最終固定batch8的640／960重驗，此次未調整訓練。困難：無。

完成精度查詢：2026-09-17 14:34:58 完成訓練與統一評估。P3第37輪最佳，第77輪因40輪未改善停止。640／batch8評估新P3的Box／Pose mAP50–95=0.634452／0.917011，舊Stage2同設定=0.62348／0.91343，新方案提高約1.10／0.36個百分點。新P3於960為0.61408／0.90788，960綜合仍由舊Stage4最高。新P3的640 ball Box／Pose=0.539151／0.894054，bat=0.729754／0.939968。已核對selection.json與report.md；本次未啟動新訓練或替換權重。困難：無；限制仍為單seed與formal val選模。
