# 2026-09-17 combine 式官方 Pose 三階段訓練

## 變更與原因

使用者要求參照 yolo_combine 獨立 Pose 訓練，允許加入判斷。此次正式啟動一組獨立新實驗，從官方 weights/yolo26m-pose.pt 開始；沿用固定 bbat5-v1 formal train／val，不抽樣、重切或變更標註。配置位於 configs/combine_recipe/，輸出 runs/combine_recipe_p{1,2,3}_640/。

| 階段 | 範圍 | 最大epochs | patience | 解析度 | batch／nbs |
| --- | --- | --- | --- | --- | --- |
| P1 | head-only，freeze23 | 17 | 10 | 640 | 32／128 |
| P2 | neck＋head，freeze11 | 22 | 12 | 640 | 32／128 |
| P3 | 全部解凍 | 100 | 20 | 640 | 32／128 |

每階段从前階段best.pt建立新optimizer／scheduler，P1使用官方權重。全程MuSGD、lr0=0.00038、lrf=0.5、momentum=0.948、WD=0.00027、cosine、warmup3、mosaic0、degrees0、translate0.1、scale0.5、無翻轉。loss權重延續原設定。

加入的判斷：physical batch固定32，暖身後約accumulate4，nbs128保持weight decay縮放為1；不宣稱等同physical128的BN。warmup_bias_lr保留0，沒有照抄combine的0.1。暫不追加960精修，先以640／960兩種解析度統一比較各best.pt，再看是否值得繼續。

參考來源為 /home/uxin/yolo/yolo_combine/src/yolo_combine/pose_stages.py 與已查證的實際run args.yaml。沿用官方架構，不加入Full35 MASF／修改attention，因此是方法移植，不是復現Full35。前次低LR單因素候選未改善，不代表整套recipe已被否定；本次多項共同改變，無法將改善歸因單一項。

## 排程與產物

scripts/run_combine_recipe.py循序啟動3階段，每600秒記錄GPU，每5秒檢查程序；共用queue.lock防重複。tmux：bbat5_combine_recipe_20260917。狀態：artifacts/combine_recipe/status.json；主日誌：logs/combine_recipe_queue.log。各階段保存best、last、last-resumable及週期checkpoint。

完成後評估新三階段best.pt的640／960結果，並補上舊Stage2／3／4的640對照；舊960引用既有相同batch8評估。各解析度分開選模，comparison.json、selection.json、report.md輸出在artifacts/combine_recipe/。既有最終權重與候選結果皆保留。

## 驗證方式與結果

三份YAML通過官方get_cfg驗證；官方模型有24層，Pose26 head位於index23，freeze23與freeze11範圍核對完成。初始化權重SHA256與canonical來源寫入provenance.json。新排程脚本語法檢查通過。GPU啟動前空閒。

## 困難與解法

無新增阻礙。沿用已驗證的獨立程序、checkpoint保存與OOM停止機制，避開先前960／batch32的顯存問題。

## 未解事項與風險

完整收斂、P2/P3顯存與最終比較仍待確認。頭部凍結策略與低LR可能使新head適應較慢，但沿用已參考的17輪上限及patience。單seed、formal val反覆選模；沒有獨立test。背景記錄與停止不代表代理交付後仍在线分析調參。沒有啟用子代理。

## 啟動驗證

12:52:49 啟動P1，已完成2輪且進入下一輪。實際args.yaml確認freeze23、batch32、nbs128、lr0=0.00038、warmup_bias_lr=0、resume=False；已完成輪次的所有CSV數值為有限值。best.pt、last.pt、last-resumable.pt均已產生。GPU使用率64%，nvidia-smi顯存13689 MiB，訓練日誌約12 GB，目前未出現OOM。P2/P3及比較程序已排妥。
