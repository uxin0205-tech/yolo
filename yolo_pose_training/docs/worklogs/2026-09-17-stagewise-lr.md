# 2026-09-17 逐階段降低學習率重新訓練

## 使用者要求與變更理由

使用者要求參考combine，讓每階段LR不同並重新實作。核對yolo_combine/src/yolo_combine/pose_stages.py，獨立Pose P1/P2/P3原本共用lr0=0.00038、lrf0.5；融合訓練才使用階段／參數群組LR與平台期降LR。先前本專案各階段內有cosine，但下一階段會重設到同樣0.00038。此次新增跨階段遞減基準LR，屬新候選，不宣稱照抄獨立Pose原配方或保證改善。

## 新三階段設定

| 階段 | 範圍 | epochs／patience | 基準lr0 | 目標基準終點 |
| --- | --- | --- | --- | --- |
| P1 | freeze23，head-only | 40／20 | 0.00038 | 0.00019 |
| P2 | freeze11，neck＋head | 60／25 | 0.00019 | 0.000095 |
| P3 | 全部解凍 | 200／40 | 0.000095 | 0.0000475 |

共同640、batch32、nbs128、MuSGD、momentum0.948、WD0.00027、cos_lr=True、lrf0.5、warmup3、warmup_bias_lr0、scale0.5、translate0.1、mosaic0；loss、seed0等其餘參數保持。每階段從前階段best.pt建立新optimizer與scheduler，只有P1從官方weights/yolo26m-pose.pt開始。各stage有自己的warmup；不宣稱跨stage optimizer時間連續。MuSGD部分參數群組使用3倍LR，表中為基準值，實際群組LR可看results.csv。提前停止時不一定到達目標终點。

configs/stagewise_lr/為獨立配置，原配方與所有checkpoint保留。新runs為stagewise_lr_p{1,2,3}_640。沒有將已完成P3直接續訓當成重新開始。

## 執行與比較

新增scripts/run_stagewise_lr.py，共用GPU queue.lock，三階段循序執行；每600秒監測GPU，每5秒確認退出。tmux=bbat5_stagewise_lr_20260917；狀態artifacts/stagewise_lr/status.json；日誌logs/stagewise_lr_queue.log。

完成後新三階段best各以640／960、batch8評估，與已保存歷史相同設定結果比較，輸出comparison.json、selection.json、report.md。平均AP改善且ball／bat各自Box／Pose AP退步不超過0.005才產生替換建議，不自動修改舊final。

資料仍為bbat5-v1，formal train5964／val683，registry=/home/uxin/yolo/configs/datasets/bbat5-v1.yaml，canonical YAML=/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml；assignment、影像、labels保持不變。初始化hash與配置差異保存於artifacts/stagewise_lr/provenance.json。

## 驗證方式與結果

三份YAML經官方get_cfg驗證；相較原combine_recipe配置，斷言差異僅name／model／lr0。程式斷言前階段lr0*lrf等於下一階段lr0，三階段邊界設計成立。新腳本語法檢查通過。GPU啟動前空閒。實際啟動結果稍後追加。

## 困難與解法

無。既有腳本已支援checkpoint保存與程序隔離，本次新建獨立排程和設定。

## 未解事項與風險

較低P2/P3 LR可能保護特徵，也可能造成適應不足；須等結果判斷。舊P1第6輪曾從17輪上限延長至40，新P1直接以40輪規劃開始，故歷史LR軌跡不完全相同，不能宣稱嚴格只改一個因子。單seed／formal val選模限制持續。沒有啟用分組backbone/head自訂LR或自動ReduceLROnPlateau，本次只測階段基準LR遞減。背景監測不代表代理交付後持續在線調參。

## 實際啟動驗證

19:29:32啟動，P1已完成2輪，CSV數值均有限；實際args確認官方權重初始化、resume=False、lr0=0.00038、epochs40、freeze23。best／last／last-resumable已保存。GPU使用率62%、13689 MiB，尚無OOM；P2／P3與最終比較均已排妥。

## 完成狀態查詢

2026-09-17 22:38:23三階段與統一評估完成；P1／P2／P3實跑40／60／200輪，P3最佳第163輪，GPU使用率0%。固定640／batch8評估：新P3 Box=0.62881、Pose=0.92062；原combine P3=0.63445／0.91701，Box下降約0.56個百分點，Pose提高約0.36個百分點。新模型ball Box／Pose=0.52916／0.89694、bat=0.72846／0.94429。640綜合最佳仍是原combine P3，Pose最高為新stagewise P3，全部checkpoint保留且未替換final。

本輪selection的960 reference取所有歷史候選平均分最高者，包含先前因bat Pose退步未通過原保護門檻的960精修；因此本輪列其為reference不代表重新證明它通過了對舊Stage4的保護，也沒有自動更換部署選擇。此次只查詢與記錄，困難：無；單seed／驗證選模限制不變。
