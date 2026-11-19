# 2026-09-17 保留目前模型並改善 Box

## 變更與原因

使用者要求依目前結果繼續改善，並明確保留當前模型。將stagewise_lr_p3_640的best.pt、last.pt、last-resumable.pt、args.yaml、results.csv、completed.json另存至artifacts/preserved/stagewise_lr_p3_640/，逐一比對SHA256；manifest.json記錄來源與摘要。best對應第163輪，last／last-resumable對應第200輪。舊run及所有先前模型亦保留，不覆寫任何final。

目前640 AP：本次保留模型Box0.628808／Pose0.920616、ball Box0.529158／Pose0.896941；原combine P3 Box0.634452／Pose0.917011、ball Box0.539151／Pose0.894054。故優先測框定位回補，並保護新模型Pose優勢。

## 實驗設計

兩組都從保留副本best初始化，固定Backbone（freeze11）、只訓練Neck＋Head，640、physical batch32、nbs128、MuSGD lr0=5e-5、lrf0.5、momentum0.948、WD0.00027、warmup3、warmup_bias_lr0、40輪上限、patience15。scale0.5、translate0.1、mosaic0、其餘loss及seed0不變。

Control：box=7.5；候選：box=10.0。程式比對兩組YAML，只有name與box不同。對照可區分共同精修與增加Box loss權重的效果；freeze、LR與原完整P3訓練不同，因此不能直接把相對原模型改善都歸因於box。

box=10影響兩個類別的框loss，不是只加權ball；因此最終必須分別檢查ball與bat。當前不增加Pose loss、不更改架構、不移植另一個模型的head，避免同時改太多因素。

資料仍為bbat5-v1，registry=/home/uxin/yolo/configs/datasets/bbat5-v1.yaml，canonical Pose YAML=/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml；formal train5964／val683不切分、不抽樣、不修改影像或labels。

## 排程與比較

configs/box_recovery/存設定；scripts/run_box_recovery.py依序執行兩組，各自保存best／last／last-resumable。完成後正式AP評估640／batch8，再對保留模型及兩個候選各做固定conf0.25的完整640誤差診斷。驗證集及診斷定義沿用先前工具。

tmux=bbat5_box_recovery_20260917；狀態artifacts/box_recovery/status.json，日誌logs/box_recovery_queue.log。每600秒記錄GPU，每5秒檢查退出，共用queue.lock防重複。

固定reference為此次保留的stagewise P3，不隨其他候選最高平均分漂移。建議條件：平均AP提高且Box AP至少+0.001；整體及ball／bat各自Pose AP最多下降0.003（0.3百分點），各自Box AP最多下降0.005。這是工程容許值，不是統計顯著性，且只產生建議、不自動替換。原combine P3也在結果表中供平衡比較。

## 驗證方式與結果

6個備份檔逐一SHA256相符。兩份YAML通過官方get_cfg；組間差異斷言通過。選模檢查包含改善Box且保住Pose可通過、ball Pose退步0.01應拒絕、無改善原模型不可當新改善。新程式語法檢查通過。GPU啟動前空閒，磁碟約504GB可用。實測啟動結果稍後追加。

## 困難與解法

無。原模型獨立副本及摘要保留；新實驗使用不同run與output。

## 未解事項與風險

框loss加權可能影響Pose與召回，不保证改善；原生best.pt按整體fitness選模，不代表所有Box/Pose取捨點都保留，另有每10輪checkpoint供追查。單seed、formal val反覆選模，沒有獨立test。本次保留原模型但不啟用外部通知；代理交付後由背景佇列監測與停止異常。

## GPU啟動驗證

22:51:25啟動對照組，已完成2輪；actual args確認保留副本初始化、resume=False、freeze11、box7.5、lr0=5e-5、batch32。最新CSV所有數值有限，best／last／last-resumable均已產生。GPU使用率98%，16853 MiB，未見OOM。box10及後續評估與診斷已排隊。
