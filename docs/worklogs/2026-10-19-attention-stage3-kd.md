# 2026-10-19：第 3 階段加入 KD 恢復

## 任務與範圍

使用者明確同意 KD 加在本 Stage。維持 D1 部署圖、原始結果與 Stage 4 停止邊界，只執行一個最多五輪的 KD 候選。先讀 optimize spatial_kd、calibration、實驗配方及負結果，保存參考 SHA，再開始修改。

## 變更內容與原因

新增帶 bat 區域保護的三尺度 spatial attention transfer；教師原始 YOLO26m eval/no_grad，不進學生、EMA、optimizer 或匯出。原教師 bat AP 略弱，區域保護是專案延伸，仍用八項 AP 選型。固定 S1 scale/bias、PWL、D1 reciprocal、BN buffers；只適應 Attention／Detect head。

重用 binaryqk 訓練引擎，只新增 loss 與採用門檻介面，舊政策預設行為不變。新增 KD manifest provenance 與 fail-closed loader，仍使用原 QueueExecutor、CLI 和 600 秒監測。四批 train-only 梯度校準 → AMP smoke → 五輪 → reload AP → 分析與選型；不加 control、不變資料集、不修改 optimize。

## 驗證方式與結果

新測試先因缺少 KD 實作失敗；後續 CPU 測試、Ruff、CLI、GPU 校準／smoke／訓練狀態逐項補記，不將 CPU smoke 當正式 AP。完整 COCO train118287／val5000，沿用既有 immutable runtime view，BBAT5 不涉及。

## 困難與解法

原始教師不是八項指標全面較強：bat AP50–95 較 D1 低 0.001489308。KD 排除 bat 區域，保留 native 任務 loss，正式採用仍要求八項各不下降超過 0.001，overall 至少增加 0.001；未通過維持 D1。

## 未解事項或風險

最終第五輪有小幅 AP 提升，但未達採用門檻（完整結果見文末）。單 seed、未新增相同額外訓練 control，無法分離 KD 與加訓效果。沒有硬體綜合／板端量測，QKV／PV／projection 仍浮點。完成本 Stage 後停下，不啟動下一階段。


## 啟動驗證

Attention 全套 202 passed（28.07 秒，包含環境可用的 CUDA 單元檢查），共用訓練／Stage 2 10 passed（39.28 秒），Ruff 與 CLI help 通過。dry-run 確認首項為校準、will_execute=false。正式佇列已依授權啟動，來源鎖定於 `yolo_attention/artifacts/stage3/kd-20261019-v1/source.json`；尚未宣稱正式 AP 改善。


## Train-only 校準結果

四批共 64 張已通過，固定 μ=1.274051769，實際加權 feature 梯度比例 0.074876～0.121202，均在 0.025～0.4 門檻內。兩處 Q/K 的 KD-only 梯度皆非零，教師及學生 state 均未改變。第 1 批含 1 個 bat GT，三尺度遮罩分別排除 880／261／96 個位置；四批沒有 ball GT，不能把這次校準解讀成球類區域精度驗證，正式每輪 val5000 檢查 ball AP。校準峰值 allocated 9.141 GiB。


## AMP smoke 結果

兩個 logical macro、512 張訓練影像通過，實際訓練 15.719 秒，AMP retry=0，峰值 allocated 6.554 GiB。兩處 Q/K 權重最大更新皆約 2.00048e-5；凍結 live／EMA state、教師 state、部署 manifest、checkpoint roundtrip、optimizer／scaler resume 檢查皆通過。smoke 僅評估 16 張，沒有官方 AP，result 的 smoke best=0 是引擎佔位，不能解讀成正式 AP 為零。正式訓練重新從 D1 出發，正在重做完整 epoch0 val5000。


## 正式起點核對

重新載入完整 epoch0 val5000 的官方 overall AP50–95=0.5141893485220431，八項 AP 與原 D1 完全一致，annotation SHA 亦相同，通過事前 1e-8 核對。正式五輪流程已啟動。未改寫原 Stage 3 source snapshot；新程式版本與 KD 配方另存於本輪 source.json。


## 背景訓練交接（2026-10-19T14:07:16.042259+08:00）

正式第 1 輪已完成 20 個 optimizer steps／5120 張，耗時 85.23 秒，AMP retry=0、梯度有限。依當前速度預估五輪與驗證於 2026-10-19 16:53（台灣時間）完成，對外以約 17:00、16:45～17:15 範圍說明。實際時間受後續速度影響。GPU worker PID 779774、背景 supervisor PID 779765，程序顯存約 8248 MiB；目前無其他 compute process。每 600 秒監測，結束自動接 reload val5000、八項 AP 比較、CSV／SVG／報告與採用判斷，之後停止。

本次合計 202＋10 項測試通過；最後重新核對 code source 與 optimize reference SHA 仍一致。長訓期間退出 working，不啟動 Stage 4。正式 KD 精度改善尚待後續各輪完整評估。


## 狀態核對（2026-10-19 16:33，台灣時間）

已完成前四輪，每輪完整 train118287／val5000；正在第 5 輪，最新進度 71680／118287 張、280／463 步。GPU 程序顯存 8248 MiB，讀取時 GPU utilization 61%、溫度 56°C，沒有其他 compute process。前四輪 AMP retry 全為 0，最新一步亦為 0；600 秒背景監測持續，沒有失敗 job。預估含最後驗證與選型約 16:48，對外以 16:50 左右回報。

| 已完成輪次 | Overall AP50–95 | Ball AP50–95 | Bat AP50–95 |
| --- | ---: | ---: | ---: |
| 0 | 0.514189349 | 0.512681031 | 0.486021483 |
| 1 | 0.514091665 | 0.513237555 | 0.485263757 |
| 2 | 0.513947140 | 0.513846228 | 0.490078565 |
| 3 | 0.513920214 | 0.511347314 | 0.488455604 |
| 4 | 0.514022740 | 0.513319775 | 0.484015670 |

目前各輪 overall 都未超過原 D1（0.514189349）；第四輪 ball 較 D1 增加約 0.000639，但 bat 下降約 0.002006，未達八項 AP 採用門檻。第二輪 bat 曾升至 0.490079，但 overall 仍略低。最佳合格模型仍為原 D1，最後一輪尚未驗證，不提前作最終採用結論。維持原五輪預算，不追加訓練、不啟動 Stage 4。

本次只查閱程序、GPU、queue、metrics 與監測紀錄，未改訓練程式／設定；程式測試不重跑。困難：尚無合格 KD 提升；處理方式是跑完已授權預算後按固定門檻選型，未通過就沿用 D1。


## 背景流程完成

KD 未通過採用門檻，維持原 D1。完成 5 輪；選定 epoch 0，重新載入官方 AP50–95 0.514189349，D1 為 0.514189349。實際採用模型：`/home/uxin/yolo/yolo_attention/result/stage3/evaluated.pt`。

八項 AP 與各輪結果見 comparison.csv，契約與教師限制見 docs/stage3-recovery-kd.md。單 seed，沒有額外 control 加訓，不能分離 KD 與額外訓練的獨立貢獻。無硬體實測。第 4 階段未開始。


## 最終收尾核對（2026-10-19 17:35）

2026-10-19 **16:48:05** 完成，五個 queue jobs 全部成功，總流程約 2 小時 45 分 36 秒。17:33 核對 GPU 已無 compute process，僅顯示系統約 611 MiB。

**KD 有小幅提升，但未達事前採用門檻，主線維持原 D1。** 第 5 輪是最高 overall AP 的 KD 結果：0.514189349 → **0.514620791**，增加 **0.000431442（0.04314 個百分點）**；事前要求至少 +0.001（0.1 個百分點）。八項 AP 各不下降超過 0.001 的條件已通過，唯一未通過的是 overall 最低提升門檻。不更改事前門檻，不追加訓練；Stage 4 未開始。


| 指標 | 原 D1 | KD 第 5 輪 | 差異 |
| --- | ---: | ---: | ---: |
| 整體 AP50 | 0.687536147 | 0.688156664 | +0.000620518 |
| 整體 AP50–95 | 0.514189349 | 0.514620791 | +0.000431442 |
| Person AP50 | 0.865018885 | 0.865282951 | +0.000264066 |
| Person AP50–95 | 0.643273092 | 0.643206570 | -0.000066522 |
| Ball AP50 | 0.716171605 | 0.719326365 | +0.003154760 |
| Ball AP50–95 | 0.512681031 | 0.513224007 | +0.000542975 |
| Bat AP50 | 0.721461724 | 0.724143671 | +0.002681948 |
| Bat AP50–95 | 0.486021483 | 0.487594089 | +0.001572606 |

確認第五輪 live／EMA resume SHA 與 next_epoch=5；CPU 嚴格載入兩種 state 後，固定 affine／PWL／reciprocal／activation manifest 均相同，教師不在 state 中，長方形輸入均有限。重載完整 GPU AP 的對象是通過門檻的 D1；第五輪 KD AP 是訓練 EMA 的完整驗證，未額外跑 KD reload AP，沒有混稱。

自動摘要先前只列選定 epoch0，容易掩蓋第五輪的小幅收益。本次補齊最終報告、final-review.json、README、整合計畫、result 交接與清理紀錄；不改方法、原始指標或事前門檻。保留唯一第五輪 resume，既有 smoke 清理釋放 277.53 MiB。沒有啟動新 GPU 實驗或 Stage 4，沒有發布 Git。

困難與解法：第五輪雖有提升，但只有 +0.000431，未達最低 +0.001；按固定門檻維持 D1，同時保留真實正面結果及候選權重，不寫成 KD 完全無效。其餘異常：無。


收尾文件驗證通過：報告八項 AP 與原 CSV 逐項一致，本輪報告 6 個本機連結可解析；只複製小型 training/result.json 至暫存目錄、停用 CUDA、無任何 checkpoint，成功重畫並解析 SVG。D1 handoff 路徑／SHA 與採用判斷相符，source snapshot 及 optimize reference 再次核對未變。此輪只有文件與 CPU 產物稽核，沿用既有 212 項測試結果，不重跑訓練／AP／無關測試。
