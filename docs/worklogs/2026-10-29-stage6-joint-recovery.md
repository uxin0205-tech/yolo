# 2026-10-29：J1 停止原因、分組診斷與雙教師恢復

## 任務與範圍

回應使用者「目前狀況」，並依已授權的最佳融合目標處理第六階段異常。所有工作仍在 `yolo_combine`；第七階段未啟動。`yolo_optimize` 唯讀。COCO train 118,287／val 5,000 與 BBAT5 v1 train 5,964／val 683 均未改來源、分割或標註。

## 已完成的 J1 與診斷

J1 在 18:46 完成第 6 輪，因 COCO baseball bat AP 低於原參考安全線停止。20:17 查詢時 GPU 已空閒；沒有把停止狀態誤報成仍在訓練。

| 指標 AP50–95 | 原獨立模型 | J1 起點 M0 E1 | J1 E6 |
|---|---:|---:|---:|
| COCO overall | 0.514621 | 0.514881 | 0.512510 |
| COCO person | 0.643207 | 0.642480 | 0.641580 |
| COCO sports ball | 0.513224 | 0.513438 | 0.516207 |
| COCO baseball bat | 0.487594 | 0.484259 | 0.462670 |
| BBAT box | 0.628402 | 0.581237 | 0.605645 |
| BBAT pose | 0.920709 | 0.884547 | 0.895802 |
| BBAT ball box | 0.529096 | 0.484925 | 0.484730 |
| BBAT ball pose | 0.897099 | 0.854019 | 0.857515 |
| BBAT bat box | 0.727708 | 0.677549 | 0.726559 |
| BBAT bat pose | 0.944319 | 0.915074 | 0.934088 |

Pose 平均值與球棒項目有進步，COCO 球棒有退化，BBAT 球框仍未恢復。這是具體的任務／類別取捨，不能描述為全部沒有改善，也不能宣稱已追平獨立模型。

事前唯讀參考 optimize `experiments/combine/bridge_v1/diagnose_j1.py` 的分組回退方法，並核對實際配方、resume 與指標。來源 SHA 與診斷規格記在 [reference-review.json](../../yolo_combine/artifacts/stage6/j1-review-20261029/reference-review.json)。新增 `yolo_combine/diagnose_stage6.py`，從真實 J1 resume 匯出 E6 EMA，進行以下五組完整 COCO／BBAT 驗證。

| 診斷 | COCO overall | COCO bat | BBAT box | BBAT pose |
|---|---:|---:|---:|---:|
| E6 原樣重現 | 0.512510 | 0.462670 | 0.605645 | 0.895802 |
| 還原 Detect BN | 0.496036 | 0.450163 | 0.605645 | 0.895802 |
| 還原 Detect parameters | 0.505024 | 0.468511 | 0.605645 | 0.895802 |
| 還原整個 Detect head | 0.511167 | 0.472178 | 0.605645 | 0.895802 |
| 還原 neck 11–22 | 0.512595 | 0.477338 | 0.582435 | 0.879665 |

E6 所有 AP 完全重現，Detect-only 回退不改 Pose 輸出。單換 BN 更差；還原 neck 可恢復 COCO bat，但丟失 Pose 改善。證據支持 head 與共享 neck 共同適應的取捨，未發現 evaluator 或 checkpoint 載入錯誤；這不等於已定位唯一因果。

原 J1 selector 要求 COCO 四項先通過最終門檻，因而未保存任何 J1 inference best；E5 只剩數據，不宣稱仍有其模型。E6 可從完整 resume 恢復，現有獨立匯出位於 `j1-review-20261029/reproduce/model.pt`，SHA `a6005897584db34c765f6accd1a4555051f449a10e4e716cf762e663ca02c570`。未覆寫 M0、P 或原獨立來源。

## 新恢復候選的事前核對

20:40 在實作前保存 [新 protocol](../../yolo_combine/artifacts/stage6/recovery-kd-20261029/protocol.json)，初始 SHA `ebc3a3bfe10cab995cf3bf08a639172a38aa1132c1acbfaf319eafc8e4405dc3`。核對 optimize `experiments/kd/dual_task_v1/` 的 `PLAN.md`、`spatial_kd.py`、`experiment.py`、`calibrate_kd.py`、`train_pair.py`，以及 `spatial-e5-seed1-v1` 的實際 resolved config／summary、KD 校準紀錄。

optimize 的 qSiLU E2 雙教師五輪未產生合格 best_joint，有球棒 Pose 收益及球框下降。不能宣稱該配方已成功。本輪是有限預算的新恢復候選，採用其三尺度空間 attention transfer、train-only 梯度比例校準與小 LR；不新增未改方法的 control 加訓。

本輪固定從 E6 EMA 出發，先跑 5 輪；fresh AdamW／EMA age0、warmup 1、cosine final 0.5，LR 為 backbone `3.8e-7`、neck `1.9e-6`、雙 head `5e-6`。開放第 9 層起的 backbone 與 neck／heads，固定全部 Attention、shared BN affine／running；active head BN 可更新。每 macro Detect 32 × 8 ＋ Pose 16，reference 64。

蒸餾在 layers 16／19／22 比較通道平方平均後的空間 L2 正規化特徵，不新增 projector。FP32 計算蒸餾、AMP 做前向；KD 先按圖求和，再沿用原 macro normalization。教師 eval／no_grad，用完移除 hooks，不進學生、EMA、optimizer 或部署模型。

教師採本輪已驗證 Stage5 D1 Detect 與受保護的已訓 Pose，並非 optimize 的 YOLO26L／full35 Pose。Detect 教師的 sports ball 比 E6 低約 0.00298，其他三個 COCO 指標較高；Pose 教師六項均較高。因此保留原生標註 loss，以約 0.1 的特徵梯度比例作溫和正則，不能保證所有指標受益。μ 由各四個 train batch 校準，接著用 16＋8 個真實 macro 重校任務權重，不用 validation 挑 loss 權重。

原最終 COCO 四項 −0.005、BBAT 六項 −0.02 的目標不變。新候選排序先看最終目標是否通過，再看 `0.5 × mean(COCO4) + 0.5 × mean(BBAT6)`，保存五輪候選以供比較；不再把未過最終門檻的有用模型全部丟掉。本輪暫時安全線為相對 E6 任一 COCO 再降逾 0.01、任一 BBAT 再降逾 0.03，或相對原獨立來源 COCO 降逾 0.05。已存在的 COCO bat 缺口保持可見。

## 實作與驗證

新增 `stage6/kd.py`、`stage6/recovery.py`，共用既有 trainer、calibration 與 600 秒 supervisor。舊配方預設不變，新 protocol 與參考 SHA 納入來源鎖。新 run 使用獨立來源鎖，不改寫已停止 J1 的歷史鎖。每輪完整驗證，預算結束後自動重載選定權重、重跑完整 AP 並產出比較報告；完成本候選不等於完成整個第六階段。

CPU：`test_stage6_rebuild.py` 與 `test_stage6_kd.py` 共 **21 passed**；涵蓋 KD 活梯度、teacher detach、microbatch 可加性、零特徵、不同通道、空間錯位拒絕、異常後 hooks 清除、恢復選型與原驗收門檻分離。相關 Ruff 全部通過。

GPU：μ 校準、兩個真實 macro smoke 與 16＋8 個任務權重校準全部通過。μD=3.2604688856、μP=11.5576622965；Pose 權重 0.085，確認組共享梯度比中位數 1.613111，落在事前 [0.5, 2] 範圍。KD-only 梯度確實抵達 neck；實際更新、固定 tensors／EMA、教師 state 不變、教師無梯度、hooks 清除、criterion weights_only roundtrip 均通過。Smoke 峰值 allocated 11.814 GiB；此數字不是整體 GPU reservation。

20:46 啟動五輪正式恢復。2026-10-29 20:49 已觀測第 1 輪 40／463 macro、EMA 40 次更新，無 AMP 重試或錯誤。GPU 使用約 13.6 GiB、利用率 99%；每輪初估 24～25 分鐘，預估 22:45～23:00 完成本候選，並非整個第六階段已完成。Supervisor PID 2600062，worker PID 2600265；背景每 600 秒監測，完成後自動重載完整驗證及產出 `report.md`，異常立即停止並寫明原因。

執行位置：`yolo_combine/artifacts/stage6/recovery-kd-20261029/`。查詢指令：`bash yolo_combine/run_stage6.sh status --output yolo_combine/artifacts/stage6/recovery-kd-20261029`。保持執行中的來源鎖，不在長訓期間修改程式或配方。

## 困難與解法

共享後 COCO bat 與 Pose 特徵有取捨；五組完整回退診斷已排除單換 BN 能解決的假設。舊 best 選型沒有保留未達最終門檻的候選，本次從可追溯 E6 resume 匯出並重驗，新候選同時保存逐輪資料與模型。

手動查來源鎖時第一次使用相對路徑，被路徑檢查拒絕；改以 supervisor 實際採用的絕對路徑後，root／train 兩份來源鎖與受保護來源 SHA 全部通過。未修改執行中的程式或來源鎖，訓練不受影響。

本次沒有清空正在使用的成果，也未新增 archive／backup；前次已刪除的舊融合產物未恢復。診斷只保存兩份有用權重，其餘保留小型數據；新五輪候選待分析後再處理不採用的大型產物。

## 未解事項與風險

尚未有本次 KD 的任務 AP，不能預告成功。只有單 seed 且同時調整 LR、範圍及 KD，無法獨立歸因 KD 效果。最終模型仍需逐項比較精度與原始目標；若有更佳取捨，先如實交付，不自動跨到 Stage7。部署保留 B2、Q10、D1 PWL exp＋LUT256 reciprocal；沒有新增硬體實測加速結論。
