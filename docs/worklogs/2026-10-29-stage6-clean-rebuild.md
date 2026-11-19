# 2026-10-29：清除舊融合內容並重新建立第六階段

## 授權與範圍

使用者確認刪除第六階段全部舊融合實驗產物，從頭依 optimize 的融合方式重建；不保留舊資料或封存。

## 已完成清理

直接刪除舊 stage6 artifacts、result、最近比較資料、相關工作紀錄及被取代的 queue／診斷／測試。沒有複製舊權重、指標或圖表。共移除 853 個一般檔案，約 5.562 GiB；符號連結只移除連結。

刪除範圍：

- `yolo_combine/artifacts/stage6`
- `yolo_combine/result/stage6`
- `docs/research/optimize-final-comparison-2026-10-29`
- `yolo_combine/stage6_queue`
- `yolo_combine/diagnostics`
- `yolo_combine/stage6_alignment.py`
- `yolo_combine/run_stage6_queue.sh`
- `yolo_combine/configs/stage6-continuation.json`
- `yolo_combine/stage6/calibration.py`
- `yolo_combine/stage6/diagnostics.py`
- `yolo_combine/stage6/loss.py`
- `yolo_combine/stage6/cli.py`
- `docs/worklogs/2026-10-28-combine-stage6-j1.md`
- `docs/worklogs/2026-10-28-combine-stage6-queue.md`
- `docs/worklogs/2026-10-28-combine-stage6-recovery-v2.md`
- `docs/worklogs/2026-10-28-combine-stage6-training-policy.md`
- `docs/worklogs/2026-10-28-combine-stage6-verification-repair.md`
- `docs/worklogs/2026-10-28-combine-stage6.md`
- `docs/worklogs/2026-10-29-combine-optimize-alignment.md`
- `docs/worklogs/2026-10-29-combine-stage6-progress.md`
- `docs/worklogs/2026-10-29-optimize-final-results-review.md`
- `yolo_combine/tests/test_stage6_contract.py`
- `yolo_combine/tests/test_stage6_queue.py`
- `yolo_combine/tests/test_stage6_recovery.py`

## 驗證與保護範圍

刪除前確認沒有融合 GPU／背景工作。optimize 未引用這批實驗產物；binaryqk 的資料防寫守衛只指定 runtime cache 目錄，重建時沿用同一位置。正式資料、optimize 模型與上游 Detect／Pose 權重的 SHA 在刪除前後一致。前五階段成果及 optimize 依賴的共用 src 程式保留。

## 困難與後續

無。舊融合結果已不再可用；將由第五階段 Detect 與受保護的已訓 Pose 重新建立同血緣適應、融合初始化與聯合訓練。新方法先唯讀核對 optimize 並保存 protocol，再實作及驗證。

## 重新建立的方法與事前參考

實作前唯讀核對 optimize `experiments/combine/bridge_v1/` 的程式、實際 resolved config、來源與結果，來源 SHA 見 [reference-review.json](../../yolo_combine/artifacts/stage6/reference-review.json)。採用 head 適應、full Pose gentle、同血緣 trunk 混合、M0／J1／J2／J3 與 train-only 梯度校準。`stage6/` 為唯一新主線，`training.py` 已替換。

先讓 Pose 分支在 Detect 起點適應，再混合；保留原 Detect anchor。每段獨立 LR、原生 loss horizon、fresh optimizer／EMA age 0。共同 BN 固定、可訓練 head BN 更新，不套用後期 Rep refinement 的全固定配方。Attention 全模組固定，保留 B2／S1 Q10／D1；與 optimize J3 的不同已記錄。條件延長為使用者授權的本專案延伸，不宣稱 optimize 已驗證。

## 實作與驗證

已實作全量資料、完整 epoch resume、AMP retry 的 BN 回復、固定 tensor／EMA 核對、600 秒背景監測、GPU 空閒等待、條件 queue 與重載／路由／AP 交付。README、研究計畫及索引已清除舊實驗指標與失效入口。

`test_stage6_rebuild.py` 的 9 項 CPU 測試通過（7.39 秒）：Detect／Pose head 精確保留、五種訓練範圍、BN／fresh EMA／LR、非零權重混合、Attention 防改、checkpoint roundtrip、逐項 gate／plateau resume。GPU macro、完整 AP 重現與正式訓練尚待執行，不能由 CPU 測試推論精度恢復。

## GPU 與限制

啟動前 RTX 5090 無其他運算程序，使用一張 GPU。先做 H0／P／J1／J2／J3 各一個實際 macro，丟棄短測權重；再重現獨立 AP，通過後才開始 H0 8 輪。後續段落依門檻與診斷接續，完成第六階段後停止。

清理與 CPU 驗證的困難：無。新融合精度仍待驗證；optimize 最終模型含後續方法，相同配方不保證相同 AP。單 seed，未新增原樣 control 加訓，沒有硬體綜合或板端量測。

## GPU 驗證與正式啟動

2026-10-29 13:44 啟動背景 supervisor。H0／P／J1／J2／J3 五種實際 macro 全部通過：有效梯度、實際參數更新、fresh EMA、固定 tensor 與硬體 manifest 均通過。初始 AMP 溢位由同 macro 降 scale／恢復 BN 後重試成功，沒有跳過樣本或更新。累計量測最高 allocated memory 19,484,205,568 bytes（約 18.15 GiB）。短測權重未保存作候選。

完整 COCO 5,000 張重現 Detect AP 0.5146207909643931，三個追蹤類別也與上游精確一致；獨立 Pose 在 BBAT 683 張重現 box AP 0.6284018405302074／pose AP 0.9207093886162863，六項一致。單任務、雙任務與原生 task view 的輸出逐位相同。剛接上 Detect 特徵的 Pose head 尚未適應，不能拿這個初始組裝輸出當作獨立 Pose 的精度。

13:45 起 H0 正式訓練已運行，Pose batch 16，每輪完整 5,964 張；Detect、共享 trunk、Attention 固定。首輪第 210/373 macro 的 forward/backward 進度穩定，loss 有限。背景 supervisor PID 2572871；每 600 秒檢查，工作完成或異常立即接續處理，整個 Stage6 完成後停止。

程式檢查：Ruff 通過、shell syntax 通過、新入口與所有 runtime module import 通過。正式 AP 恢復程度與後續段落結果仍待訓練，不預先宣稱達標。

H0 第 1 輪已完整完成並保存 resume（86.29 秒）：COCO AP 0.514620791，四項 Detect AP 與 anchor 完全一致；BBAT box AP 0.452212、pose AP 0.682490，仍在恢復中。全部凍結 tensor／EMA 核對通過，未觸發安全線。H0 共 8 輪，依首輪速度粗估今天 13:58～14:05 完成；整個第六階段包含條件式後續適應，不能以 H0 完成時間代替。
