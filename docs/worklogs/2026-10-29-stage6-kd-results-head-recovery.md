# 2026-10-29：KD 五輪完成，接續 Pose head 恢復

> 後續：第六階段已完成本輪交付，optimize 也已使用相同 batch 與 evaluator 重驗；最新結果見 [2026-10-30 完成紀錄](2026-10-30-stage6-results.md)。下方保留當時快照。

查核時間：2026-10-29 23:45（Asia/Taipei）。第六階段仍在進行；第七階段未啟動。

## KD 結果與候選比較

五輪 KD 於 **22:43:23** 完成，依事前綜合選型保留第 2 輪。重新載入後完整 COCO 5,000／BBAT 683 AP 與原結果完全相同，SHA `0790a6072f9843213d71d7bd3fe6a31cad88a11d6e762b30520aebdba87cbf96`。第 3～5 輪未超過 E2，不延長相同 KD 配方。兩次 AMP overflow 均成功重試，五輪沒有安全停止。

與 J1 E6 相比，KD E2 官方 COCO bat 從 0.462670 恢復到 0.470841；BBAT box／pose 從 0.605645／0.895802 到 0.605234／0.894992，整體沒有明顯恢復。最初獨立來源的全部 AP 目標仍未通過。

另外比較保留的 H0、H1、P、三個 rho、M0、J1 回退候選與五輪 KD。以事前 `0.5*mean(COCO4)+0.5*mean(BBAT6)` 排序，KD E2 仍最高，並非只在單次 KD 內挑選。完整候選向量見 [比較快照](../../yolo_combine/artifacts/stage6/pose-head-recovery-20261029/prior-candidate-comparison.json)。部分名稱指向同一權重，不計為獨立實驗。

## Pose head 恢復的事前參考與實作

依使用者已授權的最佳融合目標在第六階段內接續。先唯讀核對 optimize `experiments/combine/bridge_v1/recover_pose_head.py`、`pose_first.py`、`experiments/combine/j0_runtime.py`、實際 resolved config、recovery-stop、BBAT_RECOVERY_RESULTS，以及共用 stage policy 的 head BN 行為。optimize 原 J3 E12 的 head 恢復跑 9 輪、選 E5，box +0.004881、Pose +0.000676、ball box +0.006042，但 ball Pose -0.001361；只證明部分收益，不能保證本輪有效。

實作前已保存 [protocol](../../yolo_combine/artifacts/stage6/pose-head-recovery-20261029/protocol.json)，SHA `8a29736230e28f4c32d4376a40ae9007b3cc37a0de95eb8e1adb09b927f171e7`。從 KD E2 出發，只更新 Pose head；shared／Detect／完整 Attention 固定，Pose head BN 可更新。停止本段 KD 蒸餾，使用原生 Pose loss；不重訓原獨立 Pose、不新增原樣 control。

AdamW，Pose head LR 2e-5，fresh optimizer／EMA age0，warmup1，cosine final0.5，原生 loss horizon10，最多10輪、plateau4、min_delta0.0001，physical Pose16／reference64，seed20261003。每輪四項 COCO 必須與起點完全一致，任一 BBAT 比起點下降超過0.02即停止；原最終目標另行保留。選型按最終驗收與 mean(box,pose)，保留最佳權重和最新 resume。

共用既有 trainer／supervisor，擴充 head 恢復入口、兩個 macro smoke 與專用安全線。已完成的 KD source lock 保持原樣；新 run 有獨立新鎖，沒有覆寫歷史。B2 sign＋Hadamard、S1 Q10／bias、D1 整數 PWL＋LUT256 reciprocal 全部繼承，目前 activation 仍為 Stage6 SiLU。

## 驗證與最新進度

相關 CPU 測試 **23 passed**，Ruff 通過。GPU smoke 兩個真實 Pose macro 通過，258 個參數 tensor 實際更新；live／EMA 的全部固定 tensor 不變，peak allocated 約2.95 GiB。Smoke 更新丟棄，正式訓練重新載入 KD E2。

23:39 啟動 Pose head 恢復。現已完成 4 輪，目前最佳是第 1 輪；所有完成輪次 COCO 四項逐值不變。BBAT box 0.605234 → 0.609202，Pose 0.894992 → 0.899850，球框 0.486577 → 0.493883，球關鍵點 0.855794 → 0.863474。第一輪4次 AMP 重試成功；沒有 OOM、NaN 中止或固定 state 變動。

| 內部 AP50–95 | optimize 最終 Rep17＋20 E5 | 本輪目前最佳 |
|---|---:|---:|
| COCO 整體 | 0.505869 | 0.505687 |
| COCO person | 0.626986 | 0.624764 |
| BBAT 框 | 0.609797 | 0.609202 |
| BBAT 關鍵點 | 0.891041 | 0.899850 |
| 球框 | 0.492397 | 0.493883 |
| 球關鍵點 | 0.858876 | 0.863474 |
| 球棒框 | 0.727198 | 0.724521 |
| 球棒關鍵點 | 0.923205 | 0.936227 |

比較統一採內部 AP；本輪官方 COCO AP 另為0.512676。兩邊 validation batch／架構仍不同，尚未同設定重新評估 optimize，微小差距不宣稱顯著；本輪尚未完成最終重載驗證。

每輪含完整驗證約81～86秒；若跑滿10輪，預估 **23:53～23:56** 完成本候選與重載，plateau可能更早停止。背景每600秒監測；本輪完成後仍須分析第六階段交付，不自動啟動第七階段。

## 資源清理

KD 舊 supervisor 已退出、E2 已驗證且為新 run parent。核對後直接移除 E1／E3／E4／E5 的未採用 inference 與已停止 E5 optimizer resume，共10個檔案，釋放 **742.60 MiB**。E2 權重、各 epoch AP、小型資料、來源鎖、protocol、驗證均保留；沒有 archive／backup，未動 optimize、資料或其他研究成果。

為避免改動新 run 已鎖定的輸入，舊 result／history 的原始路徑記錄保持不變；實際刪除與 SHA 記於 [清理稽核](../../yolo_combine/artifacts/stage6/recovery-kd-20261029/cleanup-20261029.json)，不能再當作所有舊候選檔案仍存在。

## 困難與未解事項

首次建立事前核對清單時誤將 j0_runtime.py 當作 bridge_v1 子檔；路徑檢查在寫入 protocol 前拒絕，已追查到上層 experiments/combine/j0_runtime.py 並實際閱讀。新 run 啟動前已修正，沒有以缺失參考啟動訓練。

KD 收益有限，後段有精度取捨，因此改採有 optimize 證據的固定共享層 head 適應。這是連續研究，單 seed；訓練中較佳結果不代表全階段或原精度目標已完成。原 parent、正式資料、不可變 BBAT assignment／labels 均保持。

[程序、曲線數據與比較快照](../../yolo_combine/artifacts/stage6/pose-head-recovery-20261029/progress-review-20261029-2343.json)
