# 歷史入口快照（非目前狀態）

原路徑：`kd/dual_task_v1/README.md`。原始位元組另存同名 `.txt`；整理日期 2026-09-12。

# 方向 2：已核准雙教師 KD

目前兩臂各5輪已完成，KD 相對 K0 有 BBAT 收益但未過起點保護線，未替換正式候選。下一步見 凍結非 Pose 的專項規劃（本機／歷史參照：`../../../kd/dual_task_v1/POSE_ONLY_NEXT_PLAN.md`；未隨本次報告發布），尚未執行新訓練；下列啟動狀態為前一階段歷史。

最新狀態：使用者已核准雙教師，教師驗證八項均較強，跨尺寸空間 KD 校準與兩臂真實 smoke 已通過。MuSGD 因 attention 零更新中位數未通過校準，主線採 AdamW。正式五輪配對設定見 目前計畫（本機／歷史參照：`../../../kd/dual_task_v1/PLAN.md`；未隨本次報告發布）；以下單教師阻礙／待決策文字為核准前歷史，不代表仍等待同意。

使用者補充較大 Detect 教師的可能性，已核對官方 L-Pose 存在但人體標註不同，詳見 教師候選與最少實驗（本機／歷史參照：`../../../kd/dual_task_v1/TEACHER_OPTIONS.md`；未隨本次報告發布）。CPU score 梯度前置已通過；不代表全模型 KD 就緒。

使用者要求 activation 後接 KD，特別提醒 COCO Detect＋BBAT5 Pose 是兩份不同標註的資料。原 R2-REGION 規格已採 task-routed joint 蒸餾；本資料夾保留此次新學生的前置證據，不把舊結果混入新模型。

## 不變的資料契約

COCO batch只送Detect native loss／對應teacher訊號，BBAT5 batch只送Pose native loss／對應teacher訊號；兩者共用學生trunk，以原macro正規化和權重合併。教師與學生吃同一份完成增強的影像。不得混COCO80／BBAT2 class ID、補造未標keypoint或將另一任務未標物件當負例。Canonical BBAT5 v1來源與split不改。

## 已完成與阻礙

新學生：qSiLU P3 bridge E2，SHA256與完整驗證見 Activation結果（本機／歷史參照：`../../../activation/bridge_v1/RESULTS.md`；未隨本次報告發布）。
直接FP-QK轉換的joint teacher候選已全量測試，八項都比學生差；不能把這個候選當合格teacher。兩個head融合不代表KD不可行，真正問題是teacher須在各自任務提供有效訊號。
此外，native BinaryScore 的XNOR路徑沒有Q/K輸入梯度，`last_scores`亦是detach診斷buffer。必須使用live tap，且在既定凍結Q/K條件下驗證上游可學習。`preflight.py`對既有training-only exact-forward surrogate做CPU查核；不表示全模型KD／AMP／GT mapper已通過，沒有啟動KD訓練。

## 需要確定的教師路線

原方案是單一、已驗收的FP-QK joint teacher＋固定配對預算ranking KD。維持此方案，需要另訓練／驗證FP-QK joint teacher，不可把目前較弱的切換候選直接拿來教。
另一方案是資料集分流教師：COCO使用較強Detect teacher、BBAT5使用舊的強Pose teacher，分別教新joint學生。教師只存在訓練，不增加部署heads或每圖scale。但這是multi-teacher、可能需要改成對應task輸出／特徵KD，並非原「單一FP joint排序蒸餾」；需要使用者明確確認研究方案變更，再做teacher同口徑優勢與loss梯度校準。

不自行用新loss堆疊去掩蓋原方案前置不成立。教師確定後才排K0對照／普通KD，普通KD有收益再區域創新；保留600秒blocking monitor。現在沒有KD queue或執行中的KD job。
# 最新完成狀態（2026-09-11）

K0／雙教師 spatial KD 各 5 輪已完成，沒有合格 best_joint；Pose-focus 5 輪及兩項推論研究亦已完成。現在沒有 active queue；詳細比較、全部 checkpoint 保存與後續條件見總報告（本機／歷史參照：`../../../reports/consolidated-20260911/README.md`；未隨本次報告發布）。以下保留各階段原始說明。
