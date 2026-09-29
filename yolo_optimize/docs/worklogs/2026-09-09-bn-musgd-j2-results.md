# 2026-09-09：B-HEAD BN、MuSGD 前置與 J2 條件式重驗

## BN 單變因診斷

重用 `diagnose_native_recovery.py --case trained_parent_bn --live-snapshot .../heads-only-parent-recovery/checkpoints/epoch-0005.pt`，完整 canonical BBAT5 val683。252 個 head BN buffers 換回 parent；live 參數 hash 前後一致：`7eb3b94df6502e3437b03f4c42e3ec9596043b2b4adcde1d199fc37e3878ea97`。

Ball Box 從 E5 live 0.4984766889 回到0.5013902226，但仍低於 parent 0.5074370362，delta=−0.0060468136。exit2 是精度 gate 拒絕，不是執行崩潰；結果保留，不重啟或放寬門檻。這支持 BN 有部分影響，不能證明「凍結 BN 訓練」一定有效；沒有直接啟動另一個5epoch猜測。

結果：`artifacts/direction1-20260908/diagnostics/heads-e5-live-parent-bn/result.json`。

## 原 O 計畫：train-only MuSGD 更新校準

新增 `scripts/probe_musgd_updates.py`，從同一原 PSEL EMA 重新建立 heads-only 圖與配對 criterion，每臂 fresh optimizer，不接退化 E5。這是 O 候選的更新尺度適配前置，不是已通過長訓 handoff／resume 或20epoch比較。

AdamW16 macros → MuSGD 探測16 → 依 role median 更新比做一次 LR 修正後確認16。每臂從原 PSEL 重建，原 macro 排程、輸入、augmentation、criterion、BN 初值、EMA age 同源；逐 batch 影像、標註與 augmentation tensors 的 trace hash 相同。這是原 training loader 前16 macros，不新建 dataset subset、不改 split；不使用 validation 調 LR，所有更新丟棄、不輸出新權重。

初版把部分無有效梯度參數混入統計，不符合計畫條件，標為前置實作不足，不採它的 LR／結論。第二版使用 optimizer step pre-hook，只在成功 step 前讀取 unscaled/clipped 非零有限梯度，排除不合資格樣本。另存 v2，不覆寫初版。總費用為初版48＋修正版48 macros，不能只報48。

修正版一次校準後：Detect LR=0.0006089066182、Pose LR=0.0012429811973。這些不是正式推薦 LR。相對同 trace AdamW 更新量中位數：

| 群組 | MuSGD／AdamW | 原安全帶0.5–2.0 |
| --- | ---: | --- |
| Detect decay | 1.0633 | 通過 |
| Detect no_decay | 0.2685 | 未通過 |
| Pose decay | 1.5013 | 通過 |
| Pose no_decay | 0.9232 | 通過 |

因此 `recipe_rejected`。依原 optimizer policy，不再為同一 arm 私加 group-specific LR／新混合 optimizer，不啟動 O-M20；也不宣稱 MuSGD 本身普遍無效。這只回答目前 PSEL／heads scope／指定 builder recipe 的適配，不外推 Neck／其他 checkpoint。

修正版確認48個 macros完成、各臂固定 state 與 EMA一致、EMA更新數正確、相同訓練 trace；沒有測驗證 AP、完整resume／export 或硬體 latency。結果：`artifacts/direction1-20260908/musgd-heads-train-probe-v2/summary.json`，初版留於同名無v2目錄。

## BEST 原計畫條件式 J2 重驗

目前 ball 弱項已有全量證據，因此觸發原 master plan 的 J2 候選，不是任意換較低 parent 掩蓋退化。runner 新增明確 `j2_best_joint`，只讀 `final/full35/weights/rollback/j2/inference/best_joint.pt`，同 Float／BitTrue、COCO80 val5000＋canonical BBAT5 val683 完整重驗。

| BitTrue AP50–95 | J3 BEST | J2 BEST |
| --- | ---: | ---: |
| COCO Box | 0.498022 | 0.497346 |
| Person | 0.620381 | 0.618511 |
| BBAT Box | 0.630036 | 0.630566 |
| BBAT Pose | 0.903717 | 0.901884 |
| Ball Box | 0.507437 | 0.510866 |
| Ball Pose | 0.859909 | 0.860799 |
| Bat Box | 0.752634 | 0.750266 |
| Bat Pose | 0.947526 | 0.942970 |
| Joint | 0.711175 | 0.710038 |

J2 Float joint=0.7099508109，BitTrue=0.7100382159。Ball Box 收益+0.00342870，Bat Pose損失−0.00455588，person也超過0.001回退容許值。因此保留 J3 PSEL；J2只能明列 ball 優先的應用取捨，不能說全面更好或 joint最高。checkpoint SHA256=`bd8aad5d944e088c6c9f77b5728eed6c0f5ac0509ed96941a05210c184623cb6`。結果：`artifacts/direction1-20260908/j2-best-joint-revalidation/summary.json`。

## 監測、困難與未解事項

所有 GPU 工作以獨立 child＋shell `wait(timeout=600)`，正常不讀 log、不查 GPU；只在退出後讀結果。BN gate exit2 正常判讀；兩版 MuSGD 與 J2 exit0。沒有同時啟動多份訓練或子代理。

困難：MuSGD 初版樣本资格不足，已明示修正並另存重跑；其餘無。沒有修改原權重、資料、歷史檔案或 split，沒有刪除、commit、push。

精度回升目標仍未達成，沒有新 accepted recipe。下一步對原第一輪必要交付與部署邊界逐項稽核：候選的停止 gate 不等於整個交付完成，也不將條件式候選無限制延長。原始使用者應用案例尚缺；現有12張開發案例不是独立 test 或部署實測。真正 latency／energy、export／量化口徑仍不得憑 CPU 等價或 AMP 訓練推定。
