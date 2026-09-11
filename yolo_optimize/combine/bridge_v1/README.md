# P3 bridge MASF 融合分支

## 最新狀態（2026-09-11）

J3、新舊完整對照、Pose head 恢復與 BN 校準均已完成。恢復最佳 E5 有部分框 AP 收益，COCO 不變，但仍未通過最終 gate；不升格 best_joint、不啟動 activation。目前無 GPU job。見[完整結果與下一步](<BBAT_RECOVERY_RESULTS.md>)。

## 以下為歷史進度（非目前狀態）

目前 active run：**balanced-j3-v1**。J2 已正常完成 23 個 epoch，COCO 保持但 BBAT 尚未全數過最終 gate；最後 J3 低 LR 全 backbone／合法 attention 微調已通過校準與真實 smoke 後啟動。固定 Pose weight 0.215、最多 20 epoch／patience 5，完成後集中驗證，不自動延長。見[最新紀錄](<../../docs/worklogs/2026-09-10-balanced-j2-result-j3.md>)。

目前 active run：**balanced-j2-v1**。J1已正常完成14epoch，最佳COCO .504954／person .625437／Pose .888378，COCO過保護但bat未過最終gate。J2解凍backbone9+、固定校準Pose weight .07，真實smoke通過後啟動。見[最新結果與J2設定](<../../docs/worklogs/2026-09-10-balanced-j1-result-j2.md>)。

實際 active run 為 **balanced-j1-v2**：v1 在任何 optimizer 更新前遇到配置輸出的 Session 繼承錯誤；修正與 CPU 回歸通過後僅重啟此工作，v1 保留。設定與資料不變，見[修復紀錄](<../../docs/worklogs/2026-09-10-balanced-j1-start.md>)。

當前：**balanced-j1-v1** 正式訓練已啟動，Pose weight0.045由train-only校準，最終COCO gate0.005不變。見[實際設定與訓練安全政策](<../../docs/worklogs/2026-09-10-balanced-j1-start.md>)。以下暫緩／校準中為歷史狀態。

最新：merge J0 已完成 11 epoch 平台停止，最佳 Pose AP 0.866087；J1 smoke 通過但正式 J1 尚未啟動。使用者提供[報告實驗 4 核對](<../../docs/research/2026-09-10-combine-report-experiment4.md>)後，發現目前 Pose 加權共享梯度仍約 Detect 5–8 倍，正在執行固定 checkpoint 的 train-only 校準；同時區分最終驗收與早期 safety stop。

最新：完整 Pose 完成 42 epoch 平台停止，最佳 AP 0.897997。共享初始化驗證後選 10% Pose trunk 作候選，`merge-j0-v1` 正在固定 trunk／Detect／MASF 下適應新 Pose head；不是已接受融合。見[完整結果與接續](<../../docs/worklogs/2026-09-10-full-pose-result-merge.md>)。下方為歷史狀態。

當前 run 是 `full-pose-gentle-v1`：首輪完整 Pose 在 E1 觸發 ball／bat 保護，已保存；同起點只將全部 LR×0.1 的受控嘗試已啟動。詳見[事件紀錄](<../../docs/worklogs/2026-09-10-full-pose-gentle.md>)。

目前 **完整 Pose AdamW 適應執行中**：前置 head-only 40 epoch 已完成，最佳 Pose AP 0.852475，COCO 不變；完整 Pose 的 backbone 卷積／Neck／head 已通過真實更新與固定狀態 smoke 後啟動。硬體固定 attention／PWL／BN 契約保留，原 Detect checkpoint 不動；不是已融合模型。見[最新結果與設定](<../../docs/worklogs/2026-09-10-full-pose-adamw-start.md>)。

最新要求進一步明確為先訓練好**完整 Pose model**，不只 head。目前 head-only run 作前置適應；後續獨立 Pose 分段解凍並驗證共享特徵相容性，再融合。AdamW 作主線，MuSGD 不自動切換。見[完整 Pose 接續政策](<../../docs/worklogs/2026-09-10-full-pose-before-fusion.md>)。

## 最新：先訓練好 Pose head

J1 首 epoch 觸發 COCO 精度保護並保存停止，未完成融合。依使用者確認，已回到 J1 前的 J0 best，只延長 Pose head 適應，固定 trunk／Detect／MASF；最多 40 epoch、patience 10，通過實際 smoke 後啟動 `artifacts/fusion/j0-pose-extend-v1/`。完成後先驗收再決定聯合解凍。見[修正與診斷紀錄](<../../docs/worklogs/2026-09-10-bridge-pose-first-correction.md>)。以下 J1 啟動文字是先前歷史。

使用者已選定保留 P3 bridge；J1 正式融合訓練已啟動。完整設定、驗證及限制見[工作紀錄](<../../docs/worklogs/2026-09-10-bridge-combine-restart.md>)。舊無 MASF 結果不覆寫。

```text
共享 trunk P3 raw ─┬─→ 後續 P4／P5
                  ├─→ MASF（可訓練 α）→ Detect P3
                  └─→ Pose P3
共享 P4／P5 ──────────→ Detect／Pose 各自 head
```

α=0 可做推論開關對照，但不會自動省略計算；Pose 不隨這個 Detect-only 開關改變。正式訓練允許共享 Neck 更新，因此訓練前後 Pose 仍可能改變。

CPU 等價、完整初始 Float／BitTrue 驗證、真實混合更新 smoke 均已通過。568 個共享 trunk 狀態張量完全一致，故直接沿用已完成 J0 的 Pose head；不重跑 8 epoch。

正式配置：[j1.yaml](<full35/configs/j1.yaml>)。結果入口：`artifacts/fusion/j1-bridge-v1/`，尚未完成。監測事件：`../artifacts/logs/bridge-j1-v1.events.jsonl`。

順序：J1 → 融合指標分析／必要適應 → 同 checkpoint α 開關驗證 → activation（QSILU 優先）→ 方向 2。後兩者尚未啟動或建立自動 queue。
# 最新完成狀態（2026-09-11）

完整 Pose／J0／balanced J1–J3、Pose 恢復、activation、兩項 KD 與推論研究均已有結果，沒有新的原嚴格 gate 全過模型；目前無 active GPU queue。所有 checkpoint、raw results 與外部來源已另存並雜湊驗證，見[全階段總報告](<../../reports/consolidated-20260911/README.md>)。以下執行中或尚未開始敘述為歷史。
