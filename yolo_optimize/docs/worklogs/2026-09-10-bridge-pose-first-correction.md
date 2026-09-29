# 2026-09-10：先完成 Pose head 適應，再融合

## 變更原因

使用者再次確認必須參照 combine，先訓練好 Pose head 再融合。先前 J0 完成 8 epoch，但原驗收未通過，不應把完成回合數視為適應完成。新 J1 首 epoch（463 macro）COCO overall／person 分別下降 0.040303633／0.035166824，已觸發保護，保留 checkpoint 後停止；不是 20 epoch 完成。

## 診斷與驗證

使用 diagnosing-bugs 的重現與單變因回退流程。執行 `combine/bridge_v1/diagnose_j1.py`，以安全 state-dict 載入及完整 COCO／BBAT5 BitTrue 驗證重現保存指標，再各自只還原一組來源狀態。來源不變、不重訓。

| 同 checkpoint 條件 | COCO overall AP | person AP |
| --- | ---: | ---: |
| 重現 J1 E1 | 0.46790832 | 0.59249730 |
| 只還原 Detect BN 統計 | 0.29586992 | 0.37733684 |
| 只還原 Neck | 0.40080012 | 0.53742007 |
| 只還原 Detect 參數 | 0.40140230 | 0.48163853 |
| 只還原 MASF | 0.46790387 | 0.59247565 |

MASF 更新回退只帶來微小差異，不能把大幅下降歸因於 MASF 更新。其他單組回退沒有恢復，顯示已訓練參數與統計有耦合；尚未找出單一根因，也未證明凍結某組重新訓練即可修復。這是診斷結果，不是假稱成功修復。

## Pose-only 接續

新增 `combine/bridge_v1/pose_first.py`，從 J1 之前的 J0 best Pose head 與原選定 P3 bridge Detect 建立新 run，不沿用退化 J1。全部共享 trunk、Detect、MASF（包括 α）固定，只訓練 Pose head；Pose-only EMA 逐位保留固定狀態。AdamW／head LR 2e-4、batch 16、640、warmup 1、完整 canonical BBAT5 train 5964 張，最多加訓 40 epoch。新 optimizer 與原生 loss horizon 40，不宣稱 exact resume。

原生 J0 沒有 patience 早停，因此本入口在保存 checkpoint 後，以六項 BBAT AP 平均分數的 min_delta 0.0001、patience 10 判斷平台；不冒稱原生 J0 已有此功能。每 epoch 完整 Float／BitTrue 驗證，COCO overall／person 必須與初始值相同（容差 1e-8）。仍比較原独立 Pose 基準，不降低驗收標準。

真實 smoke 2 個 Pose batch 通過，Pose 參數確實更新、live／EMA 非 Pose 的全部 state 完全固定、硬體契約通過。結果見 `combine/bridge_v1/artifacts/fusion/j0-extend-smoke-v1/summary.json`。正式 `j0-pose-extend-v1` 已啟動，事件檔 `combine/artifacts/logs/bridge-pose-first-v1.events.jsonl`，使用 600 秒 blocking monitor。

## 未解與限制

新 Pose 適應尚未完成；不能保證 head-only 一定能回到原独立 Pose。完成或平台後先檢查 ball／bat box／pose 與收斂，再決定是否足以解凍聯合訓練。J1、activation、方向 2 不自動跳過此驗收。舊 J1 保留供分析，不重跑、不覆寫。

困難：早期 stage 推進判斷過快，已改為先檢查適應程度，不以 8 epoch 完成作驗收。此次沒有資料或權重刪除，沒有 commit／push。
