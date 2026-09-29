# 2026-09-10：J2 平台結果與最後 J3 低 LR 微調

## J2 完成

`balanced-j2-v1` 正常完成 23 個 epoch／10649 macro，平台停止，無 safety stop。最終 best_pose／best_detect 同為 E22（zero-based 21），不是先前進度快照的 E6。best_pose 定義是 box 與 pose AP 平均，不能將它誤稱最高 keypoint AP。

| 指標 | J2 E22 最佳平均候選 |
| --- | ---: |
| COCO overall | 0.505620443 |
| COCO person | 0.626900301 |
| BBAT box | 0.605188970 |
| BBAT pose | 0.883298603 |
| ball box | 0.487272310 |
| ball pose | 0.850026727 |
| bat box | 0.723105630 |
| bat pose | 0.916570479 |

COCO 仍符合原 Detect 0.005 保護；六項 BBAT 相对原独立 Pose 仍有約 0.024–0.032 的差距，未達 0.02，所以沒有 best_joint。J2 改善 bat，但 ball 相對 J1 E6 降低，不能只呈現改善項。J1、J2 與完整独立 Pose 全部保留。

## J3 準備與設定

依使用者提供報告的最後 J3 低 LR 思路，新增 `j3_stage.py`／`j3_train.py`，從 J2 E22 全部共享 trunk 與兩 heads 接續，來源 SHA256 `30df89e8791b06e89ac51e099317d810299973e7b6b5adaeb0a2ef82b44ccfcc`。新 optimizer／20 epoch loss horizon，不是 exact resume。

J3-only 明確列入 stages，不重跑 J0–J2；配置 enable_j3 保持 false 僅為避免 trainer 再 append 一次 J3，resolved config 明記 explicit_J3_only。這是既定融合路線最後一個微調階段，完成後集中驗證，不自動延長同一輪。

AdamW：backbone 全段 3.8e-7、Neck 1.9e-6、兩 heads 各 5e-6、MASF／α 1e-6、合法 attention 參數 5e-8。Q/K sign path、gamma、PoT、PWL [-10,0] 20 段保持固定；shared BN 統計及 affine 固定。warmup 1、最多 20 epoch、patience 5；資料／batch 與前階段相同。

## 梯度與 smoke 驗證

全 backbone 解凍使共享 gradient scope 改變，重新做 16＋8 個固定 checkpoint、train-only macro 校準；weight 0.25 的 norm ratio 中位數 1.1657679，提出固定 Pose weight 0.215，後 8 個確認中位數 0.9618736。只根據 training，不使用 validation 調權重。

校準第一次啟動在 Python 解析階段因新來源映射漏一個 `]` 而失敗，沒有任何 GPU 計算或 optimizer 更新。修正後 AST 語法驗證通過，僅重啟該校準，第二次正常完成；原 ERROR log 保留。有效校準輸出仍為 `j3-task-weight-calibration-v1/summary.json`，成功事件檔為 `j3-task-weight-calibration-v2.events.jsonl`。

`balanced-j3-smoke-v1` 真實兩個 joint macro 通過：layer0 backbone、合法 attention、MASF α 實際更新；context／α 梯度有限非零；固定 live／EMA、MASF BN 與硬體契約全部通過。共同 smoke 保留 J1/J2 layer0 固定斷言，僅 J3 允許全 backbone。

## 啟動與未解

正式 `balanced-j3-v1` 已啟動，事件檔 `combine/artifacts/logs/balanced-j3-v1.events.jsonl`，600 秒 blocking monitor。最終 COCO 0.005／BBAT 0.02 gate 不變，訓練安全與寬限沿用 balanced J1，Pose 災難性下降改對本次 J2 起點。

困難：J2 的各類收益不一致，J3 仍不保證通過最終驗收。完成後必須集中比較 J1／J2／J3、重新驗證匯出與同 checkpoint MASF 開／關；若未達 gate，不能自動把 activation 當成補救或宣稱融合完成。沒有覆寫、刪除、commit 或 push。
