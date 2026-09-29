# 2026-09-10：完整 Pose 模型先訓練，再融合

## 最新使用者要求

先把完整 Pose model 訓練好，不限於 Pose head；AdamW／MuSGD 由主代理依證據選擇。這取代「head-only 完成即可直接聯合融合」的推進方式。目前正常執行的 `j0-pose-extend-v1` 保留為前置適應，不修改執行中設定，不丟棄結果。

## 核對與規劃

唯讀核對原 `yolo_combine/final/full35/configs/joint.yaml`：正式 optimizer 為 AdamW，MuSGD 是 challenger，不會自動切換。原 final 的 physical Detect batch 為 64、shared BN affine 可訓練；本輪目前 physical 32、共享 BN affine 固定，因此是參照其政策，不是完全相同配置。

先採 AdamW。既有 optimize 的 MuSGD train-only 校準中，某 Detect no_decay 分組更新比 0.268 未過原安全帶；此證據不代表 MuSGD 永遠無用，但不足以直接切換本次完整 Pose 模型。未來若評估 MuSGD，需對本次實際 scope 做同起點 train-only 更新校準及獨立 run，不能只套相同 LR。

接續流程：目前 head-only 前置適應完成／平台分析 → 建立獨立 Pose 完整模型適應（Neck、backbone 分段解凍，硬體固定 Q/K／PoT／PWL 契約仍保留）→ 完整 BBAT box／keypoint 與 ball／bat 分項驗收 → 重新檢查兩模型共享特徵相容性 → 聯合融合 → activation → 方向 2。

完整 Pose 分支的 trunk 更新不得回寫正式 Detect；原 P3 bridge Detect 固定為 COCO 錨點。融合時不能把更新過的 Pose trunk 丟棄、只移 Pose head 後就宣稱保留完整 Pose 訓練收益；必須比較組裝前後完整 Pose AP，再做必要共享適應。獨立 Pose 的 COCO head 退化不能冒充正式 Detect 退化，正式 Detect 另行驗證。

## 執行與未解

本次只核對配置與更新接續政策，沒有額外 GPU job、沒有讀正常工作 log；原 Pose-only blocking monitor 持續。完整 Pose 的確切起點、解凍 LR 與回合数待前置結果及原 Pose 訓練來源核對後固定；尚未實作或啟動，不冒稱已排入自動 queue。

困難：獨立 Pose trunk 自適應與單一共享 trunk 的最終目標存在特徵相容性風險，已明確納入組裝前後 AP 驗證，尚未解決。沒有刪除、覆寫、commit 或 push。
