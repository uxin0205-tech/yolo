# 歷史入口快照（非目前狀態）

原路徑：`combine/README.md`。原始位元組另存同名 `.txt`；整理日期 2026-09-12。

# Combine：方向 1 後的獨立融合準備區

最新：head-only 加訓 40 epoch 已完成，最佳 Pose AP 0.852475、COCO 不變；**完整 Pose AdamW 適應已啟動**，更新 backbone 卷積／Neck／Pose head，保留独立 Detect。尚未融合，見[最新紀錄](<../../worklogs/2026-09-10-full-pose-adamw-start.md>)。

目前執行 **Pose head 延長適應**，不是 J1：首個 J1 epoch 觸發 COCO 精度保護，已保存停止。依使用者確認先訓練好 Pose head，`bridge_v1/pose_first.py` 從未退化的 J0 head 接續，固定 trunk／Detect／MASF；完成後先驗收再決定解凍。見[最新紀錄](<../../worklogs/2026-09-10-bridge-pose-first-correction.md>)。下方 J1 訓練中為先前歷史。

## 最新狀態（2026-09-11）

J3、新舊全量比較、Pose head 補訓與 BN 校準均已完成；部分框 AP 回升但 BBAT 仍未全部驗收。目前無 GPU job，未啟動或排隊 activation／方向 2。見完整結果（本機／歷史參照：`../../../combine/bridge_v1/BBAT_RECOVERY_RESULTS.md`；未隨本次報告發布）。本頁其餘進度敘述為歷史，權威計畫在 bridge_v1/plan.json，舊 queue 不重啟。

## 歷史狀態：P3 bridge J1 訓練中

使用者已選定保留 P3 bridge MASF，新工作在 bridge_v1（本機／歷史參照：`../../../combine/bridge_v1/README.md`；未隨本次報告發布）；安全載入、CPU 等價、完整初始驗證與混合更新 smoke 均通過。568 個共享 trunk 狀態張量完全相同，故沿用已完成 J0 的 Pose head，再以新 optimizer 啟動 J1。α 保持可訓練，PWL [-10,0]、20 段。後續先验收融合，再 activation → 方向 2；尚未建立後兩階段 queue。下方所有等待／舊 J0 執行中敘述都是歷史，舊 `plan.json` 仍停用以防重跑；目前權威計畫為 `bridge_v1/plan.json`。見[工作紀錄](<../../worklogs/2026-09-10-bridge-combine-restart.md>)。

## 目前狀態：直接驗證完成，停止等待 MASF 決定

依使用者最新要求，不再重訓，直接使用七個既有 P2／P3 Detect checkpoint 在 canonical Pose val 評估 ball／bat box，並對正式 Pose checkpoint 做原 MASF 開／關消融。完整 COCO 與 BBAT5 結果：RESULTS.md（本機／歷史參照：`../../../combine/pose-masf/RESULTS.md`；未隨本次報告發布）。無 MASF J0 已完成；新 MASF 正式訓練未開始，queue 已退出且不重啟。P2 Pose 訓練準備稿未完成驗證，不可視為可用訓練流程。`plan.json` 訓練停用；不接續其他階段。下方是先前工作歷史。

## 最新接續：無 MASF J0 已啟動

**最新停止點：** Pose MASF 成對比較完成後，合併 COCO 與 ball／bat 結果呈現，停止等待使用者決定是否採用 MASF。`run_pose_pair.py` 已排定：等現有 control 完成，只接 MASF smoke／J0 candidate；不接 J1/J2 或其他研究。

P2 最後試驗未達門檻，選 control E2 EMA 作新 Detect；來源 SHA256 見 selected-source-v1.json（本機／歷史參照：`../../../combine/artifacts/selected-source-v1.json`；未隨本次報告發布），該檔保存選定當時狀態，後續驗證見 baseline-v1（本機／歷史參照：`../../../combine/artifacts/baseline-v1/summary.json`；未隨本次報告發布）。原 graph audit 與 CPU 等價已通過；完整 COCO overall／person 保持 0.508330／0.627858，但新共享初始 Pose AP 為 0.584586，低於原獨立 Pose 0.912161，尚不能稱為融合成功。

正式 J0 `j0-no-masf-v1` 已啟動：沿用原 combine 的 8 epoch、AdamW、head LR 2e-4、warmup 1、Pose batch 16 與八項 AP／checkpoint 選擇。只更新 Pose head，Detect／trunk／共享 BN 固定；Pose-only EMA 保留固定權重。GPU smoke 兩 batch 通過，峰值已配置顯存 2,980,215,296 bytes。每 epoch Float／BitTrue 完整驗證，J0 COCO AP 如非原值即報错。完成後先分析，不直接解凍。

新使用者要求的 Pose MASF 專項（本機／歷史參照：`../../../combine/pose-masf/README.md`；未隨本次報告發布） 與此主線分開；尚未開始成對訓練。所有新 BBAT5 工作使用 canonical v1，未修改舊專案或資料。完整理由與困難見[工作紀錄](<../../worklogs/2026-09-10-p2-result-no-masf-fusion.md>)。下方是此前準備階段的歷史狀態，不代表目前仍等待選擇。

最新指示：先做最後P2 MASF試驗；若無足夠收益則放棄MASF接融合，不再等待使用者二選一。此處繼續等待P2結果及適配驗收，不啟動GPU。見[P2實驗紀錄](<../../worklogs/2026-09-10-p2-masf-last-trial.md>)。下方原選擇等待為歷史狀態。

目前狀態：**等待選定 Detect 候選，尚未組裝或訓練**。本資料夾不修改 `/home/uxin/yolo/yolo_combine/`，也不接續覆寫其既有 run。方向 1 的結果見 集中報告（本機／歷史參照：`../../../reports/direction1-20260910/README.md`；未隨本次報告發布）。

## 已完成的 CPU 稽核

執行 `CUDA_VISIBLE_DEVICES=-1 /home/uxin/yolo/yolo_combine/.venv/bin/python combine/audit_sources.py`，兩個方向 1 候選的 SHA256 與完整驗證摘要相符。使用原 joint.yaml 指定的正式 Pose P3 checkpoint，在記憶體將 normalization 對齊 Bit-True PWL，再呼叫原 `audit_task_pair`。

兩候選皆 `compatible=false`，差異全部位於 layer16：module type、parameter count、parameter signature、buffer signature。舊 Pose 含 shared MASF；新 Detect 的 MASF 已移到 Detect head，或完全不含 MASF，因此不能直接使用原 queue。完整來源雜湊與差異保存於 source-audit-v1.json（本機／歷史參照：`../../../combine/artifacts/source-audit-v1.json`；未隨本次報告發布）。CPU 稽核不等於融合 smoke 或 AP 驗收。

原 stage policy 先把 `.detect_head.` 分為 Detect head，再辨識 `.p3_masf.`；Detect-only MASF 的分組會與舊 shared MASF 不同。此外梯度橋接類別要求 MASF BN 統計固定，不能直接套用舊 head BN train 政策。後續必須明確決定這些行為並驗證，不能因可載入權重就宣稱可訓練。

## 接續順序

1. 選定方向 1 的 Detect 起點；兩候選均保留，不擅自移除 MASF。
2. 在本資料夾建立融合適配器，以選定的 Detect trunk／head 與正式 Pose head 建構；保留舊獨立 Pose 作回退及比較，不宣稱換 trunk 後 Pose 預測等價。
3. 完成 graph、載入完整性、Detect 初始化前向等價、Pose 初始完整指標、optimizer 分組、BN 與 loss 的 CPU／GPU smoke 驗證。
4. 參考原 J0 Pose-head 適應 → J1 Neck／heads → J2 後段適應；實際設定與新 baseline 確認後才建立可執行 queue。原先的 -0.08 gate 不直接沿用為本輪 COCO 保護標準。
5. 融合完成與驗收後才處理 activation；方向 2 在其後。目前這兩階段都未啟動。

`plan.json` 是明確停用的準備規格，不是正在運作或保證自動接續的 queue。GPU 工作一旦獲得完整前置條件，沿用最多 600 秒 blocking monitor；沒有 GPU 工作時不建立空監測。

## 資料與保留規範

COCO80 使用原完整 COCO2017；Pose 只使用 `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml`，由 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml` 定義。不得重切、抽樣或改標註。此處未建立 dataset View、未複製權重、未啟動 GPU、未修改舊模型。

輸入、輸出與執行證據保存於本資料夾；程式及中文文件可供日後發布，checkpoint／cache／訓練輸出不是預設 Git 提交內容。本次沒有 commit 或 push。
