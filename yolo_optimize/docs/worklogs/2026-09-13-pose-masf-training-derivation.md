# 2026-09-13：Pose MASF 專項重訓推導與完整架構圖

## 變更內容與原因

依使用者要求，只推導 Pose MASF 專項訓練與繪製完整架構，不啟動 GPU。新增 experiments/pose_masf_training_v1，保存主計畫、proposed-training.json、architecture.json、兩張 DOT／SVG／PNG 圖與 CPU 解析梯度核對。既有 Pose 移接成果、native_qk E2、scale_bias queue 與原權重不更動。

提案只做兩個必要組：Pose head-only 對照、Pose head＋獨立 P3 MASF，各 5 epoch、warmup 1；固定 backbone／neck／Attention／Detect／Detect MASF。B 的 context 複製既有參數但 α=0，以相同前向起點分離「單純加訓」與「新增 MASF 的收益」。這不是已取消的 BinaryQK 對照。

## 推導重點

區分前向 gate α 與反向 bridge β。MASF 的有效梯度為 λm gm＋βλo go；context 梯度再乘 α，α 梯度則不乘 α。Pose one2one head 自身梯度不受 β 縮放。固定共享特徵後，提案 β=1，避免原 Detect β≈0.0121 對 Pose 部署分支的監督過度壓縮；這是待測初始設定，不是最佳值結論。

核對原生 E2ELoss：fresh 5 epoch 的 one2many／one2one 權重為 0.8/0.2 → 0.625/0.375 → 0.45/0.55 → 0.275/0.725 → 0.1/0.9，每回合結束更新一次。核對固定權重 reg_max=1、keypoint shape=[2,3]；本機 DFL-free 路徑的 regression slot 是 LTRB L1，不把 loss_dfl 欄位誤當成必定為零。

## 驗證方式與結果

CPU forward 追蹤 layer 0–22 全部輸出、layer 23 輸入 [16,19,22]。P3/P4/P5 分別為 256×80×80、512×40×40、512×20×20；Detect nc=80、Pose nc=2；Attention 在 layer10 與 layer22 內。來源契約與維持 PWL [-10,0] 的定義一致。

CPU float64 小型可微模型的 5 項推導檢查全部通過：context chain rule、α chain rule、head 梯度不被 β 改變、α=0 時 context 任務梯度為零、α 更新後 context 任務梯度出現。這不是 BBAT 真實 loss、AMP 或 YOLO 訓練驗證。

Graphviz 依真實接線產生完整架構與訓練梯度 SVG／PNG，已目視核對。圖表、JSON、梯度公式與主計畫相互對照；不把新提案標成已實作。

## 困難與解法

封存候選程式的 β 上限為 0.25，所以 β=1 必須在新訓練實作中明確支援；沒有直接修改舊程式或只改設定就宣稱生效。

既有 PoseEpochRunner 每個 physical batch 就更新一次，未支援累積 8 次；文件與 JSON 明列新 runner 的必要改動，才能達成每 epoch 47 個更新、最後 76 張的有效 batch 128。Pose-only 的任務權重會抵消，不能誤以為沿用 pose_weight=0.25 就是原強度。

圖片檢視工具受到本機 bwrap 錯誤限制，改由 DOT 產生本機 JPEG 預覽供目視檢查，沒有外部上傳。其他困難：無。

## 未解事項與風險

尚未建置正式 GPU 訓練 runner、未測真實 Pose loss 的 α 啟動、β=1 穩定性或所列 LR。全部 JSON 明示 design_only、gpu_training_enabled=false。原 Attention 與 scale_bias 仍暫停；沒有 commit／push 或刪除。實作與訓練必須另依使用者執行指示開始，不能將本次文件當成已排 GPU 工作。
