# 2026-09-10：Pose head 適應結果與完整 Pose AdamW 啟動

## 已完成結果

`j0-pose-extend-v1` 正常完成 40 epoch、14920 macro，未觸發 patience。最佳是該 run 第 37 epoch（zero-based 36），Pose AP 從 0.807073386 提升到 0.852474613，box AP 從 0.514842332 提升到 0.567273450。ball box／pose 為 0.475850818／0.829325532，bat box／pose 為 0.658696082／0.875623694。COCO overall／person 每輪保持 0.508211955／0.627664127。

仍低於原独立 Pose AP 0.912160548，bat box 差距仍超過原 0.08 gate；未宣稱適應完全或融合成功。原先只跑 J0 8 epoch 確實不足以判定收斂，但完整 Pose 仍需進一步適應。

## 新完整 Pose 分支

新增 `combine/bridge_v1/full_pose.py`，來源是上列 best Pose head，SHA256 `1b9803b6cfaacc0dad59ecd3fcffaea976bc50c45d828be5981323c941047b05`，使用安全 state-dict 載入。原 P3 bridge Detect checkpoint 不改，原結果均保留。

本分支以相容的圖容器保存獨立 Pose：僅 Pose 資料與 loss 訓練 backbone 全段卷積、Neck、Pose head；Detect head 不更新。沒有 Detect loss，不能称為聯合融合訓練。圖內凍結 Detect head 接到已適應 Pose trunk 的 COCO 指標只供相容性診斷，不代表原独立 Detect 退化，也不是可直接部署的新融合模型。

AdamW：backbone LR 1.5e-5、Neck 7.5e-5、Pose head 2e-4，betas 0.948/0.999、weight decay 0.00027、clip 10。最多 60 epoch，warmup 1，cosine final factor 0.5，Pose batch 16、imgsz 640。全 canonical BBAT5 train 5964／val 683 張，沒有重切或抽樣。

硬體固定 attention、Q/K、PoT 與 [-10,0] 20 段 PWL 仍保留；shared BN 統計及 affine 固定。這是完整 Pose 主幹卷積／Neck／head 適應，不是解除全部硬體契約。MASF 是 Detect-only 分支，此次不接入 Pose、不更新 α。

使用 PoseModelEMA，只混合實際可訓練參數與 Pose head 浮點狀態，固定 Detect／MASF 不參與 EMA 算術。原 ActiveEMA 會混合 Detect head，為避免長期固定狀態漂移而換成局部 EMA；未改 J1 或舊結果。

## 驗證與監測

v1 真實兩 batch smoke 通過；EMA 政策變更後另跑 v2，驗證 backbone layer0、Neck layer16、Pose head 確實更新，固定 live／EMA 參數、Detect 全部 state、硬體契約保持。額外檢查 updates=4000 的 EMA decay 區間。結果保存在 `artifacts/fusion/full-pose-smoke-v2/summary.json`。

正式 `full-pose-adamw-v1` 已啟動，入口先核對 v2 smoke。每 epoch 完整 Float／BitTrue 驗證；任一六項 BBAT AP 比本次起點下降超過 0.03 時保存停止，六項平均分數 min_delta 0.0001、patience 12 後保存停止。COCO 只作圖內共享相容性診斷；原八項 gate 仍列出，但本分支不以 `best_joint` 升格，後續須独立驗收。事件檔 `combine/artifacts/logs/full-pose-adamw-v1.events.jsonl`，600 秒 blocking monitor。

## 困難與未解

圖容器同時保存兩 head，但此階段只有 Pose 訓練，因此必須明確區別独立 Detect 錨點與圖內診斷 COCO；已在 resolved config 標記，不把新圖當成融合完成。完整 Pose 自適應能否回到原指標、如何保持其 trunk 收益再融合，仍待結果決定。沒有刪除或覆寫來源，沒有 commit／push。
