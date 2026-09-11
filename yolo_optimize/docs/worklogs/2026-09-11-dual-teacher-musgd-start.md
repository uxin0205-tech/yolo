# 2026-09-11：核准雙教師 KD 與 MuSGD 前置

## 最新完成狀態

MuSGD：AdamW／初探各16 macro 完成且 trace 相同；attention 更新中位數為0，原確認腳本 assert 中止。已將此數值邊界改成明確 recipe_rejected，只重啟失敗確認步驟做 CPU 判讀，confirmation_updates0；不重跑正常兩臂，不私調 LR。

KD：CPU 空間 loss 契約及真實 train-only 校準通過，μD0.8214335621／μP1.7886982661；KD-only 確實更新共享 Neck、教師 hash 不變。K0／KD 各2個完整 macro 通過，fixed live／EMA 及 MASF BN 不變；KD safe state roundtrip、教師無梯度與暫存 hooks 清除通過。峰值 allocated 分別19,154,190,336／19,757,864,448 bytes。以新 [PLAN](<../../kd/dual_task_v1/PLAN.md>) 為核准後規格，接續 AdamW 五輪配對；正式結果尚未產生。

## 變更與原因

使用者同意改為任務分流雙教師，並詢問 MuSGD。保留原 R2-REGION 單 joint teacher 規格作歷史提案；新增 `kd/dual_task_v1/teachers.py`、`experiment.py`、`probe_optimizer.py` 與序列監測。主代理依 research 技能查閱本機程式及官方來源，不使用子代理。

## 教師驗證

官方 YOLO26L：`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt`，SHA256 `9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26`。靜態檢查 GLOBAL 後以明確 allowlist、hash 與 weights_only=True 載入。Pose 沿用已稽核的原独立權重。

完整 COCO val5000／canonical BBAT5 val683 通過。教師相對學生：COCO +0.033698、person +0.014724、BBAT box +0.013151、pose +0.020872、ball box +0.006245、ball pose +0.016713、bat box +0.020057、bat pose +0.025031。原始資料見 `kd/dual_task_v1/artifacts/teacher-validation-v1/summary.json`；不是 KD 增準結果。

## MuSGD 校準與方法界線

[官方配方](https://docs.ultralytics.com/guides/yolo26-training-recipe/) 使用 MuSGD。本機 stage_policy 已支援 Muon0.2／SGD1.0 語義分組。新 scope 必須重新校準：三臂各16 macro，同學生／seed1／原完整 train loader、physical16／logical128、warmup1，不以 val 調 LR、不建立新 split。AdamW 對照、MuSGD 初探、一次角色 LR 確認；各有效梯度組更新中位數須落在 AdamW 的0.5–2倍。不過則保留 AdamW；更新均丟棄，不把 AdamW state 搬到 MuSGD。舊 heads-only 前置失敗不能外推新 scope。

跨尺寸先研究通道聚合的空間 [Attention Transfer](https://arxiv.org/abs/1612.03928)，不直接配不同模型 attention heads。保留每個任務 native loss；僅對應資料使用對應教師。這是普通 KD 對照候選，不是原 BinaryQK ranking、不是創新已成立。先全模型梯度／AMP／保存契約，再 K0／KD，之後才區域創新。

## 困難與未解

局部 docs/agents 不存在，已讀父 repo 指引及 CONTEXT；不另建副本。現有檔案 apply_patch 遇到 sandbox bwrap 錯誤，改用有範圍說明的提升權限 apply_patch；不改權重。教師驗證無困難。MuSGD 新前置與 KD 訓練尚待完成。所有 GPU 工作600秒事件式 blocking monitor，正常只等待。舊結果與資料保留，無刪除、commit 或發布。
