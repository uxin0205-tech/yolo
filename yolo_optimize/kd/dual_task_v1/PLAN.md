# 核准後的雙教師 KD：第一個配對實驗

2026-09-11 使用者核准。學生固定為 qSiLU P3 bridge E2；原 R2-REGION「單 FP-QK teacher、QK pair 排名」規格保留但不再當本次實作說明。此次改為 COCO Detect／BBAT5 Pose 分流教師，先測普通跨尺寸 KD，不宣稱創新已成立。

## 已完成的前置

YOLO26L Detect 與原獨立 BBAT5 Pose 教師均通過完整同口徑驗證，八項 AP 都高於學生。來源 hash、結果見 `artifacts/teacher-validation-v1/`；資料固定 COCO train118287／val5000 與 canonical BBAT5 train5964／val683，未改 split。

MuSGD 新學生校準已完成 AdamW／MuSGD 初探各16 macro、相同 train trace；attention 實際更新中位數為0，角色 LR 比值無法有限校準，拒絕此次配方。確認階段未做 GPU 更新，保留兩份成功 probe 及原失敗事件。主線使用 AdamW，不把結果解讀成 MuSGD 普遍無效。參考 [官方 MuSGD 配方](https://docs.ultralytics.com/guides/yolo26-training-recipe/)；本機已有官方 optimizer，並非無支援。

## 普通 KD 訊號

以 [Attention Transfer](https://arxiv.org/abs/1612.03928) 的通道聚合空間圖作跨尺寸基線：

```text
COCO batch  → YOLO26L teacher P3/P4/P5 ─┐
            → joint student Detect    ├→ native Detect + μD × spatial KD
                                      │
BBAT5 batch → 原獨立 Pose teacher     ─┐│
            → joint student Pose     ├┴→ native Pose + μP × spatial KD
                                      ↓
                          既有 macro task 權重與一次正規化
```

每尺度 `A(F)=normalize(mean_channel(F²), spatial L2, eps=1e-6)`；每圖為三尺度平方距離空間總和的平均，task raw loss 加 `μ×sum_images(KD)`，由原 macro engine 做既有正規化。平方、聚合與 loss 使用 FP32，native forward 使用 AMP。

固定對齊 layer16／19／22（P3/P4/P5）空間位置，通道數可不同；不插值、不配不同 attention heads。學生 P3 tap 在 Detect bridge MASF 之前，原 MASF 接線與原生任務梯度不變。這不是 QK score loss，**不需要啟用先前測試的 surrogate 或改 BinaryQK 算式**。教师保持各自已驗證算式，不強制都 FP-QK；原 Pose 教師的 Float PWL 與學生 Bit-True 驗證不能混稱全 FP teacher。

每張影像只使用對應教師；不交叉 class IDs、不補造另一任務標註。教師 eval／no_grad，暫存 hook 用完移除，無可訓練 projector，教師不進學生 optimizer、EMA、state_dict 或匯出。訓練會增加教師 forward 成本，不能宣稱訓練加速。

## 固定設定與驗收

K0 與 K-SPATIAL 均由同一 qSiLU E2 開始、seed1、AdamW、各5 epoch、patience0、warmup1、cosine final0.5、betas0.948／0.999、WD0.00027、clip10。LR：backbone3.8e-7、Neck1.9e-6、MASF3.8e-6、合法 attention5e-8、雙 heads5e-6。Detect logical128／physical16，每 macro256 Detect＋16 Pose，task weight1／0.25。

共享 BN running 與 MASF BN 固定，affine 沿 activation 設定可訓練；Q/K 權重、gamma、PoT scale、PWL [-10,0] 20段固定。固定 μD=0.8214335621、μP=1.7886982661，來自各4個真實 train batch 的共享 Neck feature 梯度比目標0.1；不是每圖動態倍率，也不是全參數梯度比。μ 不隨 epoch 改變，warmup1 指 optimizer warmup。

CPU 通過 live gradient、teacher detach、不同 channel 等價、microbatch loss／gradient 等價、零特徵及空間錯位拒絕。GPU 校準證明 KD-only 可更新共享 Neck，教師 state 不變。兩臂各2個真實 macro 通過 AMP、固定參數／EMA、MASF BN；KD 新增 criterion metadata 的 weights_only 保存／載入一致、教師無梯度、hooks 清除。峰值 allocated：K0 19.154GB、KD 19.758GB。未宣稱任意時刻強制中斷 resume 或注入 AMP overflow 已另做完整端到端測試；沿用原 trainer。

候選相對本次學生八項 AP 的下降門檻0.001；同時列出原融合 gate，不能重設基準掩蓋舊差距。任何指標比本次起點下降超過0.05，先保存再停止。先比較 matched K0／KD 的 AP 與完整曲線；單 seed／五輪只作初篩，不保證追平教師。普通 KD 有訊號才延長／第二 seed，再評估區域與有效 keypoint 的創新配額，不預先堆疊新 loss。

## 執行與監測

`run_training_pair.py` 先 K0，再 spatial；每次 child.wait 最多600秒，正常不讀 log 或重複查 GPU。非零退出／未完成不啟動下一臂；單 job 完成確認 summary 後自動接續。全部完成後比較及重驗選定輸出。保留所有來源與 probe，不覆寫舊 run，不發布或刪除。
