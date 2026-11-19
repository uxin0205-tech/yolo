# 第 7 階段：Activation 比較與 qSiLU 主線

2026-11-03 01:00:29：本階段已完成四組各 10 輪、12 筆區域驗證（其中 2 筆無可替換位置）與四組重載完整 AP。現行入口為 [結果報告](../result/stage7/README.md) 與 [可攜交接](../handoff/README.md)。RepConv 未啟動。下文保存事前 protocol 與已執行流程。

2026-11-02 使用者已授權開始本階段，包含 GPU queue、候選恢復訓練與完整驗證。共同 parent 為 [Stage6 E1](../../yolo_combine/result/stage6/README.md)，SHA `e9b9fd7ccc88f10bbd170835138ebde5c7810db0726424496e2cac2aa462ffe7`。原融合四項獨立精度目標尚未達到的事實保持記錄。

本輪比較 qSiLU、Hardswish、PolyShift、PolyQuality。四組分別從同一 E1 出發，不重跑 SiLU control；比較選定與 RepConv 接續模型分開，後者固定 qSiLU。完整配置、來源 SHA、預算與停止規則見 [protocol.json](protocol.json)，原 optimize 核對見 [reference-review.json](reference-review.json)。

## 已完成的前置工作

先唯讀核對 optimize activation bridge 的程式、實際 10 輪 run、parent、結果及上游 J3；PolyQuality 另外核對舊 activation Full35 完整配方與結果。舊 Hardswish 直接替換退化，但訓練仍能恢復，因此四組都排同樣的恢復機會。

已刪除 226 個不再使用的舊 checkpoint 或大型預測檔，釋放 26.137 GiB，未封存。量化等外部依賴仍使用的 4 份權重、release 唯一權重、共用 `activation_lab`、小型結果與配置保留。見 [逐檔清理紀錄](cleanup-20261102.json)。`yolo_optimize` 未修改。

數值診斷見 [numerics.json](numerics.json)：公式／分段／導數／尾端、FP32／FP16／BF16，以及 qSiLU／PolyShift signed16 Q10 全 65,536 codes。這是局部定點 reference，任務模型 activation 仍用浮點參考，未宣稱整網定點或硬體加速。

CPU 檢查見 [cpu-contracts.json](cpu-contracts.json)：每組保留 187 個 activation 路徑與 30 組共享物件，參數量 26,453,936；雙任務／單任務 routing 輸出與 checkpoint 重載輸出逐值相同，並確認錯誤的 activation 類別／qSiLU 係數／reciprocal 常數被拒絕。163 個部署位置與 24 個 RealNVP loss 專用位置分開記錄。

## 已完成的 GPU queue

1. 四組完整零樣本 AP，同時記錄實際 activation 輸入及兩個 Attention sites 的 score／tail／row-sum。
2. 每組以 train macro 校準 Detect／Pose 梯度比例，再做 2 個實際 AMP 更新 smoke；smoke 權重丟棄，正式訓練重新載入共同起點。
3. 四組各最多 10 輪：完整 COCO train；Detect physical16 ×16、Pose16、reference64。AdamW、warmup1、cosine final0.5；backbone／neck／heads LR 為 3.8e-7／1.9e-6／5e-6。shared BN running 固定、affine 可訓練，head BN 可更新，Attention 全部固定。CIoU 沿用 optimize FP32 包裝，公式不變。
4. qSiLU 若未過相對門檻，但恢復確有進步，從其最佳權重降 LR 一半，最多再排 5 輪；fresh optimizer／EMA、重新校準。
5. qSiLU 與另一個較佳零樣本候選，各 6 個區域單獨替換、完整 AP sensitivity。所有區域候選仍從 E1 出發，不累積替換。
6. 每組選定權重重新載入、重驗完整 AP，輸出比較與 qSiLU 交接。完成後停在 Stage7，不啟動 RepConv。

最多 45 個訓練 epoch，另含零樣本／敏感度／逐輪／重載驗證。比較門檻為目前 E1 每項 AP 最大下降 0.015；原獨立模型門檻另列，不能混稱恢復成功。每輪保留最佳，較差末輪不覆蓋；數值非有限、契約改變或較候選零樣本再退化超過 0.05 時停止處理。

資料與 evaluator 共用已稽核 adapter：完整 COCO val 5,000／BBAT5 v1 val 683、FP32 E2E、batch8、conf0.001、iou0.7、max_det300。正式 BBAT5 v1 不改 split／來源／標註。只重用 canonical runtime View，新模型、queue、報告全部在 `yolo_activation`。

B2 二值 QK、16 個 S1 Q10 scale、signed16/1024 分解式 bias、PWL `[-10,0]`／20 段、LUT256 reciprocal、UQ1.24 probability 和 max_tokens1024 必須保持。浮點反向代理與部署前向分開；不增加最終浮點除法強制 row-sum 為 1。

```bash
# CPU 檢查
bash yolo_activation/run_stage7.sh cpu-check
# 背景 queue 使用此入口（避免重複啟動）
bash yolo_activation/run_stage7.sh run
# 即時進度
bash yolo_activation/run_stage7.sh status
```

Queue 位於 `artifacts/stage7/activation-20261102/`，每 600 秒保存 GPU、進度與來源鎖監測，每 5 秒確認 worker 是否完成；發現其他 compute process 時釋放本輪 GPU，保留最後完整 epoch resume，等待空閒後接續。`STOP` 檔可停止 queue；沒有自行跨階段的路徑。穩定長訓後回報實測 ETA 並退出 working。

[中文工作紀錄](../../docs/worklogs/2026-11-02-activation-stage7.md) 持續區分 CPU smoke、正式 AP、訓練、局部定點與未做的硬體量測。已產生 `result/stage7/` 的比較 CSV、曲線／敏感度、無 GPU 重畫程式與 qSiLU handoff。
