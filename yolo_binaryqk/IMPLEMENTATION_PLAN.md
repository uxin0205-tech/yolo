# YOLO11 BinaryAttention corrected attention-only QAT plan

本計畫保留 T0→T1→T2→T3→T4→T5/N 完整實驗矩陣，並依最新要求將正式訓練
改為 10 epochs、只更新 attention parameters，其餘 YOLO11m parameters 全部凍結。

## E 系列：零訓練

- E0：FP attention。
- E1-S：sign-only QK。
- E1：scaled-sign QK。
- E2-DUAL：使用提出的 matched residual dual basis；2 個 binary QK、1 softmax、
  1 PV，直接轉移 FP checkpoint 中可匹配 tensors，只做 COCO validation，沒有
  optimizer、沒有 QAT training。

## T/N 正式設定

所有 T/N variants 使用完整 COCO2017、640px、effective batch 128、seed 0，從
同一唯讀 FP checkpoint `../original/weight/yolo11m.pt` 獨立初始化。binary variants
啟用 clipped-STE QAT；KD variants 使用同一 frozen FP teacher。

「獨立初始化」只指 weights；T3/T4/T5 分別是 parallel、full-basis residual、
matched-basis residual dual attention，全部不使用 KD。T6 先選出 T1–T5 中
mAP 最佳的基底，再測 positional、feature、attention KD 及兩兩組合，最後選出
最佳 T6。T7 在選定 T6 上測 dense/decomposed relative-position bias 與 P/V
fake quantization。N4 從 T7+T1–T5 選 parent，但 N4-FP/I8/I4/PV 本身全部
不使用 KD。

```text
epochs=10, trainable_scope=attention_only
optimizer=AdamW, lr0=5e-5, lrf=0.1, cos_lr=True
weight_decay=0.02, effective_batch=128
micro_batch=16, accumulation=8, imgsz=640, amp=True
```

10 epochs 位於要求的 10–15 範圍下限。論文原始 300-epoch recipe 僅作參考；
本研究結果必須標示為 YOLO attention-only adaptation，不冒充 300-epoch paper run。

## 執行順序

```text
E0/E1-S/E1（已完成） + E2-DUAL zero-training validation
  ↓
T0/T1（已完成）→ T2（接續，10 epochs，attention-only）
  ↓
T3/T4/T5（parallel/full-basis/matched-basis dual；無 KD）
  ↓
選 T1–T5 最佳 → T6-O/F/A、T6-O/F、T6-O/A、T6-F/A → T6
  ↓
T7-D/R → 選最佳 bias → T7-P/V/PV
  ↓
選 T7+T1–T5 parent → N4-FP/I8/I4 → 選 magnitude → N4-PV（無 KD）
  ↓
後段比較、strict reload、final report、fail-closed audit
```

## 凍結與 KD 契約

optimizer 建立前，trainer 將全模型 `requires_grad=False`，只重新啟用 concrete
`Attention` module 內的 QKV/proj/PE、bias、threshold、basis、magnitude 等參數。
artifact 保存 trainable parameter names/count 與 frozen count，audit 要求所有
trainable names 都位於 `.attn.` 路徑。
finalizer 另逐 tensor 比較 epoch EMA 與 FP source：所有非-attention parameters 與
非-attention BN buffers 必須維持零差異，且至少一個 source-initialized Attention
tensor 必須實際改變，結果保存於 `parameter_delta_diagnostics.json`。

每個 epoch 的 COCO validation 使用 Ultralytics EMA；正式 `strict-last.pt` 同樣保存
epoch-last EMA，artifact 明確記錄 `checkpoint_weight_source=metrics_weight_source=epoch_ema`，
避免 checkpoint 與報告 AP 對應到不同權重。

T6 positional KD 使用 attention module 的 PE outputs；feature KD 使用 attention
module outputs；attention KD 使用真實 student/FP teacher attention maps。teacher
不回傳梯度，normalization 維持 eval。

## 完成證據

每個 E validation 與 22 個 T/N full run 都必須具有 resolved config、training args、
environment、architecture/checkpoint manifest、COCO metrics、diagnostics、curves、
attention-only parameter delta proof、formal log、strict checkpoint 與 experiment report。最終 selection JSON、summary
JSON/CSV、Markdown report 與 figures 均由 artifacts 自動產生並通過 audit。
