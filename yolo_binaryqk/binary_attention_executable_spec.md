# YOLO11 BinaryAttention executable specification

版本：6.0（corrected 10-epoch attention-only QAT matrix）

## Matrix

零訓練 E variants：E0、E1-S、E1、E2-DUAL。E2-DUAL 是 matched residual dual
basis（2 binary QK / 1 softmax / 1 PV），不建立 optimizer。

正式 T/N variants：T0、T1、T2、T3、T4、T5、六個 T6 KD candidates、selected T6、
T7-D/R/P/V/PV、N4-FP/I8/I4/PV。每個 run 均為 full COCO、10 epochs、
attention-only fine-tuning。

Weights 均從相同 FP source 獨立初始化。T3/T4/T5 對應原先 N1/N2/N3，全部
不使用 KD。T6 先從 T1–T5 選最佳基底，再測 positional/feature/attention KD
及兩兩組合；T7 在選定 T6 上測 bias 與 P/V quantization。N4 從 T7+T1–T5
選 parent，但 N4 全部不帶 KD。scaled binary Q/K 每個 sample/head 使用跨 channel 與 token 的單一 mean-absolute
scale；V8 按 channel 跨 token 取 scale；完整 2D bias 以 truncated-normal(0, 0.02)
初始化，與論文釋出實作一致。
上述 tensor-axis 與初始化語意必須寫入 resolved config、architecture manifest 與
checkpoint manifest；續跑和 final audit 均 fail-closed 驗證，不得僅依賴 variant hash。

## Optimization and freeze

所有 run 從相同 FP source 獨立初始化。使用 effective batch 128、AdamW、
`lr0=5e-5`、cosine `min_lr=5e-6`、weight decay 0.02、640px、seed 0、AMP。
trainer 在 optimizer construction 前凍結全模型，只允許 concrete Attention
module parameters 訓練。status/training/architecture artifacts 必須記錄並證明
trainable scope、parameter names/count 與 frozen count。

## KD and validation

positional、feature、attention KD 使用彼此獨立且正確的 tensors；attention KD
比較真實 attention probabilities。FP teacher 永遠 frozen/eval。每個 completed
run 必須執行 COCO validation、binary-path counter 檢查、strict checkpoint save/
reload 與 manifest verification。
T/N 的 strict checkpoint 與 COCO metrics 必須同時來自 epoch-last EMA，不得以 raw
training model checkpoint 搭配 EMA validation metrics。

## Completion

```bash
cd yolo_binaryqk
PYTHONPATH=. ../.venv/bin/python -m pytest -q tests
./run_formal_plan.sh
PYTHONPATH=. ../.venv/bin/python -m binary_attention.cli audit
```

audit 只有在 4 個 E artifacts、22 個 T/N artifacts、selection、summary、figures
與 final report 全部存在且一致時才通過。
