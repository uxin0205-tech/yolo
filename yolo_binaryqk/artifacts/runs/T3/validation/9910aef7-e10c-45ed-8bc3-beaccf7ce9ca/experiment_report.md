# T3 BinaryAttention run

- 目的：YOLO11m BinaryAttention accuracy/fidelity experiment。
- 相對前一 variant：`T2`；本 run 改動為 `ParallelDualBinaryAttention` / `dual`。
- dataset：COCO2017 full train2017/val2017；epochs：0；batch：128；seed：0。
- initialization：`source checkpoint validation`；paper profile：`BinaryAttention QAT fine-tuning`。
- checkpoint/metrics weights：`source checkpoint` / `source checkpoint`。
- config hash：`9a95c24f2a195beb1a90be662e10a570706a81c616323025a774b914c491c88c`。

## Attention / loss

- QK：`dual`；QAT：`True`；bias：`none`。
- quantization contract：`{"bias_initialization": "none", "bias_parameterization": "none", "p8_scale": "none", "qk_scale": "residual_basis_channel_per_token", "v8_scale": "none"}`。
- KD：`None`，components=`[]`。
- T6 KD target family：`None`；N4 non-KD parent：`not applicable`。
- 理論成本：binary QK=2、softmax=2、PV=2。
- T3/T4/T5 對應 parallel/full-basis/matched-basis dual attention；N4 的 magnitude rank-1 term 真正加入 score。

## Metrics

- mAP50_95: not available
- mAP50: not available
- mAP75: not available
- mAPs: not available
- mAPm: not available
- mAPl: not available

## Diagnostics

- {"fake_quantization": false, "hardware_speed_claim": false, "modules": [{"binary_forward_count": 1, "binary_qk_count": 2, "class": "ParallelDualBinaryAttention", "path": "", "probability_finite": true, "probability_shape": [2, 2, 16, 16], "pv_count": 2, "score_finite": true, "score_shape": [2, 2, 16, 16], "softmax_count": 2}], "theoretical_cost": {"binary_qk": 2, "pv": 2, "softmax": 2}, "variant_id": "T3"}
- Attention-only delta proof：{}
- G0–G5：未執行（依本計畫明確排除）。
- 所有 T/N runs 從 full-precision checkpoint 直接做 10-epoch attention-only fine-tuning；非-attention parameters 凍結。
- 限制：P8/V8/INT4 是 PyTorch fake quantization；不得解讀為真實 1-bit 硬體速度提升。

結論：`numerical self-check only`
