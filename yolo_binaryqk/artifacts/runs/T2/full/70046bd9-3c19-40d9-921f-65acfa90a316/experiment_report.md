# T2 BinaryAttention run

- 目的：YOLO11m BinaryAttention accuracy/fidelity experiment。
- 相對前一 variant：`T1`；本 run 改動為 `ScaledBinaryAttention` / `scaled_sign`。
- dataset：/home/hplab/uxin/yolo11/yolo_binaryqk/data/coco_full.txt；epochs：10；batch：128；seed：0。
- initialization：`full-precision source checkpoint fine-tuning`；paper profile：`10-epoch attention-only QAT fine-tuning`。
- checkpoint/metrics weights：`epoch_ema` / `epoch_ema`。
- config hash：`ed7a5b51d8e62db06315774b14190551896596eecca3e952dc6954f528aaec08`。

## Attention / loss

- QK：`scaled_sign`；QAT：`True`；bias：`none`。
- quantization contract：`{"bias_initialization": "none", "bias_parameterization": "none", "p8_scale": "none", "qk_scale": "mean_abs_channel_token_per_sample_head", "v8_scale": "none"}`。
- KD：`None`，components=`[]`。
- T6 KD target family：`None`；N4 non-KD parent：`not applicable`。
- 理論成本：binary QK=1、softmax=1、PV=1。
- T3/T4/T5 對應 parallel/full-basis/matched-basis dual attention；N4 的 magnitude rank-1 term 真正加入 score。

## Metrics

- mAP50_95: 0.50346
- mAP50: 0.67122
- mAP75: None
- coco_mAP50_95: 0.5095113365809676
- coco_mAP50: 0.6784860403112603
- coco_mAP75: 0.5540973150918809
- mAPs: 0.3320121727113555
- mAPm: 0.5631109296186375
- mAPl: 0.6699683237088675

## Diagnostics

- {"fake_quantization": false, "hardware_speed_claim": false, "modules": [{"binary_forward_count": 3131, "binary_qk_count": 3131, "class": "UltralyticsBinaryAttention", "path": "model.10.m.0.attn", "probability_finite": true, "probability_shape": [8, 4, 231, 231], "pv_count": 3131, "score_finite": true, "score_shape": [8, 4, 231, 231], "softmax_count": 3131}], "theoretical_cost": {"binary_qk": 1, "pv": 1, "softmax": 1}, "variant_id": "T2"}
- Attention-only delta proof：{"changed_source_attention_parameter_names": ["model.10.m.0.attn.qkv.conv.weight", "model.10.m.0.attn.qkv.bn.weight", "model.10.m.0.attn.qkv.bn.bias", "model.10.m.0.attn.proj.conv.weight", "model.10.m.0.attn.proj.bn.weight", "model.10.m.0.attn.proj.bn.bias", "model.10.m.0.attn.pe.conv.weight", "model.10.m.0.attn.pe.bn.weight", "model.10.m.0.attn.pe.bn.bias"], "changed_source_attention_tensor_count": 9, "frozen_non_attention_bn_buffer_count": 309, "frozen_parameter_tensor_count": 322, "max_frozen_non_attention_bn_buffer_delta": 2.86102294921875e-06, "max_frozen_parameter_delta": 6.67572021484375e-06, "missing_frozen_bn_buffer_names": [], "missing_frozen_parameter_names": [], "passed": false, "variant_id": "T2"}
- G0–G5：未執行（依本計畫明確排除）。
- 所有 T/N runs 從 full-precision checkpoint 直接做 10-epoch attention-only fine-tuning；非-attention parameters 凍結。
- 限制：P8/V8/INT4 是 PyTorch fake quantization；不得解讀為真實 1-bit 硬體速度提升。

結論：`completed formal artifact; interpret metrics from validation_metrics.json`
