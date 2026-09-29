# N4-I4 BinaryAttention run

- 目的：YOLO11m BinaryAttention accuracy/fidelity experiment。
- 相對前一 variant：`T7-D`；本 run 改動為 `MagnitudeSideChannelAttention` / `dual`。
- dataset：/home/hplab/uxin/yolo11/yolo_binaryqk/data/coco_full.txt；epochs：10；batch：128；seed：0。
- initialization：`full-precision source checkpoint fine-tuning`；paper profile：`10-epoch attention-only QAT fine-tuning`。
- checkpoint/metrics weights：`epoch_ema` / `epoch_ema`。
- config hash：`758fee34f21bc9270f7a93d9bff3fb39b26f609e3b95afab11ffff6336a89050`。

## Attention / loss

- QK：`dual`；QAT：`True`；bias：`dense_2d`。
- quantization contract：`{"bias_initialization": "truncated_normal_std_0.02", "bias_parameterization": "full_2d_relative_position", "p8_scale": "none", "qk_scale": "residual_basis_channel_per_token", "v8_scale": "none"}`。
- KD：`None`，components=`[]`。
- T6 KD target family：`None`；N4 non-KD parent：`T7-D`。
- 理論成本：binary QK=2、softmax=1、PV=1。
- T3/T4/T5 對應 parallel/full-basis/matched-basis dual attention；N4 的 magnitude rank-1 term 真正加入 score。

## Metrics

- mAP50_95: 0.50406
- mAP50: 0.67187
- mAP75: None
- coco_mAP50_95: 0.5098601509512347
- coco_mAP50: 0.67905334039538
- coco_mAP75: 0.5547287256842275
- mAPs: 0.32984682608249327
- mAPm: 0.5633205777781276
- mAPl: 0.668300129495918

## Diagnostics

- {"fake_quantization": true, "hardware_speed_claim": false, "modules": [{"binary_forward_count": 3131, "binary_qk_count": 6262, "class": "MagnitudeSideChannelAttention", "path": "model.10.m.0.attn", "probability_finite": true, "probability_shape": [8, 4, 231, 231], "pv_count": 3131, "score_finite": true, "score_shape": [8, 4, 231, 231], "softmax_count": 3131}], "theoretical_cost": {"binary_qk": 2, "pv": 1, "softmax": 1}, "variant_id": "N4-I4"}
- Attention-only delta proof：{"changed_source_attention_parameter_names": ["model.10.m.0.attn.qkv.conv.weight", "model.10.m.0.attn.qkv.bn.weight", "model.10.m.0.attn.qkv.bn.bias", "model.10.m.0.attn.proj.conv.weight", "model.10.m.0.attn.proj.bn.weight", "model.10.m.0.attn.proj.bn.bias", "model.10.m.0.attn.pe.conv.weight", "model.10.m.0.attn.pe.bn.weight", "model.10.m.0.attn.pe.bn.bias"], "changed_source_attention_tensor_count": 9, "frozen_non_attention_bn_buffer_count": 309, "frozen_parameter_tensor_count": 322, "max_frozen_non_attention_bn_buffer_delta": 2.86102294921875e-06, "max_frozen_parameter_delta": 6.67572021484375e-06, "missing_frozen_bn_buffer_names": [], "missing_frozen_parameter_names": [], "passed": false, "variant_id": "N4-I4"}
- G0–G5：未執行（依本計畫明確排除）。
- 所有 T/N runs 從 full-precision checkpoint 直接做 10-epoch attention-only fine-tuning；非-attention parameters 凍結。
- 限制：P8/V8/INT4 是 PyTorch fake quantization；不得解讀為真實 1-bit 硬體速度提升。

結論：`completed formal artifact; interpret metrics from validation_metrics.json`
