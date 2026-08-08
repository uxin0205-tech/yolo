# T6-O/A BinaryAttention run

- 目的：YOLO11m BinaryAttention accuracy/fidelity experiment。
- 相對前一 variant：`T4`；本 run 改動為 `ResidualDualFullBasisAttention` / `dual`。
- dataset：/home/hplab/uxin/yolo11/yolo_binaryqk/data/coco_full.txt；epochs：10；batch：128；seed：0。
- initialization：`full-precision source checkpoint fine-tuning`；paper profile：`10-epoch attention-only QAT fine-tuning`。
- checkpoint/metrics weights：`epoch_ema` / `epoch_ema`。
- config hash：`6e532d3286bd758ff6ddcd604cdd1067a53c3be3d0f696f8c93f79d61f787736`。

## Attention / loss

- QK：`dual`；QAT：`True`；bias：`none`。
- quantization contract：`{"bias_initialization": "none", "bias_parameterization": "none", "p8_scale": "none", "qk_scale": "residual_basis_channel_per_token", "v8_scale": "none"}`。
- KD：`positional+attention`，components=`['positional', 'attention']`。
- T6 KD target family：`T1-T5`；N4 non-KD parent：`not applicable`。
- 理論成本：binary QK=4、softmax=1、PV=1。
- T3/T4/T5 對應 parallel/full-basis/matched-basis dual attention；N4 的 magnitude rank-1 term 真正加入 score。

## Metrics

- mAP50_95: 0.50549
- mAP50: 0.67347
- mAP75: None
- coco_mAP50_95: 0.5114617026285858
- coco_mAP50: 0.6805138447493566
- coco_mAP75: 0.5566550813963086
- mAPs: 0.3325094266150388
- mAPm: 0.5630769966852388
- mAPl: 0.6691113443513907

## Diagnostics

- {"fake_quantization": false, "hardware_speed_claim": false, "modules": [{"binary_forward_count": 3131, "binary_qk_count": 12524, "class": "ResidualDualFullBasisAttention", "path": "model.10.m.0.attn", "probability_finite": true, "probability_shape": [8, 4, 231, 231], "pv_count": 3131, "score_finite": true, "score_shape": [8, 4, 231, 231], "softmax_count": 3131}], "theoretical_cost": {"binary_qk": 4, "pv": 1, "softmax": 1}, "variant_id": "T6-O/A"}
- Attention-only delta proof：{"changed_source_attention_parameter_names": ["model.10.m.0.attn.qkv.conv.weight", "model.10.m.0.attn.qkv.bn.weight", "model.10.m.0.attn.qkv.bn.bias", "model.10.m.0.attn.proj.conv.weight", "model.10.m.0.attn.proj.bn.weight", "model.10.m.0.attn.proj.bn.bias", "model.10.m.0.attn.pe.conv.weight", "model.10.m.0.attn.pe.bn.weight", "model.10.m.0.attn.pe.bn.bias"], "changed_source_attention_tensor_count": 9, "frozen_non_attention_bn_buffer_count": 309, "frozen_parameter_tensor_count": 322, "max_frozen_non_attention_bn_buffer_delta": 2.86102294921875e-06, "max_frozen_parameter_delta": 6.67572021484375e-06, "missing_frozen_bn_buffer_names": [], "missing_frozen_parameter_names": [], "passed": false, "variant_id": "T6-O/A"}
- G0–G5：未執行（依本計畫明確排除）。
- 所有 T/N runs 從 full-precision checkpoint 直接做 10-epoch attention-only fine-tuning；非-attention parameters 凍結。
- 限制：P8/V8/INT4 是 PyTorch fake quantization；不得解讀為真實 1-bit 硬體速度提升。

結論：`completed formal artifact; interpret metrics from validation_metrics.json`
