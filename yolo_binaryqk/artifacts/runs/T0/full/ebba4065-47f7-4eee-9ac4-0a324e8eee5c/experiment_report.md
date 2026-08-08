# T0 BinaryAttention run

- 目的：YOLO11m BinaryAttention accuracy/fidelity experiment。
- 相對前一 variant：`E0`；本 run 改動為 `FPAttention` / `fp`。
- dataset：/home/hplab/uxin/yolo11/yolo_binaryqk/data/coco_full.txt；epochs：10；batch：128；seed：0。
- initialization：`full-precision source checkpoint fine-tuning`；paper profile：`10-epoch attention-only FP fine-tuning`。
- checkpoint/metrics weights：`epoch_ema` / `epoch_ema`。
- config hash：`217e1156e2e22c1f9d96e5adab974e2e5adf02a3647c5e9c1d1456c320121a4e`。

## Attention / loss

- QK：`fp`；QAT：`False`；bias：`none`。
- quantization contract：`{}`。
- KD：`None`，components=`[]`。
- T6 KD target family：`None`；N4 non-KD parent：`not applicable`。
- 理論成本：binary QK=0、softmax=1、PV=1。
- T3/T4/T5 對應 parallel/full-basis/matched-basis dual attention；N4 的 magnitude rank-1 term 真正加入 score。

## Metrics

- mAP50_95: 0.50672
- mAP50: 0.67581
- mAP75: None
- coco_mAP50_95: 0.5126706001859774
- coco_mAP50: 0.6824277056728909
- coco_mAP75: 0.5571087030047439
- mAPs: 0.33101947461760745
- mAPm: 0.5653244369067247
- mAPl: 0.6732951842831945

## Diagnostics

- {"fake_quantization": false, "hardware_speed_claim": false, "modules": [], "theoretical_cost": {"binary_qk": 0, "pv": 1, "softmax": 1}, "variant_id": "T0"}
- Attention-only delta proof：{"changed_source_attention_parameter_names": ["model.10.m.0.attn.qkv.conv.weight", "model.10.m.0.attn.qkv.bn.weight", "model.10.m.0.attn.qkv.bn.bias", "model.10.m.0.attn.proj.conv.weight", "model.10.m.0.attn.proj.bn.weight", "model.10.m.0.attn.proj.bn.bias", "model.10.m.0.attn.pe.conv.weight", "model.10.m.0.attn.pe.bn.weight", "model.10.m.0.attn.pe.bn.bias"], "changed_source_attention_tensor_count": 9, "frozen_non_attention_bn_buffer_count": 309, "frozen_parameter_tensor_count": 322, "max_frozen_non_attention_bn_buffer_delta": 0.0, "max_frozen_parameter_delta": 0.0, "missing_frozen_bn_buffer_names": [], "missing_frozen_parameter_names": [], "passed": true, "variant_id": "T0"}
- G0–G5：未執行（依本計畫明確排除）。
- 所有 T/N runs 從 full-precision checkpoint 直接做 10-epoch attention-only fine-tuning；非-attention parameters 凍結。
- 限制：P8/V8/INT4 是 PyTorch fake quantization；不得解讀為真實 1-bit 硬體速度提升。

結論：`completed formal artifact; interpret metrics from validation_metrics.json`
