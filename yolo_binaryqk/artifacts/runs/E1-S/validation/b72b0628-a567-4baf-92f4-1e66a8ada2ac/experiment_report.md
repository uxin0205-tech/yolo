# E1-S BinaryAttention run

- 目的：YOLO11m BinaryAttention accuracy/fidelity experiment。
- 相對前一 variant：`E0`；本 run 改動為 `SignOnlyBinaryAttention` / `sign`。
- dataset：/home/hplab/uxin/yolo11/yolo_binaryqk/data/coco_full.txt；epochs：0；batch：128；seed：0。
- initialization：`full-precision source checkpoint fine-tuning`；paper profile：`BinaryAttention QAT fine-tuning`。
- checkpoint/metrics weights：`source checkpoint` / `source checkpoint`。
- config hash：`f218837ec904f280034d76bad425c5ee8d793354b9683c3b972d02f883aa6950`。

## Attention / loss

- QK：`sign`；QAT：`False`；bias：`none`。
- quantization contract：`{}`。
- KD：`None`，components=`[]`。
- T6 KD target family：`None`；N4 non-KD parent：`not applicable`。
- 理論成本：binary QK=1、softmax=1、PV=1。
- T3/T4/T5 對應 parallel/full-basis/matched-basis dual attention；N4 的 magnitude rank-1 term 真正加入 score。

## Metrics

- mAP50_95: 0.45070248673578833
- mAP50: 0.6036581810581465
- mAP75: None
- coco_mAP50_95: 0.4592965400249649
- coco_mAP50: 0.6150464877146296
- coco_mAP75: 0.4992353448249912
- mAPs: 0.2832132818847663
- mAPm: 0.5037298241446655
- mAPl: 0.6201403516854299

## Diagnostics

- {"fake_quantization": false, "hardware_speed_claim": false, "modules": [{"binary_forward_count": 42, "binary_qk_count": 42, "class": "UltralyticsBinaryAttention", "path": "model.10.m.0.attn", "probability_finite": true, "probability_shape": [8, 4, 231, 231], "pv_count": 42, "score_finite": true, "score_shape": [8, 4, 231, 231], "softmax_count": 42}], "theoretical_cost": {"binary_qk": 1, "pv": 1, "softmax": 1}, "variant_id": "E1-S"}
- Attention-only delta proof：{}
- G0–G5：未執行（依本計畫明確排除）。
- 所有 T/N runs 從 full-precision checkpoint 直接做 10-epoch attention-only fine-tuning；非-attention parameters 凍結。
- 限制：P8/V8/INT4 是 PyTorch fake quantization；不得解讀為真實 1-bit 硬體速度提升。

結論：`zero-training validation completed`
