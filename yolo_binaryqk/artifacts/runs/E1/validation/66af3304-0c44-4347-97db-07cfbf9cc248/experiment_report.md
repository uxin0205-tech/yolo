# E1 BinaryAttention run

- 目的：YOLO11m BinaryAttention accuracy/fidelity experiment。
- 相對前一 variant：`E0`；本 run 改動為 `ScaledBinaryAttention` / `scaled_sign`。
- dataset：/home/hplab/uxin/yolo11/yolo_binaryqk/data/coco_full.txt；epochs：0；batch：128；seed：0。
- initialization：`full-precision source checkpoint fine-tuning`；paper profile：`BinaryAttention QAT fine-tuning`。
- checkpoint/metrics weights：`source checkpoint` / `source checkpoint`。
- config hash：`402425cb8179e36e6e7b25eeb2b6f6664c0b82bbf4f6db97f6e95d49aa698d11`。

## Attention / loss

- QK：`scaled_sign`；QAT：`False`；bias：`none`。
- quantization contract：`{}`。
- KD：`None`，components=`[]`。
- T6 KD target family：`None`；N4 non-KD parent：`not applicable`。
- 理論成本：binary QK=1、softmax=1、PV=1。
- T3/T4/T5 對應 parallel/full-basis/matched-basis dual attention；N4 的 magnitude rank-1 term 真正加入 score。

## Metrics

- mAP50_95: 0.48121393727721495
- mAP50: 0.6434630843261905
- mAP75: None
- coco_mAP50_95: 0.48056105904621166
- coco_mAP50: 0.6423167062257058
- coco_mAP75: 0.5221428315678437
- mAPs: 0.298182756827083
- mAPm: 0.5322253046188234
- mAPl: 0.6412899955447279

## Diagnostics

- {"fake_quantization": false, "hardware_speed_claim": false, "modules": [{"binary_forward_count": 42, "binary_qk_count": 42, "class": "UltralyticsBinaryAttention", "path": "model.10.m.0.attn", "probability_finite": true, "probability_shape": [8, 4, 231, 231], "pv_count": 42, "score_finite": true, "score_shape": [8, 4, 231, 231], "softmax_count": 42}], "theoretical_cost": {"binary_qk": 1, "pv": 1, "softmax": 1}, "variant_id": "E1"}
- Attention-only delta proof：{}
- G0–G5：未執行（依本計畫明確排除）。
- 所有 T/N runs 從 full-precision checkpoint 直接做 10-epoch attention-only fine-tuning；非-attention parameters 凍結。
- 限制：P8/V8/INT4 是 PyTorch fake quantization；不得解讀為真實 1-bit 硬體速度提升。

結論：`zero-training validation completed`
