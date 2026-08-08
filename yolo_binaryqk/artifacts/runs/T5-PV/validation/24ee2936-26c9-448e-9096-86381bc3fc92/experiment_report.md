# T5-PV BinaryAttention run

- 目的：YOLO11m BinaryAttention accuracy/fidelity experiment。
- 相對前一 variant：`E0`；本 run 改動為 `ScaledBinaryAttention` / `scaled_sign`。
- dataset：COCO2017 full train2017/val2017；epochs：0；batch：128；seed：0。
- initialization：`source checkpoint validation`；paper profile：`BinaryAttention QAT fine-tuning`。
- config hash：`093dc5fdf063927e0dc9e4bb69750c8dfe1e101e21acb769f0d441c19ddd4363`。

## Attention / loss

- QK：`scaled_sign`；QAT：`True`；bias：`decomposed_2d`。
- KD：`hard`，components=`['output']`。
- 理論成本：binary QK=1、softmax=1、PV=1。
- N2/N3/N4 使用 residual dual basis；N4 的 magnitude rank-1 term 真正加入 score。

## Metrics

- mAP50_95: not available
- mAP50: not available
- mAP75: not available
- mAPs: not available
- mAPm: not available
- mAPl: not available

## Diagnostics

- {"fake_quantization": true, "hardware_speed_claim": false, "modules": [{"binary_forward_count": 1, "binary_qk_count": 1, "class": "ScaledBinaryAttention", "path": "", "probability_finite": true, "probability_shape": [2, 2, 16, 16], "pv_count": 1, "score_finite": true, "score_shape": [2, 2, 16, 16], "softmax_count": 1}], "theoretical_cost": {"binary_qk": 1, "pv": 1, "softmax": 1}, "variant_id": "T5-PV"}
- G0–G5：未執行（依本計畫明確排除）。
- 舊版逐級 T/N 全量重訓已停用；本正式結果是從 full-precision checkpoint 直接做 paper-aligned QAT fine-tuning。
- 限制：P8/V8/INT4 是 PyTorch fake quantization；不得解讀為真實 1-bit 硬體速度提升。

結論：`numerical self-check only`
