# N4-PV BinaryAttention run

- 目的：YOLO11m BinaryAttention accuracy/fidelity experiment。
- 相對前一 variant：`best-N4`；本 run 改動為 `MagnitudeSideChannelAttention` / `dual`。
- dataset：COCO2017 full train2017/val2017；epochs：0；batch：128；seed：0。
- config hash：`f14b1b0b396ffa7b08cab08bcd020e57fcf405c3dc11ffac1e8b53629d48a890`。

## Attention / loss

- QK：`dual`；QAT：`True`；bias：`none`。
- KD：`None`，components=`[]`。
- 理論成本：binary QK=2、softmax=1、PV=1。
- N2/N3/N4 使用 residual dual basis；N4 的 magnitude rank-1 term 真正加入 score。

## Metrics

- mAP50_95: not available
- mAP50: not available
- mAP75: not available
- mAPs: not available
- mAPm: not available
- mAPl: not available

## Diagnostics

- {"fake_quantization": true, "hardware_speed_claim": false, "modules": [{"binary_forward_count": 1, "binary_qk_count": 2, "class": "MagnitudeSideChannelAttention", "path": "", "probability_finite": true, "probability_shape": [2, 2, 16, 16], "pv_count": 1, "score_finite": true, "score_shape": [2, 2, 16, 16], "softmax_count": 1}], "theoretical_cost": {"binary_qk": 2, "pv": 1, "softmax": 1}, "variant_id": "N4-PV"}
- G0–G5：未執行（依本計畫明確排除）。
- 5k/30k 與 2/10 epoch micro/pilot：未執行。
- 限制：P8/V8/INT4 是 PyTorch fake quantization；不得解讀為真實 1-bit 硬體速度提升。

結論：`numerical self-check only`
