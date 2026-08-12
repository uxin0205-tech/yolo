# T2 BinaryAttention run

- 目的：YOLO11m BinaryAttention accuracy/fidelity experiment。
- 相對前一 variant：`T1`；本 run 改動為 `ScaledBinaryAttention` / `scaled_sign`。
- dataset：/home/hplab/uxin/yolo11/yolo_binaryqk/data/coco_full.txt；epochs：10；batch：128；seed：0。
- initialization：`full-precision source checkpoint fine-tuning`；paper profile：`BinaryAttention QAT fine-tuning`。
- checkpoint/metrics weights：`source checkpoint` / `source checkpoint`。
- config hash：`ed7a5b51d8e62db06315774b14190551896596eecca3e952dc6954f528aaec08`。

## Attention / loss

- QK：`scaled_sign`；QAT：`True`；bias：`none`。
- quantization contract：`{"bias_initialization": "none", "bias_parameterization": "none", "p8_scale": "none", "qk_scale": "mean_abs_channel_token_per_sample_head", "v8_scale": "none"}`。
- KD：`None`，components=`()`。
- T3 KD target family：`None`；N4 non-KD parent family：`not applicable`。
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

- {"status": "pending"}
- Attention-only delta proof：{"status": "pending"}
- G0–G5：未執行（依本計畫明確排除）。
- 所有 T/N runs 從 full-precision checkpoint 直接做 10-epoch attention-only fine-tuning；非-attention parameters 凍結。
- 限制：P8/V8/INT4 是 PyTorch fake quantization；不得解讀為真實 1-bit 硬體速度提升。

結論：`尚無 COCO 結果；此 artifact 可供正式訓練/驗證重建。`
