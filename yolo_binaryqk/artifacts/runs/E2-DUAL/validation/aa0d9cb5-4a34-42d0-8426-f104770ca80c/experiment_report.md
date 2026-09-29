# E2-DUAL BinaryAttention run

- 目的：YOLO11m BinaryAttention accuracy/fidelity experiment。
- 相對前一 variant：`E0`；本 run 改動為 `ResidualDualMatchedBasisAttention` / `dual`。
- dataset：/home/hplab/uxin/yolo11/yolo_binaryqk/data/coco_full.txt；epochs：0；batch：128；seed：0。
- initialization：`source checkpoint validation`；paper profile：`BinaryAttention QAT fine-tuning`。
- config hash：`a525d230d65951414b833dc0b8df42e487a44e0c5b209b46adc8cbc1420b4c3e`。

## Attention / loss

- QK：`dual`；QAT：`False`；bias：`none`。
- KD：`None`，components=`()`。
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

- {"status": "pending"}
- G0–G5：未執行（依本計畫明確排除）。
- 所有 T/N runs 從 full-precision checkpoint 直接做 10-epoch attention-only fine-tuning；非-attention parameters 凍結。
- 限制：P8/V8/INT4 是 PyTorch fake quantization；不得解讀為真實 1-bit 硬體速度提升。

結論：`尚無 COCO 結果；此 artifact 可供正式訓練/驗證重建。`
