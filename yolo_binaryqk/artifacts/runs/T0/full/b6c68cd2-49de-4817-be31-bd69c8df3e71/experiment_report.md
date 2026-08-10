# T0 BinaryAttention run

- 目的：YOLO11m BinaryAttention accuracy/fidelity experiment。
- 相對前一 variant：`E0`；本 run 改動為 `FPAttention` / `fp`。
- dataset：/home/hplab/uxin/yolo11/yolo_binaryqk/data/coco_full.txt；epochs：250；batch：128；seed：0。
- config hash：`217e1156e2e22c1f9d96e5adab974e2e5adf02a3647c5e9c1d1456c320121a4e`。

## Attention / loss

- QK：`fp`；QAT：`False`；bias：`none`。
- KD：`None`，components=`()`。
- 理論成本：binary QK=0、softmax=1、PV=1。
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
- 5k/30k 與 2/10 epoch micro/pilot：未執行。
- 限制：P8/V8/INT4 是 PyTorch fake quantization；不得解讀為真實 1-bit 硬體速度提升。

結論：`尚無 COCO 結果；此 artifact 可供正式訓練/驗證重建。`
