# T5-PV BinaryAttention run

- 目的：YOLO11m BinaryAttention accuracy/fidelity experiment。
- 相對前一 variant：`E0`；本 run 改動為 `ScaledBinaryAttention` / `scaled_sign`。
- dataset：/home/hplab/uxin/yolo11/yolo_binaryqk/data/coco_full.txt；epochs：300；batch：128；seed：0。
- initialization：`full-precision source checkpoint fine-tuning`；paper profile：`BinaryAttention QAT fine-tuning`。
- config hash：`093dc5fdf063927e0dc9e4bb69750c8dfe1e101e21acb769f0d441c19ddd4363`。

## Attention / loss

- QK：`scaled_sign`；QAT：`True`；bias：`decomposed_2d`。
- KD：`hard`，components=`('output',)`。
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
- G0–G5：未執行（依本計畫明確排除）。
- 舊版逐級 T/N 全量重訓已停用；本正式結果是從 full-precision checkpoint 直接做 paper-aligned QAT fine-tuning。
- 限制：P8/V8/INT4 是 PyTorch fake quantization；不得解讀為真實 1-bit 硬體速度提升。

結論：`尚無 COCO 結果；此 artifact 可供正式訓練/驗證重建。`
