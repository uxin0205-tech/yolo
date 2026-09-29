# N4-PV BinaryAttention run

- 目的：YOLO11m BinaryAttention accuracy/fidelity experiment。
- 相對前一 variant：`best-N4`；本 run 改動為 `MagnitudeSideChannelAttention` / `dual`。
- dataset：COCO2017 full train2017/val2017；epochs：0；batch：128；seed：0。
- config hash：`f14b1b0b396ffa7b08cab08bcd020e57fcf405c3dc11ffac1e8b53629d48a890`。

## Attention / loss

- QK：`dual`；QAT：`True`；bias：`none`。
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
- 5k/30k 與 2/10 epoch micro/pilot：未執行。
- 限制：P8/V8/INT4 是 PyTorch fake quantization；不得解讀為真實 1-bit 硬體速度提升。

結論：`尚無 COCO 結果；此 artifact 可供正式訓練/驗證重建。`
