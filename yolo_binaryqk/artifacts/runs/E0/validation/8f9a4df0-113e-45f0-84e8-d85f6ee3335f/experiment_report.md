# E0 BinaryAttention run

- 目的：YOLO11m BinaryAttention accuracy/fidelity experiment。
- 相對前一 variant：`E0`；本 run 改動為 `FPAttention` / `fp`。
- dataset：/home/hplab/uxin/yolo11/yolo_binaryqk/data/coco_full.txt；epochs：0；batch：128；seed：0。
- initialization：`full-precision source checkpoint fine-tuning`；paper profile：`BinaryAttention QAT fine-tuning`。
- checkpoint/metrics weights：`source checkpoint` / `source checkpoint`。
- config hash：`7450d39349ea67a3079218cbdcddfb465952ef1907aa0263f32470574e9cbc17`。

## Attention / loss

- QK：`fp`；QAT：`False`；bias：`none`。
- quantization contract：`{}`。
- KD：`None`，components=`[]`。
- T6 KD target family：`None`；N4 non-KD parent：`not applicable`。
- 理論成本：binary QK=0、softmax=1、PV=1。
- T3/T4/T5 對應 parallel/full-basis/matched-basis dual attention；N4 的 magnitude rank-1 term 真正加入 score。

## Metrics

- mAP50_95: 0.5048069544762941
- mAP50: 0.6741496931086107
- mAP75: None
- coco_mAP50_95: 0.510850334214168
- coco_mAP50: 0.6814455387161382
- coco_mAP75: 0.5520465575802941
- mAPs: 0.3336263419692121
- mAPm: 0.5664419010752192
- mAPl: 0.6695556482447133

## Diagnostics

- {"fake_quantization": false, "hardware_speed_claim": false, "modules": [], "theoretical_cost": {"binary_qk": 0, "pv": 1, "softmax": 1}, "variant_id": "E0"}
- Attention-only delta proof：{}
- G0–G5：未執行（依本計畫明確排除）。
- 所有 T/N runs 從 full-precision checkpoint 直接做 10-epoch attention-only fine-tuning；非-attention parameters 凍結。
- 限制：P8/V8/INT4 是 PyTorch fake quantization；不得解讀為真實 1-bit 硬體速度提升。

結論：`zero-training validation completed`
