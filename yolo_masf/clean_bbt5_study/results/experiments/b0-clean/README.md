# B0-Clean

原始 P3/P4/P5 三尺度 YOLO11m baseline；不加入 P2 或 MFAM。

- Comparison tier：strict_fair
- Family：B0
- Schedule：direct100
- Seeds：42、43

| Split | mAP50–95 mean ± std | Ball AP | Bat AP | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.46070 ± 0.01995 | 0.26499 ± 0.07677 | 0.65641 ± 0.03687 | 0.24724 ± 0.03167 | 0.70516 ± 0.01692 | 0.89839 ± 0.04302 |
| test | 0.54255 ± 0.06874 | 0.47867 ± 0.07515 | 0.60643 ± 0.06234 | 0.27741 ± 0.08908 | 0.58040 ± 0.09002 | 0.63458 ± 0.05670 |

逐 seed 證據：

- artifacts/clean-bbt5-ablation/evaluation/val/b0-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/val/b0-clean-seed43/
- artifacts/clean-bbt5-ablation/evaluation/test/b0-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/test/b0-clean-seed43/
