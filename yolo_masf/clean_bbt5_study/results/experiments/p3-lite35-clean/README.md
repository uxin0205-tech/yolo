# P3-Lite35-Clean

只在 P3 加入 DW3、DW5 與論文式雙 1x1 fusion。

- Comparison tier：strict_fair
- Family：P3
- Schedule：direct100
- Seeds：42、43

| Split | mAP50–95 mean ± std | Ball AP | Bat AP | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.43785 ± 0.00889 | 0.19576 ± 0.00947 | 0.67993 ± 0.00831 | 0.21039 ± 0.00639 | 0.69275 ± 0.08270 | 0.92409 ± 0.01017 |
| test | 0.58962 ± 0.00991 | 0.53046 ± 0.02162 | 0.64877 ± 0.00180 | 0.37453 ± 0.03524 | 0.61821 ± 0.00406 | 0.67196 ± 0.00368 |

逐 seed 證據：

- artifacts/clean-bbt5-ablation/evaluation/val/p3-lite35-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/val/p3-lite35-clean-seed43/
- artifacts/clean-bbt5-ablation/evaluation/test/p3-lite35-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/test/p3-lite35-clean-seed43/
