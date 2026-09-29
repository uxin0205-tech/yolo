# P3-M7-Clean

只在 P3 加入 legacy MFAM：DW3、DW5、factorized 7 與單層 1x1 fusion；不含 9x9。

- Comparison tier：strict_fair
- Family：P3
- Schedule：direct100
- Seeds：42、43

| Split | mAP50–95 mean ± std | Ball AP | Bat AP | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.44113 ± 0.01316 | 0.21152 ± 0.00419 | 0.67075 ± 0.03052 | 0.23703 ± 0.01947 | 0.73260 ± 0.01502 | 0.85508 ± 0.04967 |
| test | 0.57586 ± 0.00196 | 0.51841 ± 0.00801 | 0.63330 ± 0.00408 | 0.33897 ± 0.00838 | 0.59529 ± 0.04472 | 0.64750 ± 0.01017 |

逐 seed 證據：

- artifacts/clean-bbt5-ablation/evaluation/val/p3-m7-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/val/p3-m7-clean-seed43/
- artifacts/clean-bbt5-ablation/evaluation/test/p3-m7-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/test/p3-m7-clean-seed43/
