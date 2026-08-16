# P2-Partial25-Clean

P2 前 25% channels 使用 Lite35 MFAM，其餘 channels exact bypass。

- Comparison tier：strict_fair
- Family：P2
- Schedule：direct100
- Seeds：42、43

| Split | mAP50–95 mean ± std | Ball AP | Bat AP | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.44910 ± 0.00286 | 0.23862 ± 0.06809 | 0.65958 ± 0.07381 | 0.21827 ± 0.02689 | 0.70576 ± 0.03884 | 0.87320 ± 0.06380 |
| test | 0.52308 ± 0.09361 | 0.45727 ± 0.12246 | 0.58890 ± 0.06475 | 0.29488 ± 0.06346 | 0.51108 ± 0.09629 | 0.61807 ± 0.08948 |

逐 seed 證據：

- artifacts/clean-bbt5-ablation/evaluation/val/p2-partial25-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/val/p2-partial25-clean-seed43/
- artifacts/clean-bbt5-ablation/evaluation/test/p2-partial25-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/test/p2-partial25-clean-seed43/
