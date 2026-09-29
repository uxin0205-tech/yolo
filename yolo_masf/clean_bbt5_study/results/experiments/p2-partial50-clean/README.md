# P2-Partial50-Clean

P2 前 50% channels 使用 Lite35 MFAM，其餘 channels exact bypass。

- Comparison tier：strict_fair
- Family：P2
- Schedule：direct100
- Seeds：42、43

| Split | mAP50–95 mean ± std | Ball AP | Bat AP | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.44763 ± 0.00310 | 0.21344 ± 0.00348 | 0.68182 ± 0.00968 | 0.23113 ± 0.00875 | 0.73605 ± 0.02197 | 0.92498 ± 0.00085 |
| test | 0.57621 ± 0.01268 | 0.52722 ± 0.00114 | 0.62519 ± 0.02422 | 0.34018 ± 0.00546 | 0.56842 ± 0.00032 | 0.67090 ± 0.00694 |

逐 seed 證據：

- artifacts/clean-bbt5-ablation/evaluation/val/p2-partial50-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/val/p2-partial50-clean-seed43/
- artifacts/clean-bbt5-ablation/evaluation/test/p2-partial50-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/test/p2-partial50-clean-seed43/
