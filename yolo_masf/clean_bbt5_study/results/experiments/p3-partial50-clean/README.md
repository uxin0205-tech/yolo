# P3-Partial50-Clean

P3 前 50% channels 使用 Lite35 MFAM，其餘 channels exact bypass。

- Comparison tier：strict_fair
- Family：P3
- Schedule：direct100
- Seeds：42、43

| Split | mAP50–95 mean ± std | Ball AP | Bat AP | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.43065 ± 0.02142 | 0.19169 ± 0.00489 | 0.66962 ± 0.03794 | 0.21349 ± 0.00888 | 0.71301 ± 0.03652 | 0.80945 ± 0.07221 |
| test | 0.54302 ± 0.05069 | 0.44711 ± 0.09517 | 0.63894 ± 0.00622 | 0.38391 ± 0.01329 | 0.57162 ± 0.04337 | 0.58869 ± 0.06739 |

逐 seed 證據：

- artifacts/clean-bbt5-ablation/evaluation/val/p3-partial50-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/val/p3-partial50-clean-seed43/
- artifacts/clean-bbt5-ablation/evaluation/test/p3-partial50-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/test/p3-partial50-clean-seed43/
