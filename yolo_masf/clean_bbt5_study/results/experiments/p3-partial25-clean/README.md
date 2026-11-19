# P3-Partial25-Clean

P3 前 25% channels 使用 Lite35 MFAM，其餘 channels exact bypass。

- Comparison tier：strict_fair
- Family：P3
- Schedule：direct100
- Seeds：42、43

| Split | mAP50–95 mean ± std | Ball AP | Bat AP | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.44953 ± 0.01042 | 0.20760 ± 0.01851 | 0.69147 ± 0.00234 | 0.21238 ± 0.02747 | 0.74245 ± 0.00852 | 0.90210 ± 0.01845 |
| test | 0.58576 ± 0.00292 | 0.52978 ± 0.03468 | 0.64175 ± 0.02885 | 0.36955 ± 0.02532 | 0.58133 ± 0.03313 | 0.67485 ± 0.00513 |

逐 seed 證據：

- artifacts/clean-bbt5-ablation/evaluation/val/p3-partial25-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/val/p3-partial25-clean-seed43/
- artifacts/clean-bbt5-ablation/evaluation/test/p3-partial25-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/test/p3-partial25-clean-seed43/
