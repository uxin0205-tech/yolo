# P2-Direct-Clean

四尺度 P2/P3/P4/P5 baseline；新增標準高解析 P2 detection head。

- Comparison tier：strict_fair
- Family：P2
- Schedule：direct100
- Seeds：42、43

| Split | mAP50–95 mean ± std | Ball AP | Bat AP | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.44659 ± 0.00760 | 0.19323 ± 0.01770 | 0.69995 ± 0.00251 | 0.23366 ± 0.02360 | 0.73796 ± 0.01884 | 0.86188 ± 0.02594 |
| test | 0.57584 ± 0.00559 | 0.50675 ± 0.03027 | 0.64493 ± 0.01909 | 0.34821 ± 0.00406 | 0.57575 ± 0.13304 | 0.64558 ± 0.00409 |

逐 seed 證據：

- artifacts/clean-bbt5-ablation/evaluation/val/p2-direct-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/val/p2-direct-clean-seed43/
- artifacts/clean-bbt5-ablation/evaluation/test/p2-direct-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/test/p2-direct-clean-seed43/
