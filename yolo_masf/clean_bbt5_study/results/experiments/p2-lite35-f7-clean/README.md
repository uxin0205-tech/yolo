# P2-Lite35-F7-Clean

在 P2 使用 DW3、DW5、1x7→7x1 與兩層 1x1 residual fusion。

- Comparison tier：strict_fair
- Family：P2
- Schedule：direct100
- Seeds：42、43

| Split | mAP50–95 mean ± std | Ball AP | Bat AP | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.44023 ± 0.00257 | 0.19341 ± 0.01866 | 0.68705 ± 0.01352 | 0.23680 ± 0.00115 | 0.73365 ± 0.02366 | 0.88474 ± 0.00596 |
| test | 0.58982 ± 0.00541 | 0.51742 ± 0.00038 | 0.66222 ± 0.01044 | 0.35551 ± 0.03121 | 0.61844 ± 0.05446 | 0.65184 ± 0.01246 |

逐 seed 證據：

- artifacts/clean-bbt5-ablation/evaluation/val/p2-lite35-f7-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/val/p2-lite35-f7-clean-seed43/
- artifacts/clean-bbt5-ablation/evaluation/test/p2-lite35-f7-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/test/p2-lite35-f7-clean-seed43/
