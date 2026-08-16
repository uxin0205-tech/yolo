# P2-Control-Clean-Head

P2 最佳化 control A：只訓練新增 P2 slot/head，最多 20 epochs；不進 strict-fair ranking。

- Comparison tier：optimization_control
- Family：P2
- Schedule：head20
- Seeds：42、43

| Split | mAP50–95 mean ± std | Ball AP | Bat AP | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.00599 ± 0.00224 | 0.00113 ± 0.00160 | 0.01084 ± 0.00608 | 0.00022 ± 0.00031 | 0.00754 ± 0.00004 | 0.06826 ± 0.09653 |
| test | 0.00579 ± 0.00803 | 0.00579 ± 0.00819 | 0.00579 ± 0.00787 | 0.00077 ± 0.00108 | 0.00685 ± 0.00823 | 0.02979 ± 0.04213 |

逐 seed 證據：

- artifacts/clean-bbt5-ablation/evaluation/val/p2-control-clean-head-seed42/
- artifacts/clean-bbt5-ablation/evaluation/val/p2-control-clean-head-seed43/
- artifacts/clean-bbt5-ablation/evaluation/test/p2-control-clean-head-seed42/
- artifacts/clean-bbt5-ablation/evaluation/test/p2-control-clean-head-seed43/
