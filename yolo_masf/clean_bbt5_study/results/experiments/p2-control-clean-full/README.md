# P2-Control-Clean-Full

P2 最佳化 control B：承接同 seed 的 Head best，全模型最多 80 epochs、lr0=0.001；不進 strict-fair ranking。

- Comparison tier：optimization_control
- Family：P2
- Schedule：full80
- Seeds：42、43

| Split | mAP50–95 mean ± std | Ball AP | Bat AP | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.49288 ± 0.03876 | 0.26938 ± 0.09166 | 0.71638 ± 0.01415 | 0.35054 ± 0.06764 | 0.75894 ± 0.00336 | 0.90285 ± 0.04134 |
| test | 0.65049 ± 0.00138 | 0.62066 ± 0.01940 | 0.68033 ± 0.02216 | 0.42250 ± 0.01700 | 0.60119 ± 0.01163 | 0.76683 ± 0.01499 |

逐 seed 證據：

- artifacts/clean-bbt5-ablation/evaluation/val/p2-control-clean-full-seed42/
- artifacts/clean-bbt5-ablation/evaluation/val/p2-control-clean-full-seed43/
- artifacts/clean-bbt5-ablation/evaluation/test/p2-control-clean-full-seed42/
- artifacts/clean-bbt5-ablation/evaluation/test/p2-control-clean-full-seed43/
