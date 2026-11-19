# P2-PaperFormula-Clean

在 P2 加入論文式 MFAM：DW3、DW5、factorized 7、factorized 9，加 identity 與兩層 1x1 residual fusion。

- Comparison tier：strict_fair
- Family：P2
- Schedule：direct100
- Seeds：42、43

| Split | mAP50–95 mean ± std | Ball AP | Bat AP | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.45251 ± 0.01368 | 0.24056 ± 0.04743 | 0.66445 ± 0.02008 | 0.25092 ± 0.01847 | 0.69509 ± 0.02978 | 0.89928 ± 0.01991 |
| test | 0.59378 ± 0.02073 | 0.53867 ± 0.04643 | 0.64890 ± 0.00498 | 0.34615 ± 0.03928 | 0.61480 ± 0.06471 | 0.66526 ± 0.02448 |

逐 seed 證據：

- artifacts/clean-bbt5-ablation/evaluation/val/p2-paperformula-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/val/p2-paperformula-clean-seed43/
- artifacts/clean-bbt5-ablation/evaluation/test/p2-paperformula-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/test/p2-paperformula-clean-seed43/
