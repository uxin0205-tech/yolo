# Clean BBT5 結果

完整訓練、統一評估、selection freeze、historical test 與硬體量測均已完成。

- 正式中文報告：[REPORT.md](REPORT.md)
- 逐 seed 指標：[per_seed_metrics.csv](per_seed_metrics.csv)
- Mean/std 總表：[comparison_mean_std.csv](comparison_mean_std.csv)
- 硬體資料：[hardware_profiles.csv](hardware_profiles.csv)
- 最終一致性稽核：[final_audit.json](../../artifacts/clean-bbt5-ablation/final_audit.json)
- 正式 run 稽核：[formal_run_audit.csv](formal_run_audit.csv)
- 各實驗說明與結果：[experiments/](experiments/)

Historical test 已被過去研究查看，不可用於 unseen claim；selection 由 validation 先行凍結。
