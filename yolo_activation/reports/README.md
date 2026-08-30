# Full35 Activation 交付报告

- [已完成 Activation 权威整合报告](completed-activation-integrated-report.md)：六种 activation 的公式、完成程度、完整结果、SIPA／BCSP 边界与量化决策。
- [最终整合分析](full35-activation-final-analysis.md)：资料、实验顺序、训练配置、结果、停止点与量化交接建议。
- [Machine-readable JSON](full35-activation-results.json)：包含 full Float／Bit-True zero-shot、八项 selector、
  11-region sensitivity、queue、OOM probe 与权重 lineage。
- [八项指标 CSV](full35-activation-results.csv)：可直接汇入试算表或量化分析程式。
- [六种 activation 完整数学推导](../docs/research/full35-activation-mathematical-derivations.md)。
- [可重建权重](../release/weights/README.md)：accepted SiLU 与已完成的 qSiLU 10-epoch inference 权重。

JSON／CSV 由 `scripts/export_full35_results.py` 从本机不可提交的原始 artifacts 产生；发布版保留生成后的
静态结果，因此 clone 后不需要原始 29 GB runs 也能读取全部关键数值。
