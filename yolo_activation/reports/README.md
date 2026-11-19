# Full35 Activation 交付報告

- [已完成 Activation 權威整合報告](completed-activation-integrated-report.md)：五個保留比較函數的公式、
  完成程度、四項摘要主表與八項 gate、中央函數誤差、運算量估計、SIPA／BCSP 邊界與量化決策。
- [報告圖表與 CSV](figures/README.md)：高解析度精度總覽、中央誤差放大圖、運算量圖及三份對應 CSV；
  最終視覺比較依使用者要求排除 ReLU。
- [最終整合分析](full35-activation-final-analysis.md)：資料、實驗順序、訓練配置、結果、停止點與量化交接建議。
- [Machine-readable JSON](full35-activation-results.json)：包含 full Float／Bit-True zero-shot、八項 selector、
  11-region sensitivity、queue、OOM probe 與權重 lineage；歷史 ReLU 診斷仍保留於此。
- [八項指標 CSV](full35-activation-results.csv)：可直接匯入試算表或量化分析程式。
- [六種 activation 完整數學推導](../docs/research/full35-activation-mathematical-derivations.md)。
- [可重建權重](../release/weights/README.md)：accepted SiLU 與已完成的 qSiLU 10-epoch inference 權重。

JSON／CSV 由 `scripts/export_full35_results.py` 從本機不可提交的原始 artifacts 產生；圖表由
`scripts/generate_activation_report_figures.py` 產生。發布版保存靜態結果，因此 clone 後不需要原始
29 GB runs 也能讀取全部關鍵數值。
