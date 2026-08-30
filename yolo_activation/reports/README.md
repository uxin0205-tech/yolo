# Full35 Activation 報告與結果

## 權威入口

- [已完成 Activation 權威整合報告](completed-activation-integrated-report.md)：目前完成範圍、六種 activation、可讀公式、完整八項結果、SIPA／BCSP 邊界、權重與量化決策。
- [六種 activation 可讀版完整數學推導](../docs/research/full35-activation-mathematical-derivations.md)：每個係數與限制條件的逐步來源。
- [可重建權重](../release/weights/README.md)：accepted SiLU 與完成 10-epoch recovery 的 qSiLU inference 權重。

## Machine-readable 結果

- [完整 JSON](full35-activation-results.json)：Float／Bit-True zero-shot、八項 selector、11-region sensitivity、queue、OOM probe 與權重 lineage。
- [八項指標 CSV](full35-activation-results.csv)：可直接匯入試算表或量化分析程式。

JSON／CSV 由 `scripts/export_full35_results.py` 從本機原始 artifacts 產生。GitHub 保存靜態結果，因此
clone 後不需要下載約 29 GB 原始 runs，也能核對主要數值。

## 歷史詳細報告

- [2026-08-29 最終整合分析](full35-activation-final-analysis.md)：保留當時完整表格、停止點與量化交接說明。內容仍可追溯，但目前狀態與閱讀順序以權威整合報告為準。

報告中的硬體成本是結構 proxy；目前沒有 FPGA／ASIC 板上 latency、power、LUT、DSP 或 BRAM 實測。
