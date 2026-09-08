# 歷史區：需要追溯時才閱讀

這裡收納歷史入口，以及經核准移入的[十二份報告](reports/README.md)。有必要機讀／hash 依賴的報告仍保留原位。舊數字保留日期與比較條件，不能直接拿來當本輪 V36 的隔離結論。返回[目前計畫](../CURRENT_PLAN.md)。

## 保留的歷史證據

| 主題 | 入口 | 為何保留 |
| --- | --- | --- |
| activation／Q3 | [Hardswish 修訂](reports/2026-08-31-hardswish-policy-revision.md)、[Q3 血緣](../reports/2026-08-31-q3-activation-parent-evidence.md) | 解釋 qSiLU 主線與旁支，不冒稱其他 parent 可互換 |
| 舊 activation 圖 | [老師版與限制](../reports/2026-08-29-activation-preselection-report.md) | 已發行證據；不是目前完整比較結論 |
| poly_shift／V19／V29 | [執行報告](reports/2026-09-04-v19-v29-progressive-quantization-execution.md)、[封存與接手](reports/2026-09-05-v29-selection-and-qsilu-handoff-plan.md) | checkpoint handoff、負面 QAT 與決策血緣 |
| V35／V36 舊盤點 | [盤點快照](reports/2026-09-07-full-model-audit-four-day-plan.md)、[權重分布快照](../reports/2026-09-07-weight-distribution-detailed-analysis.md) | 1,332 筆 CPU 誤差、cross-parent 限制及 V36 起點來源 |
| 研究來源 | [Paper-TWN](../research/2026-09-04-paper-twn-primary-literature.md)、[文獻稽核](../research/2026-09-01-quantization-literature-and-innovation-audit.md)、[原始方法規格](../../quantize_spec.md) | 方法定義與研究假說；不是執行狀態 |

其他舊報告與命令見[整理前報告索引](report-index-before-2026-09-08.md)、[整理前首頁](project-readme-before-2026-09-08.md)、[腳本索引](scripts-before-2026-09-08.md)、[設定索引](config-index-before-2026-09-08.md)。这些是歷史快照，包含已被取代的狀態，不能作為啟動 queue 的指示。

## 薄弱結果如何處理

降低結論強度、移出現行首頁，不直接刪掉負面證據。只有證明無依賴且可安全移除的冗餘才列入[清除提案](../organization/cleanup-proposal-2026-09-08.md)。已有 parent、checkpoint、公開數字或論文方法引用的檔案保留。
