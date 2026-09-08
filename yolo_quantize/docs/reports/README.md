# 報告：先看目前這輪

日常只看下列五份。[精簡清單](../organization/report-retention-2026-09-08.md)已獲核准並執行：10 份保留原位、12 份移入歷史區、3 份重複報告備份後刪除。日後接續先看[恢復交接表](../RESUME.md)。

| 閱讀順序 | 報告 | 用途 |
| --- | --- | --- |
| 先看 | [階段收尾與量化延後](2026-09-08-quantization-phase-handoff.md) | 六組 QAT、十二個累積 PTQ、完整指標與恢復研究條件 |
| 方法 | [量化方法與決策](2026-09-08-quantization-method-and-decisions.md) | LSQ+、activation、權重分布與逐區累積的依據 |
| 執行沿革 | [有限累積 PTQ](2026-09-08-cumulative-ptq-plan.md) | 已完成的十二候選提名與接受規則；非新啟動授權 |
| 1 | [QAT 恢復結果](2026-09-07-continuous-qat-recovery-results.md) | 本輪已完成工作的逐回合、16 指標與 PTQ→QAT 差異 |
| 2 | [比較條件與證據層級](2026-09-07-evidence-consistency-boundaries.md) | 同 parent 稽核及權重誤差／probe／mAP／QAT 的界線 |

計畫只讀[CURRENT_PLAN](../CURRENT_PLAN.md)。其餘日期報告保留為歷史證據，不再並列「最新」；需要追溯時進[歷史索引](../archive/README.md)。

所有結果均須區分 search 與 formal、單 seed 與重複實驗、量化模擬與實際整數硬體。

## 另外保留的必要依賴

- [舊 activation smoke](2026-08-29-activation-output-smoke-v2.md)、[老師版 activation 預選](2026-08-29-activation-preselection-report.md)：發行與圖表引用。
- [Q3 血緣](2026-08-31-q3-activation-parent-evidence.md)：條件旁支與 manifest 的 report SHA。
- [V4 基準驗證](2026-09-03-v4-qsilu-backbone-early-w8-search-validation.md)：既有機讀契約引用。
- [權重分布詳細分析](2026-09-07-weight-distribution-detailed-analysis.md)：獨有分布證據與生成附錄引用，須保留跨 parent 限制。

這五份不作目前方法排名；其餘有獨有數據的舊報告見[十二份歷史報告](../archive/reports/README.md)。原始 artifact 與 checkpoint 未刪除。
