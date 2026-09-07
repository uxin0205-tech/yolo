# 研究報告索引

詳細補充：[SD4／三元分布與實驗分析](2026-09-07-weight-distribution-detailed-analysis.md)，含十區格式表、25格PTQ與QAT全16項指標。

最新入口：[2026-09-07全模型盤點、V36結果更正與四天計畫](2026-09-07-full-model-audit-four-day-plan.md)。用途為整合既有產物與尚未完成工作；數據由CPU腳本讀artifacts重建。新報告可納入Git，原始checkpoint保留本機。

建議閱讀順序：

1. [量化整合流程與Paper-TWN逐區規劃](2026-09-04-integrated-quantization-and-paper-twn-plan.md)：保留既有主線，新增backbone→neck→head階段鎖定、三條recovery lanes與TWN版本校正。
2. [Paper-TWN第一手文獻與逐區實驗契約](../research/2026-09-04-paper-twn-primary-literature.md)：TWN版本、filter-wise QAT、TTQ、INQ與exact ternary的原始來源邊界。
3. [V5 qSiLU＋A8九區W8敏感度](2026-09-03-v5-nine-region-w8-sensitivity.md)：8格green、attention-safe recover及isolated Pareto角色。
4. [V5 mAP50總門檻與backbone early bit邊界](2026-09-03-v5-map50-bit-boundary-search.md)：activation已納入`-0.015`總budget，W8通過、W7 recover、W6–W4淘汰。
5. [V4 qSiLU＋A8 backbone early W8搜尋驗證](2026-09-03-v4-qsilu-backbone-early-w8-search-validation.md)：凍結的mAP50–95歷史判定與三角色raw evidence。
6. [架構與實驗配置稽核](2026-09-01-architecture-experiment-audit.md)：現行架構、已修問題、分層矩陣與剩餘blocker。
7. [量化文獻與研究新意稽核](../research/2026-09-01-quantization-literature-and-innovation-audit.md)：33組第一手來源與可否證研究假說。
8. [Hardswish policy修訂](2026-08-31-hardswish-policy-revision.md)：現行active矩陣、uniform／regional證據、CPU產物與停止線。
9. [Q3 activation parent證據查核](2026-08-31-q3-activation-parent-evidence.md)：指定GitHub來源、selector差異與可／不可取用邊界。
10. [V0–V3 CPU實作與靜態分析](2026-08-30-v1-v3-cpu-implementation.md)：凍結歷史weight結果、十個region、SD4 routing與V3 gate。
11. [Activation預先選擇報告](2026-08-29-activation-preselection-report.md)：歷史老師版選擇依據；現行shortlist已由2026-08-31修訂取代。
12. [Activation-output無訓練smoke報告](2026-08-29-activation-output-smoke-v2.md)：30格矩陣、graph契約與原始限制。
13. [Weight region重新規劃](2026-08-29-weight-region-replan.md)：歷史V1 region與PTQ設計。
14. [backbone_early PTQ v1](2026-08-29-weight-ptq-backbone-early-v1.md)：歷史6格診斷，不是完整validation。

NRMSE與TopK overlap都是工程proxy，不是mAP；舊`passed`也不代表精度gate通過，現行名稱為`diagnostic_pass`。現行v5使用八項mAP50與`-0.015`總門檻，且包含activation替換。qSiLU＋A8的十區W8 isolated sensitivity已完成：九區green，只有`backbone_attention_safe` recover；`backbone_early` W7 recover、W6–W4淘汰。Paper-TWN舊Pose groups已有mAP並失敗，但backbone／neck／Detect逐path與ternary QAT尚未完成；LS-SD4、A-SD4亦尚無最終結果。
