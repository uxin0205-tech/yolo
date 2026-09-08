# 報告精簡：已核准清單與執行結果

狀態：**2026-09-08 使用者核准後已執行。** R01–R03 已備份後刪除；M01–M12 已封存；K01–K10 保留原位。依 finish-work 先列精確對象再執行。此次不刪 checkpoint、原始結果、訓練 log、資料集、PDF、圖表、研究來源或工作紀錄。

備份位置：`/tmp/yolo-quantize-report-backup-0908.rJlv9s/`，原 25 份報告 SHA 均核對通過。根目錄報告已由 25 份降至 10 份；12 份歷史報告保留原文數值，22 份文件共修正 76 處連結。見[執行證據](report-retention-execution-2026-09-08.json)。下列表格與執行規則保留為**核准當時的提案快照**，其中「待核准／尚未備份／不 commit」不代表本次最新狀態；最新使用者另授權以 `5090 Stop 0908` 發布。

目前有 25 份報告（不含 README）。建議日常只列 5 份，另保留 5 份有必要依賴的原位文件；3 份冗餘刪除、12 份獨有歷史成果移入 `docs/archive/reports/`。核准並完成後，`docs/reports/` 根層將由 25 份降為 10 份，不把弱結果或負結果直接抹掉。

## R：建議刪除，須先備份與核准

| ID | 精確路徑 | 類型／bytes | 已被何者涵蓋 | 可恢復性／風險 |
| --- | --- | ---: | --- | --- |
| R01 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-07-continuous-five-epoch-plan.md` | Markdown／4264 | 5-epoch 與四天規劃已由 CURRENT_PLAN、實際六組結果及 phase-hold 取代；舊 planned_not_enqueued 會混淆進度。 | 執行前另建 /tmp SHA 備份；目前尚未備份。低至中：僅冗餘敘述；先核對引用、建立備份，不能靠 Git 保證恢復。 |
| R02 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-08-29-weight-region-replan.md` | Markdown／6215 | 舊 151-layer、poly_quality 與 20-epoch 規劃已被取代；A3–A8 數值另有保留的 activation 報告，現行 148 路徑與比較契約已集中。 | 執行前另建 /tmp SHA 備份；目前尚未備份。低至中：僅冗餘敘述；先核對引用、建立備份，不能靠 Git 保證恢復。 |
| R03 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-07-ptq-to-qat-recovery-interpretation.md` | Markdown／2572 | PTQ 不等於 QAT 最終精度的說明已整合到方法／收尾報告，具體恢復結果另有完整六組報告。 | 執行前另建 /tmp SHA 備份；目前尚未備份。低至中：僅冗餘敘述；先核對引用、建立備份，不能靠 Git 保證恢復。 |

三份合計 13,051 bytes，目的是減少閱讀負擔，不是釋放大量磁碟。先核准、再備份；不保證未追蹤檔案可由 Git 恢復。此 R 編號只對本清單生效，不包含先前的 cache／其他刪除提案。

## M：審慎封存，保留原文數值與獨有證據

| ID | 精確來源 | 精確目的地 | 類型／bytes | 必须保留的原因 |
| --- | --- | --- | ---: | --- |
| M01 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-08-29-weight-ptq-backbone-early-v1.md` | `/home/uxin/yolo/yolo_quantize/docs/archive/reports/2026-08-29-weight-ptq-backbone-early-v1.md` | Markdown／5412 | 早期 W8／W4 六格 probe 與 collapse 負結果，不能當現行 mAP。 |
| M02 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-08-30-v1-v3-cpu-implementation.md` | `/home/uxin/yolo/yolo_quantize/docs/archive/reports/2026-08-30-v1-v3-cpu-implementation.md` | Markdown／11827 | 8,880 筆歷史 CPU 分析、catalog 修訂與舊發行沿革。 |
| M03 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-08-31-hardswish-policy-revision.md` | `/home/uxin/yolo/yolo_quantize/docs/archive/reports/2026-08-31-hardswish-policy-revision.md` | Markdown／11157 | Hardswish／poly_quality 政策變更及 CPU 證據，保留原日期和限制。 |
| M04 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-01-architecture-experiment-audit.md` | `/home/uxin/yolo/yolo_quantize/docs/archive/reports/2026-09-01-architecture-experiment-audit.md` | Markdown／12910 | 舊 catalog／BN fold／grid solver／formal leakage 的修正依據。 |
| M05 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-01-cpu-p0-integer-exact-routing.md` | `/home/uxin/yolo/yolo_quantize/docs/archive/reports/2026-09-01-cpu-p0-integer-exact-routing.md` | Markdown／7321 | 整數邊界及 exact W4／SD4 routing 的 CPU 工程證據。 |
| M06 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-03-poly-shift-ptq-special-format-and-qat-entry.md` | `/home/uxin/yolo/yolo_quantize/docs/archive/reports/2026-09-03-poly-shift-ptq-special-format-and-qat-entry.md` | Markdown／8226 | poly_shift 位元搜尋、特殊格式與耦合 PTQ 的獨有結果。 |
| M07 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-03-v5-map50-bit-boundary-search.md` | `/home/uxin/yolo/yolo_quantize/docs/archive/reports/2026-09-03-v5-map50-bit-boundary-search.md` | Markdown／6633 | 早期 qSiLU 的 W7–W4 mAP 負結果與總門檻沿革。 |
| M08 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-03-v5-nine-region-w8-sensitivity.md` | `/home/uxin/yolo/yolo_quantize/docs/archive/reports/2026-09-03-v5-nine-region-w8-sensitivity.md` | Markdown／5212 | 早期九區 W8 隔離 mAP，保留同 parent 限制。 |
| M09 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-04-integrated-quantization-and-paper-twn-plan.md` | `/home/uxin/yolo/yolo_quantize/docs/archive/reports/2026-09-04-integrated-quantization-and-paper-twn-plan.md` | Markdown／12980 | 含 Paper-TWN 早期實測反例和候選設計，不只是一份排程。 |
| M10 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-04-v19-v29-progressive-quantization-execution.md` | `/home/uxin/yolo/yolo_quantize/docs/archive/reports/2026-09-04-v19-v29-progressive-quantization-execution.md` | Markdown／12457 | V19／V29 結果及 recovery gate 修復的歷史證據。 |
| M11 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-05-v29-selection-and-qsilu-handoff-plan.md` | `/home/uxin/yolo/yolo_quantize/docs/archive/reports/2026-09-05-v29-selection-and-qsilu-handoff-plan.md` | Markdown／6239 | 七組選擇依據、負 QAT、poly_shift 封存與 qSiLU 接手原因。 |
| M12 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-07-full-model-audit-four-day-plan.md` | `/home/uxin/yolo/yolo_quantize/docs/archive/reports/2026-09-07-full-model-audit-four-day-plan.md` | Markdown／11078 | V35／V36 盤點及跨 parent 證據限制；三回合草案不是目前排程。 |

十二份合計 111,452 bytes；移入專案歷史區，不永久刪除，也不宣稱釋放這些 bytes。共同風險為相對連結及來源依賴：須更新真正指向報告的連結、報告內相對路徑與歷史索引。只改位置／導航，不改實測數字、方法當時狀態或原始 artifact。若發現需要破壞 hash-bound 文件才能維持依賴，該項保留原位並回報，不自行解除來源檢查。

## K：必要報告，保留原位

| ID | 類別 | 精確路徑 | bytes | 必要性 |
| --- | --- | --- | ---: | --- |
| K01 | 日常五份 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-08-quantization-phase-handoff.md` | 11616 | 本輪完整收尾、十二候選與兩個取捨點。 |
| K02 | 日常五份 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-07-continuous-qat-recovery-results.md` | 7979 | 六組 QAT 的三十回合及逐指標 PTQ→QAT 差異。 |
| K03 | 日常五份 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-07-evidence-consistency-boundaries.md` | 4389 | 同 benchmark／同 parent 的證據分層及不可混用邊界。 |
| K04 | 日常五份 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-08-quantization-method-and-decisions.md` | 7478 | qSiLU、LSQ+、SD4／三元的選擇理由。 |
| K05 | 日常五份 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-08-cumulative-ptq-plan.md` | 4520 | 十二個累積候選的提名規則、同前綴比較與接受邏輯。 |
| K06 | 依賴保留 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-08-29-activation-output-smoke-v2.md` | 6148 | PUBLICATION_MANIFEST.yaml 有具體檔案與 SHA；不因清理移動。 |
| K07 | 依賴保留 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-08-29-activation-preselection-report.md` | 9878 | 圖表、activation-preselection-v1.yaml 及發行 manifest 引用。 |
| K08 | 依賴保留 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-08-31-q3-activation-parent-evidence.md` | 18827 | v32 條件旁支與 v1-v3-cpu-delivery-v2.yaml 釘住 report SHA。 |
| K09 | 依賴保留 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-03-v4-qsilu-backbone-early-w8-search-validation.md` | 9402 | full35-quantization-plan-v4.yaml 的 first_gpu_bridge.report 引用。 |
| K10 | 依賴保留 | `/home/uxin/yolo/yolo_quantize/docs/reports/2026-09-07-weight-distribution-detailed-analysis.md` | 12095 | 權重分布獨有數據；report_weight_evidence.py 生成的附錄仍連結此報告。 |

保留依賴文件不表示其中所有舊數字可與本輪直接相比。特別是舊 activation 發行 manifest 保存的是發行當時 hash，報告後加歷史說明後可能已有不同 hash；此次不回寫歷史 manifest 或假裝全部舊版本逐位元一致。

## 引用盤點與執行規則

[機讀清單](report-retention-2026-09-08.json)保存全部 25 份當前大小、SHA-256 及引用命中檔案／行號。初步掃描 1,565 個文字檔；以 rg 篩得 47 個有報告檔名的文件。M 組除了舊 inventory 快照外，未見程式／機讀執行引用；補查 M 組當前 hash 也未見快照以外的字面引用。這仍不是對動態或外部依賴的完全證明。

1. 核准後重新核對每個來源的 regular-file／非 symlink 狀態、hash、大小與精確目的地；目的地已存在則不覆寫。
2. 先解析引用的實際目標。工作紀錄常與報告同名，不能全域字串取代，誤改 `docs/worklogs/README.md` 等工作紀錄連結。
3. R01、R02、R03 的有效敘述入口分別導向 CURRENT_PLAN／方法與收尾報告；原始結果仍保留。歷史 inventory JSON 是盤點快照，不回寫成現況。
4. M 組建立歷史索引，必要時只調整 Markdown 連結。所有原始 JSON／CSV、queue、parent、checkpoint、datasets 與來源 hash 契約不變。
5. 完成後檢查檔案數、來源／備份 hash、現行與歷史入口的連結，記錄移除與可恢復位置。不啟動 queue／monitor／GPU，不 commit 或 push。

本次執行限 R01–R03 與 M01–M12，未擴張至 K 組或其他資料刪除；GitHub 同位置同步僅限 PUBLICATION_0908.yaml 定義的可發布內容，不上傳或刪除本機權重。
