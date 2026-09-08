# 前批完整執行證據：已完成、保留

2026-09-08：本批六組 QAT 已完成，後續十二個累積 PTQ 也已完成；依使用者決定延後量化，不再接新訓練。最新決策見[階段收尾](../../../docs/reports/2026-09-08-quantization-phase-handoff.md)，不重啟本批。

來源為V36 best_joint epoch index1；parent-manifest.json同時釘住完整resume與export，parent-preflight.json記錄16指標與獨立前向parity。projection-parity.json為明確裝置上的EMA投影比對，CPU/GPU投影不能要求混為同一數值路徑。

weight-distribution.json是148路兩view原權重／export分布；operator-precision.json為固定64x64兩任務的ATen精度觀測，不是純整數部署證明。generated/isolated-special-plan.json是40格同parent計畫，每格只改一區，不改其他層。

40 格 PTQ、592 格單層輸出探測及 `selected-qat-jobs-v2.json` 的六組 5epoch QAT 均已完成。原 runner 與清單保留為重現入口，不是目前待啟動工作；不重啟舊 supervisor。

| 檔案 | 正確用途 |
| --- | --- |
| `execution-status.json` | 本批歷史終點狀態，不作現行 monitor |
| `parent-manifest.json` | V36 inference／full-resume 的共同血緣 |
| `single-layer-probe.json` | 148 × 4 輸出探測，不是逐層 mAP |
| `isolated-special-report.json` | 40 組原始 PTQ；舊 reference／selection 欄位的限制見報告 |
| `special-dual-summary.json` | 16 指標 PTQ 雙門檻；不要改讀舊 mAP50-only selection |
| `selected-qat-jobs-v2.json` | 已完成的六組計畫與 hash；v1 保留歷史 |
| `qat-recovery-summary.json` | 已完成 QAT 的逐回合／逐指標比較 |
| `evidence-consistency-audit.json` | 本輪 parent 血緣稽核，不是重新 accuracy validation |

日常看[人讀結果](../../../docs/reports/2026-09-07-continuous-qat-recovery-results.md)及[目前計畫](../../../docs/CURRENT_PLAN.md)。log 只在錯誤事件時診斷，不日常輪詢。不要直接修改被計畫釘住的 JSON 或搬移 checkpoint。

本目錄為本機live queue證據，預設不整包公開；checkpoint、data、log與暫存檔不隨報告發行，後續可擇出結果摘要。歷史V36與資料assignment不改動。
# 本批已完成，現行監測已移交

2026-09-08：六組各 5 epochs 全部完成。以下保留本批來源與執行紀錄；現在請看[累積 PTQ queue](../full-model-cumulative-0908/README.md)，不要重啟本批訓練。
