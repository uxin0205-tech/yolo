# Full35 小型參考與外部依賴

此處供舊方法與結果追溯，不是 Stage7 的模型血緣。現行工作與交接請使用 [Stage7](../stage7/README.md)。

[evidence.json](evidence.json) 保存已移除舊 run 的 JSON 指標／設定、epoch／gate／validation CSV、runtime manifest 與來源 SHA。逐 macro／event 只保留首尾及診斷摘要，預覽圖、資料 views、cache、未使用 queue 與錯誤嘗試目錄已刪；不是保存整套 legacy。

既有結論與數學圖表仍見 [Full35 最終分析](../../reports/full35-activation-final-analysis.md)、[machine-readable 結果](../../reports/full35-activation-results.json)、[整合報告](../../reports/completed-activation-integrated-report.md)、[圖表](../../reports/figures/README.md)、[數學推導](../../docs/research/full35-activation-mathematical-derivations.md)。舊來源中的 artifacts 路徑可能屬已清理歷史；可按相對路徑從 evidence.json 查找抽取紀錄，不應將那些舊 run 當成可 resume 工作。

下列四份舊權重仍被 `yolo_quantize` 配方、intake 與 adapter 使用，保留原路徑與 SHA：

- `artifacts/runs/full35/short-recovery-v2-lr01-uniform-qsilu-pq-seed1/inference/best_joint.pt`
- `artifacts/runs/full35/short-recovery-v2-lr01-uniform-hardswish-seed1/inference/best_joint.pt`
- `artifacts/runs/full35/short-recovery-v2-lr01-uniform-poly-shift-seed1/inference/best_joint.pt`
- `artifacts/runs/full35/short-recovery-v2-lr01-uniform-poly-quality-seed1/inference/best_joint.pt`

另保留明確引用的 experiment／resolved-config／metrics／gate 檔。canonical `src/activation_lab` 與既有 [release/weights](../../release/weights/README.md) 維持其他專案可用。精確受保護清單與刪除紀錄在 [Stage7 cleanup](../stage7/cleanup-20261103.json)。
