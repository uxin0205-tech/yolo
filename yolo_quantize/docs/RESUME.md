# 日後恢復量化研究

這是 `5090 Stop 0908` 的接手入口。**目前量化延後，沒有待啟動訓練，也不應重啟舊 queue。** 使用者明確要求恢復後，才重新確認下一批範圍；保留的舊命令不是執行授權。

## 已完成與尚未完成

已完成同 V36 parent 的 592 組單層輸出 probes、40 組十區 PTQ、六組各 5 epochs QAT、12 個逐區累積 PTQ。592 組不是逐層 mAP；CPU 權重重建誤差也不是 mAP。148 個部署 Conv/Linear 權重路徑都有量化方案、124 個 qSiLU／LSQ+ A8 activation quantizers，但保護算子與浮點執行邊界仍在，不能宣稱完整純整數部署。

尚未完成：擴展逐層 mAP、額外累積 QAT、finalists 選定、最終 export／formal、實際整數 kernel 與硬體 benchmark。保留兩個精度／成本取捨點，**沒有 final winner**。完整結果見[階段收尾](reports/2026-09-08-quantization-phase-handoff.md)及[QAT 恢復分析](reports/2026-09-07-continuous-qat-recovery-results.md)。

## 接手先讀的機讀證據

| 資料 | 作用 |
| --- | --- |
| [phase-hold.json](../artifacts/queues/full-model-cumulative-0908/phase-hold.json) | 最新延後決策，優先於歷史 execution status；它不是 OS／CLI 鎖 |
| [parent-manifest.json](../artifacts/queues/full-model-continuous-0907/parent-manifest.json) | V36 起點、EMA／export 血緣與校驗 |
| [六組清單](../artifacts/queues/full-model-continuous-0907/selected-qat-jobs-v2.json) | 已執行 QAT 方法與範圍，不是待跑 queue |
| [QAT 結果摘要](../artifacts/queues/full-model-continuous-0907/qat-recovery-summary.json) | 各方法 PTQ→QAT 恢復與限制 |
| [累積計畫](../artifacts/queues/full-model-cumulative-0908/plan.json) | 已執行的階段與比較前綴 |

原 `execution-status.json` 保留 `decision_required`，不是錯誤、不是運行中，也不代表自動接 QAT。其他歷史 queue 可能仍有當時的 running／PID 字段；GitHub 全部只作凍結快照，不能據此判斷現在 GPU 狀態。monitor 已在終點退出，本次不重新啟動。

## 必須繼續保留的本機資產

- 模型來源：`/home/uxin/yolo/yolo_combine/final/full35/`，唯讀。
- V36 起點：`artifacts/runs/qat/v36-qsilu-full-coverage-short-v2/v36-qsilu-full-coverage-short-qat-v1-qat-seed1/` 下 `inference/best_joint.pt` 與 `checkpoints/best_joint.pt`，分別是部署 export 和 EMA full-resume，不可互換。
- 六組 QAT：`artifacts/runs/qat/continuous-0907-special-v1/`。保留 checkpoint、逐回合 metrics、設定與血緣；`last` 不一定等於選出的 `best_joint`。
- 其他 `artifacts/runs/`、`artifacts/datasets/`、既有 queue、原始 JSON／CSV 與資料集不因本次清理而刪除。
- BBAT5：不可變 `/home/uxin/yolo/original/pose/derived/bbat5-v1/`，registry `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`；Pose `configs/pose.yaml`、ball/bat Detect `configs/detect.yaml`。COCO80／Person 為 `/home/uxin/yolo/coco2017.yaml`。沿用原 assignment／labels，不能重切或自行新增抽樣。

主要校驗值（SHA-256；仍以 manifest 與原檔驗證為準）：

| 項目 | SHA-256 |
| --- | --- |
| parent manifest | `f25a3caaebe6b7d51663d615c0202be7dedec431683cb0ad0c8fe9dccfc3d48a` |
| V36 inference/best_joint.pt | `8c1f3652fd21c0e38221cfc3acba7ee013c5e88df641b9331c33796f6f11b66c` |
| V36 checkpoints/best_joint.pt | `f83e1ba22adc737ed05abf68152f621a67915a2104ace3bfd5a3b630d66dcb78` |
| 累積 plan.json | `c2e83797f85e858cf6cec7c897dd75f22579e7b49823a4e8eda461e5ac7982e1` |

GitHub 保存程式、設定、報告、圖表與凍結 metadata；**沒有 checkpoint、完整 runs、資料集副本、訓練 log 或大型逐影像 predictions**。在原機可承接既有資產；換機只 clone 並不足以恢復，還需要另外移交上述資料與權重，先核 SHA 再使用。此次沒有替本機資產建立異地備份。

## 明確恢復後的順序

1. 確認最終模型是否仍是 V36。若換 parent，舊結果只作候選提名依據，不能移植 mAP 結論；重新鎖定 parent／export、activation、全模型覆蓋與保護邊界。
2. 按[同 benchmark 契約](CURRENT_PLAN.md)對齊資料版本與 split、16 指標、evaluator／前後處理、校正預算；QAT 再對齊初始化、訓練預算、超參數與 checkpoint 選取規則。這是目前的書面契約，尚未新增自動 gate。
3. 承接 SD4／LS-SD4／三元的分布假說及已有恢復證據，按同一前綴比較 backbone→neck→head；不直接拼接各層獨立最優。必要時再 W6／W5／W7／W4；PTQ 不佳可以提名少量 QAT，不宣稱必然回升。
4. 重新提名少數候選及時間預算；既有 5 epochs／patience 5／AdamW 只作已驗證起始配置，不重跑已完成六組、不另訓 baseline，除非新起點或比較目的確有必要且獲授權。
5. 新建版本化 queue／輸出目錄，保留舊證據，驗證完成／異常事件與下一步規則。獲准訓練後再使用 600 秒 shell blocking monitor；不得把 shell 等待說成對話結束後仍能自動喚醒模型修復。
6. 最後才選最多兩個 finalists，重載 export 並跑 formal 與部署 benchmark；同時報 mAP50、mAP50–95、量化覆蓋、成本與真實硬體限制。

目前門檻：相對 accepted 的 8 項 mAP50 各下降 ≤1.5 pp，8 項 mAP50–95 各下降 ≤4 pp，包含 activation 替換與 COCO Person；未來若需求變更，另留決策，不回寫舊數字。
