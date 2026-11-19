# Binary Q/K 與固定 scale／bias 研究

**第 2 階段已完成，使用者已指定第 3 階段接續。** 2026-10-19 04:58（Asia/Taipei）完成 S1（Q10）／S2（二次冪）各五輪，總耗時約 4 小時 23 分鐘。兩組從 [result/stage1](result/stage1/README.md) 的同一 B2 獨立開始，官方 AP50–95 分別為 **0.514152／0.513711**，依事前規則選 S1；B2 為 0.509889。Q/K 二值化保留，加入固定 scale／共同分解式 bias。ball／bat 未超過 B2，沒有宣稱所有類別改善。共用算子位於 `yolo_attention`，本輪結果在 `artifacts/stage2/`；選定 S1 已搬至 [result/stage2](result/stage2/README.md)，新研究位於 [yolo_attention](../yolo_attention/docs/stage3-protocol.md)。

- [第 2 階段 protocol](docs/stage2-protocol.md)、[來源核對](artifacts/stage2/reference-review.json)、[啟動紀錄](artifacts/stage2/formal-launch.json)。
- [正式結果與負結果](artifacts/stage2/formal-20261019-v1/RESULTS.md)、[八項 AP CSV](artifacts/stage2/formal-20261019-v1/comparison.csv)、[曲線](artifacts/stage2/formal-20261019-v1/curves.svg)、[完成工作紀錄](../docs/worklogs/2026-10-19-binaryqk-stage2-results.md)。
- [隊列完成狀態](artifacts/stage2/formal-20261019-v1/supervisor.json)、[每 600 秒監測](artifacts/stage2/formal-20261019-v1/monitor.jsonl)、[下一階段 handoff](artifacts/stage2/formal-20261019-v1/handoff.json)。
- [驗證摘要](artifacts/stage2/verification.json)：Binary 50 項、Attention 168 項通過／1 項略過，兩組 GPU smoke 通過；result 搬移後 6 項相關測試通過。[實際交付驗證](artifacts/stage2/formal-20261019-v1/handoff-verification.json) 已核對最佳權重、CPU 重建／前向、常數、SHA、共同 parent、完整資料與監測。
- 新入口：`PYTHONPATH=yolo_binaryqk:yolo_attention/src .venv/bin/python -m binaryqk.scale --help`。兩組結果已完成；第 3 階段依使用者新指示從 `result/stage2/` 接續，本資料夾不重跑 Stage 2。

## 第 1 階段已完成

本輪只比較 B1（sign＋residual）、B2（sign＋Hadamard）、B3（sign＋residual＋Hadamard），三組從相同 YOLO26m COCO80 Detect parent 獨立出發。每項工作前先參考 optimize，來源、實際設定與差異見 [事前核對](artifacts/stage1/reference-review-v2.json) 及 [現行 protocol](docs/stage1-protocol.md)。

**第 1 階段已於 2026-10-18 23:10（Asia/Taipei）完成，並由使用者指定接續第 2 階段。** B1／B2／B3 各 10 epochs Attention 初篩最佳官方 AP50–95 為 **0.503966／0.509889／0.506355**，依事前規則選 B2。其全模型恢復實際執行 5 epochs，未超過初篩最佳，依 patience 5 提前停止；目前交付位置為 `result/stage1/epoch-00.pt`，權重等於 B2 初篩第 10 epoch，搬移 SHA 全部相同。

正式隊列共約 **10 小時 24 分鐘**，因恢復提前停止，早於原先跑滿 40 epochs 的 ETA。B2 官方 AP 仍比原始 parent 0.524686 低 1.48 個百分點；internal AP 為 0.503328，勿與官方指標混比。初篩勝出不代表已恢復原始精度或已完成硬體部署。

- [正式結果與負結果分析](artifacts/stage1/formal-20261018-v2/RESULTS.md)、[八項 AP 與成本 CSV](artifacts/stage1/formal-20261018-v2/comparison.csv)、[曲線](artifacts/stage1/formal-20261018-v2/curves.svg)。
- [B2 運算量核對](artifacts/stage1/formal-20261018-v2/COST.md)：640 × 640 下整網浮點 MAC 小計減少 0.109%；二值與其他新增運算另計，未宣稱實測加速。
- [下一階段 handoff](artifacts/stage1/formal-20261018-v2/handoff.json)、[交付驗證](artifacts/stage1/formal-20261018-v2/handoff-verification.json)、[完成工作紀錄](../docs/worklogs/2026-10-18-binaryqk-stage1-results.md)。

- [目前隊列狀態](artifacts/stage1/formal-20261018-v2/supervisor.json)、[每 600 秒監測](artifacts/stage1/formal-20261018-v2/monitor.jsonl)。
- [啟動命令與來源 SHA](artifacts/stage1/formal-launch-v2.json)。
- [事前方法核對](artifacts/stage1/reference-review-v2.json)：optimize 後期 A0 與上游初次 H-SCR／W-DIR 分開核對。
- [44 項 CPU／GPU 驗證摘要](artifacts/stage1/verification-v2.json)、[四項 GPU 短測](artifacts/stage1/recovery-v2-smoke-summary.json)：不是正式 AP。
- [共同校準係數](artifacts/stage1/calibration.json)、[真實 score／P／P·V 診斷](artifacts/stage1/score-diagnostics-v2.json)。
- [原始 parent 完整重驗](artifacts/stage1/protocol-audit-20261018/parent-evaluation.json)：internal 0.518002／官方 COCO 0.524686；沒有加訓 baseline。
- [錯誤 v1 直接刪除清單](artifacts/stage1/cleanup-invalid-v1.json)、[成功短測暫存權重清理](artifacts/stage1/cleanup-smoke-v2.json)：沒有 archive／backup。
- [中文工作紀錄](../docs/worklogs/2026-10-18-binaryqk-recovery-v2.md)。

舊 Q/K-only／`5e-7` 配方與其 run／權重已直接移除。必要的負結果數據及理由仍可在 [配方稽核](docs/protocol-audit-2026-10-18.md) 查閱，不作另一套可執行方案。

## 入口

以下從 repository 根層執行。正式隊列已完成；這些是查詢與驗證入口，不代表新訓練授權。新 run 不覆寫既有輸出。

```bash
PYTHONPATH=yolo_binaryqk .venv/bin/python -m binaryqk.cli --help
PYTHONPATH=yolo_binaryqk .venv/bin/python -m binaryqk.train --help
PYTHONPATH=yolo_binaryqk .venv/bin/python -m binaryqk.supervise --help
CUDA_VISIBLE_DEVICES='' PYTHONPATH=yolo_binaryqk .venv/bin/python -m pytest yolo_binaryqk/tests -q
```

Supervisor 完成後在隊列根目錄產生 `comparison.json`、`comparison.csv`、`curves.svg`、`RESULTS.md`、`handoff.json`。圖表可用 `python -m binaryqk.report --plot <隊列目錄>` 僅讀 JSON 重畫，不需 GPU 或 checkpoint。

`--continue-queue` 可續接中斷隊列，需來源 SHA 不變、合法 resume、完成結果可驗證；GPU 忙碌每 600 秒等待重查，不終止他人工作。隊列根目錄的 `STOP` 會在 batch 邊界停止，評估先完成當次工作。短測的可重建權重已清除，重做短測須用新輸出目錄，不能將短測當正式 parent。

`binaryqk/` 是唯一研究實作。`binary_attention/` 僅提供受保護歷史正式權重的最小重建相容層。既有必要權重與 optimize 外部依賴保持原位；正式資料、BBAT5 v1、Pose、未指定階段及唯讀 `yolo_optimize` 不改動。

整體研究與最終硬體契約見 [整合計畫](../docs/research/yolo-research-plan-2026-09-21.md)。RepConv 最終必須繼承 qSiLU、已選 normalization／PWL／reciprocal；本階段不能把 PyTorch reference 當硬體加速量測。未 commit 或發布。
