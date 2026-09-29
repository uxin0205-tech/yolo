# Artifact 最終清理紀錄

日期：2026-08-16

本次依使用者指示，將 repository 整理為「程式、正式結果 provenance、報告與必要 checkpoint」版本。訓練中間產物永久刪除，不以空目錄或重複權重占用空間。

## 保留原則

保留能直接啟動基準、主線最終模型與 BDCN 最終分支的最小 checkpoint 集合：

- `weights/yolo26m.pt`：官方 YOLO26m baseline。
- `artifacts/runs/v1-br/ultralytics/weights/best.pt`：A0 與 normalization／BDCN 共用 parent。
- `artifacts/runs/n1-shift/ultralytics/weights/best.pt`：queue-selected A-FINAL。
- `artifacts/runs/bdcn-v3-learn/ultralytics/weights/best.pt`：V3-R1 的實際 parent 與 anchored codebook checkpoint。

保留檔案的 SHA-256：

- V1-BR：`9e2ca0d93793785f0f7e514d876580204070bac57b4163d7c5f0c5ca352b9c3f`。
- N1-SHIFT：`6bd4b25ea36bfe453f437cec6647d7d7dbab745e1e0035b1610928a454e63ce1`。
- BDCN-V3-LEARN：`0bbd0684affaf507d816a5c94fffd39fcd528436dcfc9ab519294ebcd4176506`。
- YOLO26m：`401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`。

## 永久刪除內容

- 所有 `last.pt`，因正式比較與後續 parent 都使用 best checkpoint。
- H-SCR、W-DIR、D1-SHARED-10 的階段權重；其 metrics、CSV、manifest 與報告仍保留。
- BDCN-V2-LEARN 的 best/last 權重；V2 是已確認的 table-drift 負結果。
- Ultralytics curve、batch、label、prediction、confusion-matrix 圖與 worker logs/lock。
- calibration、invalidated、非 winner checkpoint 與空的 checkpoints/exports/logs/weights/ultralytics leaf。
- Python、pytest、Ruff、egg-info、patch backup 與其他 cache。
- 非正式 `yolo26n.pt`。

專案從最初約 3.9 GB，經主線與 BDCN 兩次整理後降至約 248 MB。正式 metrics、manifest、variant/training YAML、analytical profile、`results.csv`、queue state/events、COCO 設定、研究報告、必要 checkpoint 與論文來源均保留。

## Provenance 限制

`TRAINING_AUDIT.md` 的 checkpoint load/finite 結論是在清理前完成；被刪除的權重不能再從目前資料夾直接載入，但其正式數值與 provenance 沒有刪除。`checkpoint_sizes.csv` 與對應圖是清理前的實際量測，不代表所有列出的檔案仍被保留。

安全 postprocess retry 原本要求同時存在 `best.pt`、`last.pt` 與完整 `results.csv`。清理後的既有 runs 被視為 archived completed results，不再支援原地 retry；若要重跑，必須建立新的唯一 `run_id`，不得偽造或覆寫歷史 run。

BDCN V2/V3 的最終結果與保留 V3 checkpoint 用途見 [BDCN_V3_REPORT.md](BDCN_V3_REPORT.md)。
