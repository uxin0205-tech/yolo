# 整理盤點與待核准刪除清單

此清單尚未執行刪除。容量為盤點快照，訓練可能持續新增檔案。

## 建議清除／審慎確認

| ID | 精確路徑 | 類型 | 檔數／bytes | 未使用依據及建議 | 可恢復性 | 風險 |
| --- | --- | --- | ---: | --- | --- | --- |
| C01 | `/home/uxin/yolo/yolo_quantize/.pytest_cache` | 可重建快取 | 5／34680 | 待使用者核准後刪除；訓練期間可先保留 | 可重建，不保證原快取逐位元恢復 | 低；測試或 import 時可能再生成，執行前須重新核對 |
| C02 | `/home/uxin/yolo/yolo_quantize/.ruff_cache` | 可重建快取 | 6／8550 | 待使用者核准後刪除；訓練期間可先保留 | 可重建，不保證原快取逐位元恢復 | 低；測試或 import 時可能再生成，執行前須重新核對 |
| C03 | `/home/uxin/yolo/yolo_quantize/scripts/__pycache__` | 可重建快取 | 8／67339 | 待使用者核准後刪除；訓練期間可先保留 | 可重建，不保證原快取逐位元恢復 | 低；測試或 import 時可能再生成，執行前須重新核對 |
| C04 | `/home/uxin/yolo/yolo_quantize/tests/__pycache__` | 可重建快取 | 67／1266806 | 待使用者核准後刪除；訓練期間可先保留 | 可重建，不保證原快取逐位元恢復 | 低；測試或 import 時可能再生成，執行前須重新核對 |
| C05 | `/home/uxin/yolo/yolo_quantize/src/yolo_quantize/__pycache__` | 可重建快取 | 55／1299374 | 待使用者核准後刪除；訓練期間可先保留 | 可重建，不保證原快取逐位元恢復 | 低；測試或 import 時可能再生成，執行前須重新核對 |
| C06 | `/home/uxin/yolo/yolo_quantize/tests/test_qat_runtime.py.orig` | 歷史修改備份 | 1／10975 | 審慎確認；未核准前保留 | 未確認可靠備份，不保證可恢復 | 中；不是現行 pytest 檔，但可能有唯一歷史內容 |

C01–C05 是工具生成快取，不承載模型權重或原始資料；C06 不會被 pytest 直接發現，但不代表沒有歷史價值。

## 必須保留

| 路徑 | 原因 |
| --- | --- |
| `artifacts/queues/` | 正在執行的 queue 與歷史 handoff、plan/hash 引用；不能移動 |
| `artifacts/runs/` | V36 parent、V35 外部 sham、V4 accepted 指標與部署/續跑 checkpoint；未逐一證明其他 run 可刪，全部先保留 |
| `artifacts/manifests/`、`configs/` | 資料/模型/方法血緣與雜湊契約；舊版本可能仍是依賴 |
| `artifacts/reports/`、`deliverables/`、`docs/worklogs/` | 公開數字、負面實驗、決策與復現證據 |
| `src/`、`tests/`、資料集、PDF | 實作、回歸、不可變來源與方法出處 |

## 資料夾容量

| 頂層 | 一般檔 bytes | 檔案／symlink 數 |
| --- | ---: | ---: |
| `.gitattributes` | 212 | 1／0 |
| `.gitignore` | 92 | 1／0 |
| `.pytest_cache` | 34680 | 5／0 |
| `.ruff_cache` | 8550 | 6／0 |
| `IMPLEMENTATION_PLAN.md` | 22883 | 1／0 |
| `PUBLICATION_0907.yaml` | 923 | 1／0 |
| `PUBLICATION_MANIFEST.yaml` | 3798 | 1／0 |
| `README.md` | 3582 | 1／0 |
| `artifacts` | 58451109989 | 228165／226632 |
| `configs` | 310237 | 60／0 |
| `deliverables` | 1824802 | 21／0 |
| `docs` | 1289990 | 71／0 |
| `pyproject.toml` | 1770 | 1／0 |
| `quantize_spec.md` | 80661 | 1／0 |
| `requirements-report.txt` | 19 | 1／0 |
| `scripts` | 176712 | 28／0 |
| `src` | 2345263 | 111／0 |
| `tests` | 1614972 | 128／0 |
| `三元權重變形器神經網路加速電路之設計與晶片實現.pdf` | 10461095 | 1／0 |

現行入口引用閉包記錄 834 個既有路徑；詳見 [JSON 清單](inventory-2026-09-08.json)。
這不是未引用檔案可刪的證明；動態 import、外部專案、已發表結果與其他使用者都可能使用它們。

核准時請指定 C01–C05 或個別 ID；不將「全部」擴張到未列為刪除候選的 runs／checkpoint。
