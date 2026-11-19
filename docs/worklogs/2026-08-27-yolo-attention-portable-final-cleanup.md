# 2026-08-27：YOLO26m PWL Attention 可攜包清理

## 任務與範圍

依使用者明確授權刪除前一份清理清單中的 C01、C02、C03。範圍只包含 patch
暫存檔與可重建 Python／pytest／Ruff cache；不刪除權重、資料、實驗 runs、queue state、
metrics 或 provenance。

本次沒有改動 COCO80 或 BBAT5 v1 的影像、標註、split 與資料集設定，也沒有啟動 GPU 訓練。

## 變更內容與原因

- C01：刪除 10 個 `.orig`／`.rej` patch 暫存檔，磁碟占用 56 KB。
- C02：刪除 `final/__pycache__/` 與 `final/yolo_attention/__pycache__/`，共 144 KB。
- C03：刪除 `.pytest_cache/`、`.ruff_cache/`、`tests/__pycache__/` 與
  `src/yolo_attention/__pycache__/`，共 568 KB。
- 合計回收約 768 KB。所有項目都可由 patch、Python、pytest 或 Ruff 重新產生。

清理前已逐一路徑確認類型；10 個項目都是 regular file、6 個項目都是 directory，
沒有 symlink。

## 驗證方式與結果

- 確認所有 C01、C02、C03 精確路徑均已不存在，且子專案內沒有殘留
  `.orig`、`.rej` 或 `.pyc`。
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests -q`
  - 結果：`25 passed`。
- `ruff check --no-cache .`
  - 結果：通過。
- `ruff format --check final/run.py final/training_recipe.py tests/test_portable_final.py`
  - 結果：3 個檔案格式正確。
- `python final/run.py check`
  - 結果：正式 checkpoint SHA-256、YOLO26m scale m、80 classes、兩個 Attention sites
    與 Bit-True PWL contract 全部通過。
- `doctor_report()` 與 LR sweep state validation
  - 結果：三個必要 checkpoint hash 全部正確，19-job queue graph 通過。
- `python -m json.tool final/results.json` 與
  `bash -n scripts/build_portable_final.sh`
  - 結果：通過。

## 困難與解法

- Ruff format check 重新產生已授權的 `.ruff_cache/`；完成其他驗證後再次刪除，
  並停止執行會重建 cache 的命令。

## 未解事項或風險

- 可攜包不包含 COCO2017 圖片與標註，所以未設定資料路徑前
  `dataset_ok=false`；使用者仍需執行 `python run.py configure --data-root ...`。
- 本次只做清理與非 GPU 驗證，沒有重新執行訓練或 accuracy evaluation。
- 依清理清單明確保留 `artifacts/runs/`、`artifacts/invalidated/`、
  正式 checkpoint、training parent checkpoints 與研究 provenance。
- 根層 `docs/worklogs/README.md.orig` 與
  `docs/worklogs/2026-08-27-yolo-attention-portable-final-cleanup.md.orig`
  未列入原 C01 清單，因此沒有擴張授權刪除；它們不影響可攜包執行。
