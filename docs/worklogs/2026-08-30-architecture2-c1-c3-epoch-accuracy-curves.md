# 2026-08-30：architecture_2 C1～C3 逐回合精度曲線

## 任務與範圍

本次只把已完成且已封存的 Float20 C1、C2、C3 每回合驗證指標畫成可讀曲線。

- 不重新訓練、不啟動 queue、不使用 GPU。
- 不重新選擇 C_best，也不恢復 full、PTQ 或 QAT-lite。
- 不改寫正式 Float20 報告、manifest、hash、checkpoint 或資料集。

## 變更內容與原因

- 新增 `scripts/plot_float20_epoch_accuracy.py`，逐一驗證 C1～C3 各 20 份 Float metrics。
- 輸出四面板 PNG 與 60 筆 CSV，分開呈現 COCO Detect mAP50-95、BBAT5 Pose box
  mAP50-95、BBAT5 keypoint mAP50-95 與 BBAT5 keypoint Macro F1。
- 新增 `results/supplementary/full35-j3-float20-seed0/README.md`，記錄最高點與讀圖限制。
- 更新 architecture_2 的 README 與 RUNBOOK，加入圖表入口及 CPU 重畫命令。
- `.gitignore` 只放行這一張補充 PNG，其他資料集／訓練影像仍保持忽略。
- 未修改 `results/full35-j3-float20-seed0/` 的任何正式交付檔。

## 驗證方式與結果

- CPU 重畫：成功讀取 60 筆指標，C1、C2、C3 各 20 回合。
- CSV：共 61 行（表頭 1 行＋資料 60 行），epoch 集合均為 1～20。
- 圖片：PNG 2399×1618；已視覺確認中文字、圖例、四面板、曲線與最佳點星號可讀。
- `ruff check scripts/plot_float20_epoch_accuracy.py`：通過。
- `python -m py_compile scripts/plot_float20_epoch_accuracy.py`：通過。
- 以空白 run root 測試：因 C1 為 0／20 份 metrics 而拒絕，fail-closed 行為通過。
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q`：94 passed、1 skipped、1 failed。

所有本次新增繪圖路徑均以 `CUDA_VISIBLE_DEVICES=''` 執行，沒有啟動 CUDA。

## 資料處理紀錄

- 本次只讀取 `artifacts/runs/full35-j3-float20-seed0/*/validation/epoch-*/float/metrics.json`。
- 未更動 BBAT5、COCO、影像、標註、split、assignment、patch manifest 或任何資料來源。

## 困難與解法

- 系統的 `apply_patch` 因 `bwrap: loopback: Failed RTM_NEWADDR` 無法讀檔；改以標準
  `patch` 套用相同 unified diff，且逐段核對，沒有覆寫既有檔案。
- 第一版檢查時發現 Macro F1 誤取 Pose box route；正式報告使用 Pose keypoint route。交付前已改為
  `pose.keypoints.f1.macro_f1`，重畫 PNG／CSV，並逐一對回正式報告的第 20 回合數值。
- 完整 pytest 唯一失敗是根層 `configs/datasets/bbat5-v1.yaml` 目前缺少既有測試要求的
  `policy.publication_root`；這與本次繪圖無關，因此沒有擴大範圍修改全域資料集政策。

## 未解事項或風險


- 根層 dataset registry 的上述既有契約失敗仍待資料集治理工作另行處理。
- C2 keypoint Macro F1 於第 17 回合最高 0.839794，第 20 回合為 0.839558；C3 於第 19 回合
  最高 0.722234，第 20 回合為 0.714643。
- 本圖只比較 C1～C3 的歷史初篩走勢；正式「相對 C0 降幅過大、不採用」結論維持不變。
- 其他未解事項：無。
