# Clean Operations

CPU inspect：

```bash
CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest -q tests/clean
CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m masf_yolo.clean.inspect \
  --config configs/clean/clean_ablation.yaml \
  --output configs/clean/FEASIBILITY.json
```

`masf_yolo.clean.queue` 是唯一正式單 GPU 排程入口。它先執行 12-architecture CUDA acceptance，再依序執行 12 smoke、24 formal 與 4 control stages；每個 stage 寫入 state/result，重啟時跳過已完成 job，若存在 `last.pt` 則續跑。`masf_yolo.clean.worker` 只負責單一 job。

訓練完成後的唯一完整後處理入口：

```bash
../.venv/bin/python -m masf_yolo.clean.postprocess all
```

它依序執行 28-checkpoint audit/fresh reload、28 validation、14 hardware profiles、selection freeze、28 historical test 與中文報告；具 process lock、hash 檢查與逐項續跑。test 在 selection freeze 前 fail closed。

資料來源與舊工具保留在 `bbt5-detect-baseline/`；正式 locked evidence 位於 `artifacts/locked-bbt5-dataset/`；正式結果只寫入 `artifacts/clean-bbt5-ablation/` 與 `clean_bbt5_study/results/`。
