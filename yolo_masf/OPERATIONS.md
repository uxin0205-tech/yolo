# Clean Operations

目前只允許 CPU inspect，不啟動 GPU：

```bash
CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest -q tests/clean
CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m masf_yolo.clean.inspect \
  --config configs/clean/clean_ablation.yaml \
  --output configs/clean/FEASIBILITY.json
```

`masf_yolo.clean.worker` 是唯一能啟動單一 Clean training job 的介面；確認 GPU 空閒並完成 queue 後才使用。舊 `static-phase1`、`retest` 與 data-exposed pipeline 命令不再是正式操作入口。

資料來源與舊工具保留在 `bbt5-detect-baseline/`；正式 locked evidence 位於 `artifacts/locked-bbt5-dataset/`；未來結果只寫入 `artifacts/clean-bbt5-ablation/` 與 `clean_bbt5_study/results/`。
