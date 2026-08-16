# artifacts

~~~text
artifacts/
├── queue/
│   ├── queue.json       # schema v1，唯一排程狀態
│   ├── events.jsonl     # append-only transitions/failures
│   ├── worker.lock      # non-blocking single-worker lock
│   └── generated/       # winner-derived variant YAML
├── runs/<run-id>/       # 下列每次實驗輸出
└── invalidated/          # queue rewind 保存的失效 run/generated artifacts
~~~

`queue init` 只建立 queue，不建立 run、載入 CUDA 或開始 COCO。`queue run-next` 預設也只預覽；只有明確加入 `--execute` 才執行一個 job。Queue 不可覆寫。`queue rewind SELECTION_JOB_ID` 只回退已完成 selection，利用 queue 的嚴格遞增 order 邊界將其後續 materialized jobs 移到 `invalidated/`，在 `events.jsonl` 記錄原因與路徑，再重新排程；不會刪除既有結果。order 邊界也能修復舊 queue 中錯誤的 parent link。

新建立的 D1 queue 只有 `d1-shared`、`d1-pattn`、`d1-phead` 三個完整 10-epoch run，沒有 extension migration。repository 內既有 `d1-*-10`、`d1_extended` event 與相關 checkpoints 是 2026-08 已完成 staged 5+5 實驗的不可改寫歷史 provenance；結果報告應維持原 run 名稱，不能把它們宣稱為新版單次 10-epoch run。

`ArtifactStore` 在真正 launch training 前建立：

~~~text
artifacts/runs/<run_id>/
├── manifest.json
├── variant.yaml
├── training.yaml
├── checkpoints/
├── metrics/
├── profiles/
├── exports/
├── logs/
└── ultralytics/
~~~

`metrics/queue-result.json` 是 queue 的標準結果 contract。Fixed-scale `checkpoints/calibrated.pt` 必須保存未融合 Conv+BN training graph 與校準完成的兩處 Attention coefficients；fused checkpoint 只可作 inference/export，不可作下一個 training parent。`profiles/analytical.json` 保存固定 shape 假設下的 operation/memory proxy，`hardware_measurement: false`；它用於相同 evaluator 的 tie-break，不代表實際板端 latency 或 energy。

2026-08-16 最終清理刪除 invalidated tree、worker logs、所有 last、階段/non-winner/calibration checkpoints、Ultralytics 圖與空 leaf；保留 queue、所有正式數值 provenance，以及 V1-BR、N1-SHIFT、BDCN-V3-LEARN 三個必要 best checkpoints。既有 immutable `profiles/analytical.json` 是舊 schema-1 產物，Hadamard 總量少算第二個 binary basis；不覆寫原始 artifact，正式修正值與公式放在 `../reports/COMPUTE_AND_SIZE.md`／`compute_and_size.csv`。

manifest 記錄 variant、training recipe、Python、PyTorch、Ultralytics、platform 與 Git revision。run directory 使用 `exist_ok=False`，避免覆寫正式 V1。

runtime artifacts 預設由 `.gitignore` 排除；本次正式 metrics/profile 與三個必要 checkpoints 經使用者明確核准提交 Git。未來的 checkpoint、ONNX、engine 與 NumPy dump 仍不得在沒有明確授權時提交。
