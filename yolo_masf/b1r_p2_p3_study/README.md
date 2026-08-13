# B1R / P2 / P3 第二輪實驗

這是第二輪實驗的 GitHub 穩定入口。報告、metrics、profiles 與執行 metadata 都是實體檔案，不再連到被 `.gitignore` 排除的本機 runtime symlink。

## 閱讀順序

1. [完整實驗流程](EXPERIMENT_PROCESS.md)：資料、初始化、baseline、MFAM、GPU queue 與後處理。
2. [完整結果報告](results/REPORT.md)：val/test 全表、硬體成本與分析。
3. [逐模型索引](results/experiments/README.md)：每個模型自己的做法、metrics、profile 與 lineage。
4. [權重狀態](results/weights/CHECKPOINT_STATUS.md)：哪些 `.pt` 可下載、哪些只剩 SHA-256。
5. [第一輪歷史摘要](results/LEGACY_RESULTS.md)：B0/B1、M0–M7、P3M、SP2/SP2P。

## 目錄

```text
b1r_p2_p3_study/
├── README.md
├── EXPERIMENT_PROCESS.md
├── config.yaml
├── data/README.md
├── scripts/README.md
└── results/
    ├── README.md
    ├── REPORT.md
    ├── comparison.csv
    ├── summary.json
    ├── PUBLICATION_MANIFEST.json
    ├── experiments/<model>/
    │   ├── README.md
    │   ├── val_metrics.json
    │   ├── test_metrics.json
    │   ├── profile.json
    │   └── checkpoint.json
    ├── profiles/
    ├── metadata/{requests,worker}/
    └── weights/CHECKPOINT_STATUS.md
```

## 實驗矩陣

- Baselines：B0 Original 3Scale、P2 Base Direct、P2 Control Head、P2 Control Full。
- P2：PaperFormula-Full、Lite-35、Lite-35-F7、Partial50-35、Partial25-35。
- P3：相同五種變體，只在 P3 placement，不增加完整 P2 Detect head。
- Formal 設定：640、batch 16、SGD、cosine LR、AMP、seed 42；各 MFAM formal 100 epochs。

Dataset 不在 GitHub 發布包。來源 initializer 已接觸 BBT5，所以此 study 只支持同源操作性比較。
