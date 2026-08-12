# B1R / P2 / P3 Study

這是本 repo 的正式研究入口，採用 `yolo_p2/p2_study` 的分層方式整理第二輪
BBT5 實驗。訓練、驗證、profile 與報告已完成；原始 runtime 產物仍保留在
`artifacts/b1r-p2-p3-retest/`，本目錄提供穩定、易讀的 study 入口。

## 實驗順序

1. CPU contract、模型 forward/backward 與 transfer audit。
2. B1R-A freeze 0–10、10 epochs。
3. B1R-B full 90 epochs；若退化則 direct 100 epochs。
4. BBT5 head-only control 20 epochs + full 80 epochs。
5. P2 五種 MFAM 先 smoke 3 epochs，再 formal 100 epochs。
6. P3 五種 MFAM 先 smoke 3 epochs，再 formal 100 epochs。
7. 統一 val/test、AP_S/M/L、Ball/Bat、FP 與硬體成本。

## 目錄

```text
b1r_p2_p3_study/
├── README.md
├── config.yaml                 # 唯一實驗設定摘要
├── data/README.md              # 資料來源、split 與 hash
├── results/
│   ├── README.md
│   ├── REPORT.md               # 中文正式報告入口
│   ├── comparison.csv
│   ├── summary.json
│   ├── metrics/                # val/test 指標索引
│   ├── weights/                # formal checkpoint 索引
│   ├── profiles/               # 硬體成本索引
│   └── metadata/               # lineage、queue、audit
└── scripts/README.md           # 如何重建後處理
```

## 資料與限制

- dataset：`bbt5-detect-baseline/dataset`，固定 group split 80/10/10，hash 見 `data/README.md`。
- initializer：`bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`。
- B0 是資料暴露的 operational reference，不參與選模。
- smoke 只驗證工程穩定性；正式比較只使用 formal best checkpoint。
- P2/P3 的 placement 與 partial variants 是任務 adaptation，不宣稱完整重現論文 placement。

## 重建命令

```bash
../.venv/bin/python -m masf_yolo.retest.postprocess
../.venv/bin/python -m masf_yolo.retest.profile_all
../.venv/bin/python -m masf_yolo.retest.report
../.venv/bin/python -m masf_yolo.retest.audit
```
