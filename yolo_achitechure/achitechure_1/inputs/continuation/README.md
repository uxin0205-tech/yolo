# Accepted A2 續跑來源

這裡只放舊工作站匯入的 accepted Phase A2 Float checkpoint 與小型來源描述，不放 prepared checkpoint，
也不要放已跑完的 Full35 Phase B checkpoint。

使用 `scripts/import_fraction03_source.py` 後應形成：

```text
inputs/continuation/
├── full35-accepted-a2/
│   ├── candidate.json
│   └── float-best.pt
└── partial75-accepted-a2/
    ├── candidate.json
    └── float-best.pt
```

`.pt` 已由 `.gitignore` 排除。Queue 會驗證 `candidate.json` 的 variant、boundary 與 SHA256，再用獨立
子程序確認 checkpoint 架構及 Float-PWL backend；通過後才會重新產生本機 Bit-True checkpoint 與完整
COCO validation metrics。同一對來源也供 `fraction=1.0` Phase B control 使用，確保 fraction 比較不更換
起始權重。
