# 實體目錄與新舊路徑

## 現在的分類

```text
yolo_optimize/
├── README.md
├── experiments/             全部研究程式與產物
│   ├── studies/             融合前方向 1
│   ├── combine/             Pose 適應、融合及 BBAT 恢復
│   ├── activation/          activation 配對
│   ├── kd/                  雙教師及 Pose-head KD
│   ├── inference/           推論分支研究
│   ├── artifacts/           早期融合後原始產物
│   ├── scripts/             早期融合後研究工具
│   ├── src/                 共用模組
│   └── tests/               CPU 契約測試
├── reports/
│   ├── final/               全階段詳細報告與數值表
│   ├── direction1/          融合前專題報告
│   ├── checkpoints/         權重導航，不另複製權重
│   └── publication/         發布入口與驗證紀錄
├── proposals/               原始優化提案
├── docs/
│   ├── research/            論文解讀與推導
│   ├── references/          使用者參考 PDF
│   ├── worklogs/            中文操作紀錄
│   └── history/             修改前原文與遷移證據
├── tools/                   封存、整理與發布工具
└── archives/                固定封存副本
```

這次是真正搬移資料夾，包括 checkpoint，不是只改 README，也不保留根層舊目錄 symlink。各階段內部關係保留，以減少程式相依風險。

## 舊新路徑對照

| 舊路徑 | 新路徑 |
| --- | --- |
| studies、combine、activation、kd、inference | experiments 下的同名目錄 |
| artifacts、src、tests、研究 scripts | experiments 下的同名目錄 |
| scripts/maintenance | tools |
| optimizations | proposals |
| reports/consolidated-20260911 | reports/final |
| reports/direction1-20260910 | reports/direction1 |
| reports/5090-done-0912 | reports/publication |

精確的 14 筆資料夾搬移、檔案路徑對應與修改前快照見[manifest](<history/layout-v2-20260912/manifest.json>)。JSON／CSV 中既有數值不变，只更新必要的路徑文字；修改前原文仍可追溯。checkpoint 內嵌的歷史路徑不改寫，權重 bytes 不變。

## 未來存放規則

新實驗放 experiments 的對應階段，採新的 run ID；跨階段結論更新 reports/final，新假說另放 proposals。文獻放 docs/research，原 PDF 放 docs/references，過程放 docs/worklogs。不要再把進度堆在首頁，也不要另建一套「final-final」。

封存包不修改；同名歷史 run 不覆寫。新的訓練必須使用新輸出目錄及 canonical 資料，不把遷移視為重新執行 queue 的理由。[操作與恢復邊界](<OPERATIONS.md>)說明何者已驗證、何者仍需重新驗收。
