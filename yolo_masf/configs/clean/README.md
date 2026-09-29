# Clean Initializer 完整公平實驗

這是唯一有效的正式設定入口。官方 COCO80 `yolo11m.pt` 已與 Ultralytics v8.3.0 release 比對，SHA-256 為 `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`。

## 資料可見性

| Split | 用途 |
|---|---|
| train | 訓練與更新權重 |
| validation | checkpoint、選模、calibration |
| historical test | selection freeze 後報告；不進 trainer YAML |
| final holdout | 尚未建立；unseen claim fail closed |

## 矩陣

- Strict fair：B0、P2 baseline、P2 五種、P3 五種，共 12 模型 × seeds 42/43。
- P3 禁止 9×9；P3-M7 單 fusion 與 P3-Lite35-F7 雙 fusion 分開。
- Optimization control：P2 head20 + full80，共 2 seeds。
- Selective extension：SP2/SP2P 僅在 validation freeze 後執行。
- 完整規格見 [`../../codex_plan.md`](../../codex_plan.md)。

## 證據與完成狀態

- 設定：[`clean_ablation.yaml`](clean_ablation.yaml)
- Feasibility：[`FEASIBILITY.json`](FEASIBILITY.json)
- GPU jobs：40/40 完成；28 個 formal/control checkpoints 完整。
- 統一後處理：validation 28/28、historical test 28/28、hardware profiles 14/14。
- 報告：[`../../clean_bbt5_study/results/REPORT.md`](../../clean_bbt5_study/results/REPORT.md)。

CPU inspect 驗證 initializer/environment/dataset contract、每個 unique architecture 的 stride、參數與逐 tensor transfer disposition。12 個 smoke jobs 只作工程 gate，不列入正式結果；selection 只讀 validation，historical test 在 freeze 後才執行。
