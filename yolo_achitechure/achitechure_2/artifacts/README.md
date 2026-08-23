# 產物目錄

本目錄只保存執行時產物，不擁有資料集、正式 YAML、split、labels 或 lineage。

BBAT5 的唯一資料資產庫是 `/home/uxin/yolo/original/pose/`；正式入口是
`/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。2026-08-22 曾存在於 `artifacts/datasets/`
的可攜快照已由 2026-08-23 統一決策取代，僅保留在 Git 歷史。

本目錄中的內容原則上可由 handoff、設定或 run 重建，且預設不提交。

## 執行時可能產生但不提交的目錄

| 路徑 | 內容 |
|---|---|
| intake/ | accepted handoff 與驗收報告 |
| cpu-validation/ | detect／pose／both、loss、gradient、freeze、reload 證據 |
| effective-configs/ | handoff recipe 與本地 runtime 值解析後的完整設定 |
| candidates/ | resolved candidate graph snapshot、transfer report、checkpoint lineage |
| assessment/ | C0 delta、描述性敏感度與 Pareto 報告 |
| runs/ | 正式訓練輸出、best／last、metrics 與 stop reason |
| quant/ | Q0／Q1／Q2 simulation checkpoint、node coverage 與 regression |

目前沒有正式 handoff、Float 長訓練、Pose 長訓練、C_best 或量化精度成果。任何空目錄或 CPU
plumbing 都不能當成正式實驗結果。

## 禁止放入本目錄

- 任一資料集副本、images、labels、dataset YAML、split 或 dataset manifests。
- `.pt`、`.pth`、`.onnx`、engine checkpoint 或未獲准的權重。
- Git 要追蹤的 Ultralytics cache、runs 或第三方原始碼副本。
- 未通過 materialized graph 驗收的 accepted.json。
