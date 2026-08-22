# 產物目錄

本目錄只把「可重建與可稽核的 metadata」納入 Git。影像、衍生 labels、cache、checkpoint、
training runs 與 GPU profile 一律留在版本控制之外。

## 已提交的資料 metadata

artifacts/datasets/bbat5-v1 是
/home/uxin/yolo/original/pose/derived/bbat5-v1 的唯讀副本索引，只包含：

- 中文資料說明。
- formal Pose／Detect 與 search Pose／Detect dataset YAML。
- split manifest。
- patch manifest。
- source audit manifest。
- COCO exclusion manifest。
- rebuild manifest。

它不含 images、labels 或 symlink。實際資料可用以下命令重建：

    export PYTHONPATH="$PWD/src"
    /home/uxin/yolo/.venv/bin/python -m achitechure_2 prepare-pose-data --execute

bbat5-v1 是不可覆寫版本。規則或來源改變時必須建立 bbat5-v2。
其 manifests 保留建立時 spec 2.0.0 的 hash；專案升到 2.0.1 後不回溯改寫。

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

- 原始或衍生影像。
- 原始或衍生 label 檔。
- pt、onnx、engine checkpoint。
- Ultralytics cache、runs 或第三方原始碼副本。
- 未通過 materialized graph 驗收的 accepted.json。

舊 artifacts/datasets/pose_grouped 已淘汰；正式名稱與位置只有 bbat5-v1。
