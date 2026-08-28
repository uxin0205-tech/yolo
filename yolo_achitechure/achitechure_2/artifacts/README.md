# 產物目錄

本目錄保存可重建 metadata，以及使用者於 2026-08-22 明確核准的完整 GitHub dataset snapshot。
完整資料只能出現在指定子目錄；cache、checkpoint、weights、training runs 與 GPU profile 仍一律
留在版本控制之外。

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

metadata 層不含 images、labels 或 symlink。實際資料可用以下命令重建：

    export PYTHONPATH="$PWD/src"
    /home/uxin/yolo/.venv/bin/python -m achitechure_2 prepare-pose-data --execute

bbat5-v1 是不可覆寫版本。規則或來源改變時必須建立 bbat5-v2。
其 manifests 保留建立時 spec 2.0.0 的 hash；專案升到 2.3.0 後仍不回溯改寫。

## 固定20%架構篩選 View

`datasets/architecture-screen-20-v1/` 只保存 COCO／BBAT5 train manifests、互斥 search-val、
來源／輸出 SHA256、類別統計與中文 README；不含影像或 labels，也不改變 canonical bbat5-v1。
它供 C0–C3 Float20 與 eligible candidates 的 Q1／Q2L 使用，不能產生正式 C_best。
Ultralytics label index 只可寫在該目錄的 `runtime-cache/`，這些 `.cache` 可重建且不提交；不得在
`/home/uxin/yolo/coco2017/` 或 `/home/uxin/yolo/original/pose/derived/bbat5-v1/` 旁留下
source-adjacent cache。

## 已核准的完整 GitHub snapshot

唯一允許提交完整資料的位置是 `datasets/bbat5-v1/github-dataset/`，其中包含：

- 6,647 張唯一影像在 Pose／Detect 兩個可直接對應 labels 的一般檔案路徑。
- Pose 與 Detect labels；Detect 每列仍等於 Pose 前五欄。
- formal/search portable split lists 與各 view 的 `data.yaml`／`data-search.yaml`。
- 中文 README 與 `publication-manifest.json`。

此 snapshot 不含 symlink、絕對 split path、test、weight、checkpoint、cache 或 run。它只改變發布
儲存形式，不建立新 split，也不改寫 bbat5-v1 的 2.0.0 lineage。

## 執行時可能產生但不提交的目錄

| 路徑 | 內容 |
|---|---|
| intake/ | accepted handoff 與驗收報告 |
| cpu-validation/ | detect／pose／both、loss、gradient、freeze、reload 證據 |
| effective-configs/ | handoff recipe 與本地 runtime 值解析後的完整設定 |
| candidates/ | resolved candidate graph snapshot、transfer report、checkpoint lineage |
| assessment/ | C0 delta、描述性敏感度與 Pareto 報告 |
| runs/ | 正式訓練輸出、best／last、metrics 與 stop reason |
| quant/ | Q0／Q1／Q2L／Q2 simulation checkpoint、node coverage 與 regression |
| queue/ | GPU queue 的 lock、state 與 log；保留中斷與啟動證據 |

Full35 J3 handoff、C0～C3 CPU contract、Float20 Detect＋Pose、RTX 5090成本profile與正式匯出均已完成。
使用者於2026-08-28因精度下降過大決定不採用本階段；checkpoint、inference副本、候選log、runtime
cache與CPU驗收暫存已清除。正式負結果與hash manifest保留，C2/C3 full、PTQ、QAT-lite未執行。
`github-dataset/`是已核准的Portable GitHub Snapshot；它不是另一份正式資料版本，也不改變canonical
BBAT5 v1 assignment或lineage。

## 禁止放入本目錄

- `datasets/bbat5-v1/github-dataset/` 以外的原始或衍生影像／labels。
- pt、onnx、engine checkpoint。
- Ultralytics cache、runs 或第三方原始碼副本。
- 未通過 materialized graph 驗收的 accepted.json。

舊 artifacts/datasets/pose_grouped 已淘汰；正式名稱與位置只有 bbat5-v1。
