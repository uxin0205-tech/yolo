# 結果目錄

目前沒有 yolo_combine 正式 winner handoff，也沒有 C0–C3 Float 長訓練結果。因此本目錄只有
空白範本，不能把 CPU dry-run、資料分割或 skipped integration test 當成精度成果。

## 兩份範本

- results-template.csv：每個 handoff reference、control 與 resolved candidate 一列，保存任務指標、
  成本、執行狀態與完整 lineage。
- report-template.md：正式報告骨架，明確保留 Pose 是否執行、C_best 使用者決策與量化資格。

shared winner 可先使用 C0-Handoff、C0、C1、C2、C3 五列。routed 或 partial-shared winner 必須依
resolve-candidates 的實際輸出新增 D-、P- 或 S- 列，不能把不同 region 合併成一筆。

## 正式指標

| 範圍 | 必填欄位 |
|---|---|
| COCO Detect | overall mAP50、mAP50-95；person AP50、AP50-95 |
| BBAT5 Pose box | mAP50、mAP50-95 |
| BBAT5 keypoint | mAP50、mAP50-95 |
| Pose checkpoint | pose-only best 選擇方式；官方 Pose+Box combined fitness 另列 |
| ball／bat | box AP50、AP50-95；keypoint AP50、AP50-95；Precision、Recall、F1 |
| 整體 F1 | Macro F1、Micro F1、固定 confidence threshold |
| 成本 | Params、GFLOPs、latency、peak VRAM 與量測裝置 |

AP50 只看 IoU／OKS 0.50；AP50-95 平均 0.50 到 0.95。mAP 是 class 平均，所以結果名稱必須
說清楚 mAP50 或 mAP50-95。Macro F1 是 ball／bat F1 的未加權平均；Micro F1 先彙總所有
TP／FP／FN。

F1 threshold 只用 C0-Control 的 search-val 決定一次，之後所有候選與 formal val 固定相同值。
若 Pose 未獲 opt-in，所有 Pose／ball／bat／F1 欄位保持空白，formal_comparison_ready 必須是
false，且不能呼叫完整 assess 產生 C_best 結論。

Stock Pose best.pt 依 combined fitness，而非 pose-only mAP50-95。若執行 Pose，CSV 的
pose_checkpoint_selection 必須標明研究 checkpoint 的保存方式，且
pose_official_combined_fitness 必須分欄記錄。

## 描述性比較

完成所有必要指標後，將實測 JSON 傳給：

    python -m achitechure_2 assess --input /path/float-metrics.json --output artifacts/assessment/float.json

JSON 每個 candidate 的欄位對應 achitechure_2.decisions.CandidateMetrics。此命令只計算：

- 相對 C0 的各指標差值。
- Params／GFLOPs／latency／VRAM 降幅。
- 舊 0.005／0.008 drop 的描述性 band。
- Pareto front。

它不會自動淘汰候選，也不會自動選 C_best。輸出固定為 c_best=null 與
selection_status=pending_user_decision，等使用者看完完整 Float 結果後決定。

## Lineage

每列至少保存 spec、architecture YAML、training YAML、Detect dataset YAML、Pose dataset YAML、
parent checkpoint、result checkpoint 的 SHA256，以及 handoff revision、winner、git revision、seed
與 resolved effective config。舊結果永遠保留當時 hash，規格升版後不可回填新 hash。
