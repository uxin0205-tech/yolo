# 結果目錄

Full35 J3 handoff與C0～C3 Float20 Detect＋Pose已完成，正式成果位於
[full35-j3-float20-seed0](full35-j3-float20-seed0/REPORT.md)。

- 狀態：`completed_screening`，四候選各20 epochs，正式RTX 5090成本profile已完成。
- C0精度最高；C2是最佳簡化候選，但相對C0的四項主要精度下降都超過0.008。
- 原始報告固定`c_best=null`與`pending_user_decision`。
- 使用者已決定不採用本階段；C2/C3 full、PTQ、QAT-lite永久關閉，不存在可宣稱的下游結果。

CPU dry-run、真實batch loss smoke與資料驗證仍不得冒充accuracy／GPU成本成果。

## 兩份範本

- results-template.csv：每個 handoff reference、control 與 resolved candidate 一列，保存任務指標、
  成本、執行狀態與完整 lineage。
- report-template.md：正式報告骨架，明確保留 Pose 是否執行、C_best 使用者決策與量化資格。

本輪不是手工抄值；下列命令只用於依既有完整matrix與profile重建正式匯出：

    python -m achitechure_2 export-float20-results --execute

工具已建立 `full35-j3-float20-seed0/`，內含metrics JSON／CSV、selection、lineage、中文報告、
比較圖及hash manifest；缺任何候選、Pose指標、固定F1 threshold或lineage hash時都拒絕輸出。

Float20原始報告維持`c_best=null`。後續設定是`closed_rejected_by_user`；以下原定成果不會在本
revision建立：

- `full35-j3-c2-c3-quant-seed0/`：Q0／Q1／Q2L逐stage正式指標與gap。
- `full35-j3-c2-c3-final-seed0/`：完整Float選擇、量化相容stage、中文報告與hash manifest。

本次C2/C3皆未通過且已封存；沒有建立上述下游成果，也不會以缺資料或CPU smoke補成accuracy結果。

本輪固定使用Full35 shared winner與 C0-Handoff、C0、C1、C2、C3五列；修改路徑是
`graph.model.6/8/13/19`。不得在同一結果檔混入routed／partial-shared或另一個handoff revision。

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
說清楚 mAP50 或 mAP50-95。Macro F1 是ball／bat F1的未加權平均。Micro F1由官方P/R curves與
class supports在固定threshold下估算合併TP／FP／FN，必須另記
`estimated_from_precision_recall_curves_and_supports`，不可稱為逐預測exact重新配對。

Float20的F1 threshold只用C0-Control screen search-val決定一次，C1～C3在同一screen search-val
固定相同值；本輪不使用formal val。
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

這個Float20通用assess命令不會自動淘汰候選，也不會自動選C_best；輸出固定為c_best=null與
selection_status=pending_user_decision。本輪run-specific接續與最終選擇由獨立正式YAML處理，不回寫
這份原始初篩報告。

## Lineage

每列至少保存 spec、architecture YAML、training YAML、Detect dataset YAML、Pose dataset YAML、
parent checkpoint、result checkpoint 的 SHA256，以及 handoff revision、winner、git revision、seed
與 resolved effective config。舊結果永遠保留當時 hash，規格升版後不可回填新 hash。
