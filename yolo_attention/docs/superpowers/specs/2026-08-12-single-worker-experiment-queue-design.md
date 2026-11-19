# YOLO26m 單工作自動實驗 Queue 設計

> 日期：2026-08-12
> 狀態：accepted
> 範圍：CPU-safe queue 建立、狀態持久化、自動 winner gate、未來單一 GPU worker。
> 本規格不授權現在啟動 GPU、訓練、下載資料或進入 Optional Phase Q。

## 1. 目標

把 `plan.md` 的正式漏斗變成可恢復、可檢查、一次只執行一個 job 的自動 queue：

~~~text
建立 queue（CPU only）
       ↓
檢查資料、權重、設定與 parent checkpoint
       ↓
選出唯一 ready job
       ↓
未來明確加 --execute 後，單 worker 執行
       ↓
保存 checkpoint / metrics / status
       ↓
依 plan.md gate 自動選 winner
       ↓
動態展開下一階段 jobs
~~~

本次實作只準備 queue 建立、驗證、預覽、未來執行 seam 與測試；GPU 被其他人使用時，不執行任何 job。

## 2. 非目標

- 不做多 GPU、平行 job 或 scheduler service。
- 不自動下載 COCO2017 或 `yolo26m.pt`。
- 不以 CPU smoke 取代 COCO mAP。
- 不猜測缺失的 metrics 或 checkpoint。
- 不自動進入量化、ternary 或 SD4。
- 不改寫 Ultralytics 內部 trainer；只使用既有 `TrainingRequest`／`HardwareAttentionTrainer` seam。

## 3. Queue 狀態與檔案

預設根目錄：

~~~text
artifacts/queue/
├── queue.json              # queue schema、workflow、job 狀態及選擇紀錄
├── events.jsonl            # append-only 狀態事件
└── generated/              # 從 winner parent 產生的 immutable variant/recipe
~~~

實際 run 仍寫入既有：

~~~text
artifacts/runs/<run_id>/
├── manifest.json
├── variant.yaml
├── training.yaml
├── checkpoints/
├── metrics/
├── profiles/
├── exports/
└── logs/
~~~

`queue.json` 以原子替換寫入；已存在的 queue 不可由 `queue init` 靜默覆寫。所有 job 以穩定 ID 串接，路徑保存為相對 project root 或明確 absolute path。

## 4. Job schema

每個 job 至少包含：

| 欄位 | 意義 |
|---|---|
| `id` | 唯一小寫 job ID |
| `run_name` | B26-FP、P0、I-SCR 等研究名稱 |
| `stage` | workflow 階段 |
| `kind` | validate、train、evaluate 或 select |
| `variant_path` | variant YAML；selection job 可為 null |
| `training_path` | recipe YAML；zero-train/select 可為 null |
| `evaluation_path` | evaluation recipe；train/select/P0 可為 null |
| `parent_job_ids` | 所有前置 job |
| `parent_checkpoint` | 真正提供權重的成功 parent；未決定時為 null |
| `status` | blocked、ready、queued、running、succeeded、failed、interrupted、skipped |
| `requires_gpu` | 是否會使用 GPU |
| `attempts` | 執行次數 |
| `metrics_path` | 標準 COCO metrics JSON |
| `checkpoint_path` | best checkpoint |
| `failure_reason` | 失敗原因；正常時為 null |
| `decision` | winner、淘汰與 gate 依據 |

狀態轉換：

~~~text
blocked → ready → queued → running → succeeded
                              ├→ failed
                              └→ interrupted

blocked/ready → skipped       # gate 淘汰或未選中的條件分支
failed/interrupted → queued   # 使用明確 retry 命令
~~~

`ready` 表示依賴已滿足但尚未排入 worker；`queued` 表示已由 scheduler 選為下一個 job。`queue next` 只預覽，不改狀態；`run-next --execute` 才在取得 lock 後把最早的 ready job原子轉為 queued，再立刻轉為 running。同一份 queue 最多只能有一個 `queued` 與一個 `running`，且兩者不能同時存在。非法轉換、缺 parent、重複 ID 或多 worker 狀態都 fail closed。

## 5. 動態 workflow

### 5.1 初始 queue

`queue init` 只 materialize 可以在尚無實驗結果時明確定義的工作：

~~~text
B26-FP evaluation（初始唯一 ready）
      ↓
P0 equivalence（blocked）
      ↓
I-SCR（blocked）
      ↓
H-SCR（blocked）
      ↓
T5-SCR（blocked）
      ↓
architecture-select（blocked）
~~~

I/H/T5 都從同一 P0 checkpoint 分支，但 queue dependency 額外串成 `I-SCR → H-SCR → T5-SCR`，只為保證單 worker 固定順序；H/T5 的 model parent 仍是 P0，不可錯接前一個 screening checkpoint。

### 5.2 Winner 後動態展開

`architecture-select` 成功後，由 winner checkpoint 產生：

~~~text
W-DIR
W-PROG
recovery-select
~~~

`recovery-select` 形成 formal V1，接著依 `plan.md` 展開 scale、bias 與 A0 selection。A0 完成後建立兩個邏輯分支，但仍序列執行：

~~~text
Normalization: N0 → 最多兩個 N1 → normalization-select
BDCN: D0 → D1 → D2 → R0/R1/R2 → bdcn-select
                                  ↓
                              final-select
                                  ↓
                               A-FINAL
~~~

未選中的候選標為 `skipped`，不刪除紀錄。只有 gate 通過且依賴完整時才產生 recovery job。

## 6. 自動選擇規則

所有比較使用同一 evaluator 的 COCO `map50_95`，缺值即 selection 失敗，不能自動猜 winner。

| 選擇 | 規則 |
|---|---|
| I/H/T5 | 最高 mAP；與最佳差小於 0.001 時依 plan 的較低額外成本順序選擇 |
| W-DIR/W-PROG | 最高 mAP；差小於 0.001 時選 W-DIR |
| Scale | 依 parent-relative 0.001 gate；只在需要時展開 3-epoch recovery |
| Bias | 先與相同訓練時間 B0 比較；Dense/Decomposed 差小於 0.001 時選 Decomposed |
| N0 | 相對 A0 loss 不超過 0.01；依 accuracy/cost 最多保留兩個 |
| N1 | 與 A0 exact 比較 accuracy/cost，選 normalization winner |
| D1 | 最高 mAP；與最佳差小於 0.001 時選 global → per-attention → per-head 中最簡單者 |
| D2 | 相對 A0 loss 不超過 0.01；最多一個候選補 5 epochs |
| R0/R1/R2 | R1 是 primary；R1 相對 R0 mAP loss 超過 0.002 或 row-sum max error 超過 0.01 才展開 Newton job |
| A-FINAL | 先保留與最高 mAP 差小於 0.001 的候選，再選 estimated memory traffic 最低者；仍相同時選 arithmetic proxy 最低者 |

成本 tie-break 必須讀取 `profiles/*.json` 的 `estimated_memory_traffic` 與 `arithmetic_cost_proxy`。若所需成本資料不存在，selection job 進入 failed 並指出缺少欄位，不用寫死偏好冒充測量。

## 7. CLI

~~~bash
# 建立 queue；只寫狀態，不訓練
python -m yolo_attention.cli queue init

# 人類可讀或 JSON 狀態
python -m yolo_attention.cli queue status [--json]

# CPU-only 檢查 schema、依賴、YAML、COCO、權重及現有 checkpoint/metrics
python -m yolo_attention.cli queue validate [--json]

# 顯示唯一下一個 job 及完整命令，不執行
python -m yolo_attention.cli queue next [--json]

# 未來 GPU 空閒後，一次執行一個
python -m yolo_attention.cli queue run-next --execute

# 未來持續序列執行，仍只有一個 running job
python -m yolo_attention.cli queue run --execute

# 明確重排失敗或中斷 job
python -m yolo_attention.cli queue retry JOB_ID
~~~

`run-next`／`run` 沒有 `--execute` 時只 dry-run。所有 GPU job 還需確認 recipe 的 `device` 不是 CPU，並取得 queue lock；lock 已被持有時立即停止，不等待第二個 worker。

## 8. 執行與恢復

worker 執行順序：

1. 取得 queue lock。
2. 重讀 queue 並驗證沒有其他 running job。
3. 將唯一 job 原子轉為 running，記錄 event。
4. 呼叫既有 runner。
5. 驗證標準 metrics 與 checkpoint 是否存在。
6. 成功則標為 succeeded；例外則標為 failed 並保存原因。
7. 執行可決定的 selection job並動態展開下一階段。
8. 釋放 lock。

程序收到中斷時，把 running job 改為 interrupted。重新啟動不會重跑 succeeded job；failed/interrupted 必須使用 `retry` 明確回到 queued。

## 9. CPU-safe validation

`queue validate` 報告分成 error 與 warning：

- error：schema 錯誤、循環依賴、兩個 running、缺 variant/recipe、YAML 無法解析、應存在的 parent artifact 缺失。
- warning：GPU 忙碌狀態無法由本地可靠判斷、未來階段尚未 materialize、optional quantization locked。

COCO 路徑與 `yolo26m.pt` 只檢查存在性／可讀性，不載入完整資料、不啟動 CUDA。`queue init/status/validate/next/retry` 不得 import 或呼叫會開始訓練的 side effect。

## 10. 測試

採 TDD，至少涵蓋：

- 初始 job、依賴與固定序列。
- init 不覆寫既有 queue。
- 原子 round-trip 與非法 schema。
- 最多一個 running job。
- 沒有 `--execute` 時 runner 絕不被呼叫。
- 缺 checkpoint／metrics 時 fail closed。
- I/H/T5、Direct/Progressive、D1、N0、R1 gate 的 tie-break 與動態展開。
- retry 只接受 failed/interrupted。
- succeeded job 不重跑。
- optional quantization 不會被自動加入。
- CLI init/status/validate/next JSON contract。

完整 suite、Ruff 與 CPU CLI smoke 通過後，才可宣稱 queue 架構完成。GPU training 與 COCO mAP 仍標記 `not_run`。

## 11. 文件同步

實作時同步更新：

- 根 `README.md`：增加 queue 快速使用。
- `AGENTS.md`：單 worker、`--execute` gate、不可自動量化。
- `docs/architecture.md`：queue 與 runner seam。
- `src/README.md`：新增 queue module 責任。
- `configs/README.md`：動態 winner-derived YAML 規則。
- `artifacts/README.md`：queue state、run artifacts 與提交政策。
- `tests/README.md`：queue contract。
