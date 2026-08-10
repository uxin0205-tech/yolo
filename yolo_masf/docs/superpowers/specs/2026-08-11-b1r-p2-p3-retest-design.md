# B1R 與 P2/P3 M0–M4 加測設計

日期：2026-08-11

## 目標

找出目前四尺度 B1 明顯落後三尺度 B0 的原因，建立盡可能保留 B0 能力的 revised P2 baseline（B1R），再以一致的 M0–M4 配方比較 MFAM 放在 P2 與 P3 的效果。正式結果必須報告 overall、Ball、Bat 的 mAP 與 COCO small/medium/large AP。

本設計不使用 knowledge distillation，不覆蓋 Phase 1 checkpoint，不以 test 指標決定模型或修改實驗。

## 已確認的 B1 問題

### 權重轉移不完整

現有 B0→B1 transfer manifest 顯示：

- 成功轉移 601 個 tensors。
- 154 個 tensors 沒有來源，包含新 P2 path、重建 neck 與 P2 detect branch。
- 48 個 tensors shape mismatch，而且全部在 `model.31.cv3.*` classification towers。

現有四尺度 `Detect` 會以第一個 P2 input channel 數決定共同 classification tower width，導致原本 B0 的 P3/P4/P5 classification tensors 也無法完整轉移。因此現有 B1 不是「B0 完整保留後只多一個 P2 head」。

### 全模型微調退化

- B1-A 凍結 backbone 0–10 訓練 10 epochs，validation mAP50–95 最佳值在 epoch 10，為 0.6989。
- B1-B 全解凍 90 epochs，validation mAP50–95 最佳值已出現在 epoch 1，為 0.7092；epoch 90 降至 0.5991。
- 正式 checkpoint 正確保存 `best.pt`，所以不是誤用 last checkpoint；曲線仍證明目前 `lr0=0.001` 的長時間全解凍沒有改善 B1。

### 問題型態

B1 test recall 與 B0 接近，但 Ball false positives 增加 395、Bat false positives 增加 568；overall AP_S/AP_M/AP_L 分別下降約 0.0280/0.0301/0.0560。主要問題是 inherited representation/head 被改動後的分類排序與校準退化，不是單純缺少小物件 recall。

## B1R 架構

B1R 以 B0 三尺度 graph 為母體，不重建 B0 的 P3/P4/P5 neck 與 detect towers：

1. B0 backbone、neck、P3/P4/P5 box towers 與 classification towers逐 tensor 完整轉移。
2. 從 backbone stride-4 feature 建立獨立 P2 adapter。
3. 新增 P2 box/class prediction towers；P2 同時預測 Ball 與 Bat。
4. forward 將 P2 raw predictions 與原 P3/P4/P5 raw predictions組成四尺度標準 detection loss input。
5. P2 classification bias 使用低 prior 初始化，使未訓練 P2 不搶占 B0 原有 predictions；不得以 test set 調整此 prior。

B1R 的 transfer contract：

- B0 的所有 state-dict tensors 都必須有明確 destination，shape mismatch 為 0。
- `unexpected`、`missing inherited`、`shape mismatch` 皆為 0；manifest 另外列出只屬於 P2 的新 tensors。
- 在 P2 branch 關閉的 synthetic forward 中，B1R 的 P3/P4/P5 raw outputs 必須與 B0 數值一致。

## B1R 訓練

### B1R-A：只學新增 P2

- 初始化：B0 `yolo11m_bat_detect_init.pt`。
- epochs：10。
- freeze：所有繼承自 B0 的 tensors；只有 P2 adapter、P2 box tower、P2 class tower可訓練。
- optimizer：SGD，沿用 batch 16、momentum 0.937、cosine schedule、seed 42、AMP 與既有 augmentation。
- `lr0`：0.001，不沿用現有 B1-A 的 0.01。

B1R-A 完成後先執行 validation gate。若 overall mAP50–95 與 B0 validation 相差超過 0.01，停止後續 90 epochs，先輸出 P2/P3–P5 branch 的 prediction counts、score distribution、per-class FP 與 AP_S/AP_M/AP_L，修正初始化或融合問題後才重跑。

### B1R-B：保守全模型微調

- 初始化：B1R-A validation best。
- epochs：90。
- freeze：無，全模型可訓練。
- `lr0`：0.0001，是舊 B1-B 的十分之一。
- checkpoint：每 epoch 依 validation fitness 保存 best；不得使用 test 選 epoch。
- B1R 最終 canonical：在 B1R-A best 與 B1R-B best 中，以 validation overall mAP50–95 為第一順位、AP_S 為同分 tie-break 選擇。

若 B1R-B 沒有超過 B1R-A，保留 B1R-A 為 B1R canonical，並把 full fine-tuning degradation 記入報告；不以延長 epochs 或讀取 test 反向改參數。

### B1R-D：條件式直接全模型訓練

若 B1R-B 低於 B1R-A，或 staged B1R 最佳 validation overall mAP50–95 仍比 B0 低超過 0.01，才啟動 B1R-D：

- 初始化：與 B1R-A 完全相同，使用同一份 B0 tensors、P2 adapter/head 初始化與 classification prior。
- epochs：100。
- freeze：無；從 epoch 1 起全模型直接訓練。
- `lr0`：0.001，搭配相同 SGD、cosine schedule、batch、seed 與 augmentation。
- 不從 B1R-A 或 B1R-B checkpoint 接續，避免把 staged training 的退化帶入直接訓練組。
- 這裡的「直接訓練」仍使用 B0 初始化，不是 random initialization。

B1R-A/B 與 B1R-D 的總預算都是 100 epochs；兩者只改變 freeze schedule 與全解凍 learning rate。B1R parent selection 只讀 validation，以 overall mAP50–95 為第一順位、AP_S 為 tie-break；test 在 parent 凍結後才執行。

## Identity-preserving MFAM

Phase 1 MFAM 的 random module 會立即取代父 feature，初始狀態不等於 identity。新 P2/P3 系列使用 residual-gated MFAM：

```text
output = input + gate × mfam_delta(input)
```

- `gate` 是每 channel 可學參數，初始化為 0。
- gate=0 時必須 bitwise 或在固定 tolerance 內等於 input。
- M0–M4 都使用相同 gate 形式，避免只讓其中一個變體獲得初始化優勢。
- Partial variants 的 bypass channels 永遠保持 exact identity；processed channels 使用相同零起始 gate。

## 統一顯示名稱

| 顯示名稱 | 內部意義 |
|---|---|
| `B0-Original-3Scale` | 原始 B0 三尺度 operational reference |
| `B1-Legacy-P2` | Phase 1 舊四尺度 B1，只作問題對照 |
| `P2-Base-Staged` | B1R-A 10 epochs 加 B1R-B 90 epochs 的分階段候選 |
| `P2-Base-Direct` | B1R-D 從 epoch 1 全模型直接訓練 100 epochs |
| `P2-Base-Selected` | validation 在 staged/direct 候選中凍結的 P2 parent |
| `P3-Base-Original` | 原始 B0 三尺度權重，作為 P3 family parent |

A/B/D 只用於 pipeline stage 名稱，不作報告主要實驗名稱。新 artifact directories 使用表中的 ASCII kebab-case 名稱轉成 lowercase snake_case。

## 新實驗矩陣

### P2 系列

共同父權重為 validation 選出的 B1R-A/B 或 B1R-D canonical，MFAM 只在 P2。新 artifacts 使用語意化 ID，M0–M4 只作為與舊討論對照的 alias：

| 新 ID | 舊 alias | processed channels | branches |
|---|---|---:|---|
| `P2-Full-35` | P2-M1 | 100% | DW3、DW5 |
| `P2-Partial50-35` | P2-M2 | 50% | DW3、DW5 |
| `P2-Partial25-35` | P2-M3 | 25% | DW3、DW5 |
| `P2-Full-35-F7` | P2-M4 | 100% | DW3、DW5、DW1×7→DW7×1 |
| `P2-Full-35-F7-F9` | P2-M0 | 100% | DW3、DW5、factorized DW7、factorized DW9 |

### P3 系列

共同父權重為原始 B0，維持三尺度 P3/P4/P5 detection，MFAM 只在 P3：

| 新 ID | 舊 alias | processed channels | branches |
|---|---|---:|---|
| `P3-Full-35` | P3-M1 | 100% | DW3、DW5 |
| `P3-Partial50-35` | P3-M2 | 50% | DW3、DW5 |
| `P3-Partial25-35` | P3-M3 | 25% | DW3、DW5 |
| `P3-Full-35-F7` | P3-M4 | 100% | DW3、DW5、DW1×7→DW7×1 |
| `P3-Full-35-F7-F9` | P3-M0 | 100% | DW3、DW5、factorized DW7、factorized DW9 |

`F7` 與 `F9` 分別表示 factorized `DW1×7→DW7×1` 與 `DW1×9→DW9×1`。舊 `M7`、`M0`～`M3` 與 `P3M` 名稱只保留在 Phase 1，不回寫或重新命名舊 artifacts。

## 語意化 variants 訓練

每個變體採兩階段：

- A stage：10 epochs，只訓練新 MFAM/gate，父模型 tensors freeze，`lr0=0.001`。
- B stage：90 epochs，全解凍，`lr0=0.0001`。
- 最終 checkpoint：A/B validation best 中依 overall mAP50–95、AP_S tie-break 選擇。
- 若 A stage 後比各自 parent overall mAP50–95 低超過 0.01，該變體停止 B stage並產生診斷，不浪費 90 epochs。

P2 與 P3 families 的 parent 不同，因此跨 family 結果是 operational comparison；MFAM 配方的嚴格消融必須先在各 family 內相對自己的 parent 判讀。

## 評估與報告

Validation 用於 gate、選 epoch 與排序；test 只在所有選擇凍結後執行一次。每個 parent 與 M0–M4 都輸出：

- overall：mAP50–95、mAP50、mAP75、AP_S、AP_M、AP_L。
- Ball/Bat：AP、AP50、AP75、AP_S、AP_M、AP_L、AR100。
- 固定 inference contract：precision、recall、prediction count、false positives、missed count。
- hardware：parameters、MACs、GFLOPs、mean/P95 latency、peak/P2/P3 activation。
- training：最佳 epoch、best/last gap、loss curves、trainable tensor count、initializer lineage與 hashes。

COCO AP_S/AP_M/AP_L 與 baseball-specific tiny/small/large recall 必須分表，不能用相同「小/中/大」名稱混為一談。

## Artifact 與相容性

- 新 artifact root：`artifacts/b1r-p2-p3-retest/`。
- Phase 1 的 `artifacts/static-phase1/`、checkpoint、selection 與 report 全部唯讀保留。
- 新 variant IDs 不加入舊 Phase 1 contracts；使用獨立 config、pipeline state、stage manifests與 final audit。
- pipeline 仍使用單一 GPU queue，排在既有 GPU 工作後，不啟動平行訓練。

## GPU 執行順序

1. CPU contracts、unit tests、strict reload、synthetic forward parity、loss/backward/step preflight。
2. B1R-A 10 epochs與 validation gate。
3. gate 通過才執行 `P2-Base-Staged` 的 B1R-B 90 epochs。
4. 若 staged B1R 觸發 fallback 條件，從同一初始化執行 `P2-Base-Direct` 100 epochs。
5. 只用 validation 凍結 P2 base selection。
6. 依序執行 `P2-Full-35`、`P2-Partial50-35`、`P2-Partial25-35`、`P2-Full-35-F7`、`P2-Full-35-F7-F9`；每組先 A-stage，通過才執行 B-stage。
7. 以相同順序執行五個 P3 variants。
8. 所有 validation selection 凍結後，統一執行 test 與同硬體 profile。
9. final audit、中文詳細報告與 GitHub push。

## 完成條件

- B1R 對所有 B0 tensors 的 transfer coverage 為 100%，shape mismatch 為 0。
- CPU preflight 不使用 GPU 且所有測試通過。
- GPU pipeline 可安全續跑、gate 可阻止無效 90-epoch stage、舊 artifacts 不被修改。
- 報告能區分 B0 高分的已知原因、B1R 修正效果、P2/P3 family 內消融，以及 data exposure/單 seed 限制。
- 不宣稱 B1R 必然追平 B0；以 validation/test 證據決定是否達成。
