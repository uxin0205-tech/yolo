# B1R 與 P2/P3 Paper-Formula MFAM 加測設計

日期：2026-08-11

## 目標

找出目前四尺度 B1 明顯落後三尺度 B0 的原因，建立權重轉移較完整的 revised P2 baseline（B1R），再以一致的論文公式 skeleton 比較 MFAM 放在 P2 與 P3 的效果。正式結果必須報告 overall、Ball、Bat 的 mAP 與 COCO small/medium/large AP。

本設計不使用 knowledge distillation，不覆蓋 Phase 1 checkpoint，不以 test 指標決定模型或修改實驗。論文忠實度依 [`docs/research/2026-08-11-masf-paper-implementation-audit.md`](../../research/2026-08-11-masf-paper-implementation-audit.md) 判定。

## 已確認的 B1 問題

### 權重轉移不完整

現有 B0→B1 transfer manifest 顯示：

- 成功轉移 601 個 tensors。
- 154 個 destination tensors 沒有來源，包含新 P2 path、重建 neck 與 P2 detect branch。
- 48 個 destination tensors shape mismatch，而且全部在 `model.31.cv3.*` classification towers。
- B0 source tensors 沒有 unexpected keys，但上述 48 個來源因 shape 不同而未真正載入。

現有四尺度 `Detect` 以第一個 P2 input channel 數決定共同 classification tower width，連原本 B0 的 P3/P4/P5 classification tensors 也無法完整轉移。這是 B1R 必須先修正的具體缺陷。

### 全模型微調退化

- B1-A 凍結 backbone 0–10 訓練 10 epochs，validation mAP50–95 最佳值在 epoch 10，為 0.6989。
- B1-B 全解凍 90 epochs，最佳值出現在 epoch 1，為 0.7092；epoch 90 降至 0.5991。
- 正式 checkpoint 正確保存 `best.pt`，所以不是誤用 last checkpoint；曲線仍顯示目前 B1 長時間 fine-tuning 沒有改善。

### 問題型態

B1 test recall 與 B0 接近，但 Ball false positives 增加 395、Bat false positives 增加 568；overall AP_S/AP_M/AP_L 分別下降約 0.0280/0.0301/0.0560。主要問題是新 P2/neck 與 classification head 的權重承接、排序及校準，不是單純缺少小物件 recall。

## B1R 架構與轉移

B1R 保留 repository-owned 標準四尺度 P2 template：

1. 維持 P2 top-down fusion、P2→P3 bottom-up path，以及 P2/P3/P4/P5 四尺度 Detect。
2. Detect 改為 per-scale classification tower width；P2 branch可使用自己的寬度，P3/P4/P5 branch 保持 B0 的原始 shapes。
3. B0 backbone、可對應 neck、P3/P4/P5 box/class towers 與 DFL tensors 全部明確轉移。
4. 新 P2 path、因四尺度 neck 才新增的 layers 與 P2 detect branch 保持新建，manifest 必須逐 tensor 列出。
5. P2 classification bias 使用固定 low prior 初始化；不得以 test set 調整。

B1R transfer contract：

- 所有 B0 source tensors 必須被消費並成功載入，source coverage 100%。
- `unexpected source` 與 `shape mismatch` 都必須為 0。
- destination missing 只允許 contract 明列的新 P2/neck tensors；不得混入 P3/P4/P5 classification tensors。
- CPU 測試必須證明 P3/P4/P5 detect tower tensors 與 B0 完全相同。

## B1R 訓練

### `P2-Base-Staged`

B1R-A：

- 初始化：B0 `yolo11m_bat_detect_init.pt` 加 B1R 新 layers。
- epochs：10。
- freeze：實際 backbone indices 0–10；neck、P2 path 與 Detect 全部可訓練。
- SGD、`lr0=0.01`、momentum 0.937、cosine、batch 16、seed 42、AMP 與既有 augmentation。

B1R-B：

- 初始化：B1R-A validation best。
- epochs：90。
- freeze：無。
- SGD、`lr0=0.001`，其餘設定與 B1R-A 相同。
- staged candidate 在 A/B validation best 中以 overall mAP50–95、AP_S tie-break 選擇。

若 B1R-B 沒有超過 B1R-A，報告 full fine-tuning degradation，並觸發 direct diagnostic。

### `P2-Base-Direct`

若 staged candidate 比 B0 validation overall mAP50–95 低超過 0.01，或 B1R-B 沒有超過 B1R-A，執行 direct candidate：

- 初始化與 B1R-A 完全相同，不從 A/B checkpoint 接續。
- 從 epoch 1 全模型不凍結，訓練 100 epochs。
- SGD、`lr0=0.001`、momentum 0.937、cosine、batch 16、seed 42、AMP 與相同 augmentation。
- 這裡的 direct training 仍使用 B0 初始化，不是 random initialization。

`P2-Base-Selected` 只用 validation 在 staged/direct candidates 中選擇；test 在 parent 凍結後才執行。

## 論文公式與 adaptation 邊界

MASF-YOLO 論文式 (1)–(6) 定義：

```text
Y1 = DW3×3(X)
Y2 = DW5×5(X)
Y3 = DW7×1(DW1×7(X))
Y4 = DW9×1(DW1×9(X))
Z  = X ⊕ Y1 ⊕ Y2 ⊕ Y3 ⊕ Y4
W  = Conv1×1(Conv1×1(Z) ⊕ X)
```

新實作以此公式為準，不加入 learnable branch weights、scale 或 zero gate。論文 Figure 2 只畫一個 1×1，與式 (6) 不一致；Phase 1 現有 `MFAM` 屬於 Figure-2/legacy 實作，新一輪 Full anchor 屬於 `PaperFormulaMFAM`。

忠實度必須分層描述：

- `PaperFormula-Full`：branch set 與尾端公式符合公開式 (1)–(6)。
- P2-only/P3-only placement：使用者指定的 BBT5 硬體友善 placement adaptation；不是論文 backbone 四處 placement。
- `Lite-35`、`Lite-35-F7`、`Partial50-35`、`Partial25-35`：branch/channel adaptations。
- B0 initializer、batch 16、per-class/size AP 與 repo training schedule：任務 adaptation，不是 VisDrone 完整重現。

## 統一顯示名稱

| 顯示名稱 | 內部意義 |
|---|---|
| `B0-Original-3Scale` | 原始 B0 三尺度 operational reference |
| `B1-Legacy-P2` | Phase 1 舊四尺度 B1，只作問題對照 |
| `P2-Base-Staged` | 修正 transfer 後的 B1R 10+90 candidate |
| `P2-Base-Direct` | 相同初始化、全模型直接 100-epoch diagnostic |
| `P2-Base-Selected` | validation 凍結的 P2 family parent |
| `P3-Base-Original` | 原始 B0 三尺度 P3 family parent |

A/B stage 只用於 pipeline 內部，不作報告主要實驗名稱。新 artifact directories 使用 lowercase snake_case。

## P2/P3 實驗矩陣

P2 family 共同父權重為 `P2-Base-Selected`，模組只放在 P2 feature slot。P3 family 共同父權重為 `P3-Base-Original`，維持三尺度 P3/P4/P5 Detect，模組只放在 P3 feature slot。

| 配方名稱 | 舊 alias | processed channels | branches / 公式 |
|---|---|---:|---|
| `PaperFormula-Full` | M0 | 100% | 3、5、F7、F9；完整式 (1)–(6) |
| `Lite-35` | M1 | 100% | 3、5；沿用雙 1×1 residual skeleton |
| `Lite-35-F7` | M4 | 100% | 3、5、F7；沿用雙 1×1 residual skeleton |
| `Partial50-35` | M2 | 50% | processed branch 使用 Lite-35；50% exact bypass |
| `Partial25-35` | M3 | 25% | processed branch 使用 Lite-35；75% exact bypass |

正式 ID 在配方前加 `P2-` 或 `P3-`。`F7/F9` 分別代表 `DW1×7→DW7×1` 與 `DW1×9→DW9×1`。

## MFAM variants 訓練

依 `MFAM_plan.md` 的正式消融規則，每個 P2/P3 variant：

- 從各自共同 parent 初始化所有可對應 tensors；新 MFAM random init。
- 先做 3-epoch engineering smoke，只檢查 forward/backward、NaN、save/reload 與顯存，不作論文結果。
- formal 為全模型不凍結直接 100 epochs。
- SGD、`lr0=0.001`、momentum 0.937、cosine、batch 16、seed 42、AMP 與完全相同 augmentation。
- validation best 作 canonical；不得用 test 選 epoch。
- 所有五種配方使用相同 `PaperFormulaMFAM` fusion/residual skeleton，只改表中 branches/processed ratio。

P2 與 P3 family 的 parent 不同，因此跨 family 是 operational comparison；嚴格消融先在各 family 內相對自己的 parent 判讀。

## 評估與報告

Validation 用於 gate、選 epoch 與 parent；test 只在所有選擇凍結後執行一次。每個 parent 與 variant 都輸出：

- overall：mAP50–95、mAP50、mAP75、AP_S、AP_M、AP_L。
- Ball/Bat：AP、AP50、AP75、AP_S、AP_M、AP_L、AR100。
- 固定 inference contract：precision、recall、prediction count、false positives、missed count。
- hardware：parameters、MACs、GFLOPs、mean/P95 latency、peak/P2/P3 activation。
- training：最佳 epoch、best/last gap、loss curves、trainable tensor count、initializer lineage 與 hashes。

COCO AP_S/AP_M/AP_L 與 baseball-specific tiny/small/large recall 分表呈現。報告必須分開列 paper-reported settings、repo actual settings，以及公式忠實/placement adaptation。

## Artifact 與相容性

- 新 artifact root：`artifacts/b1r-p2-p3-retest/`。
- Phase 1 的 `artifacts/static-phase1/`、checkpoint、selection 與 report 全部唯讀保留。
- 新 IDs 不加入舊 Phase 1 contracts；使用獨立 config、pipeline state、stage manifests 與 final audit。
- pipeline 使用單一 GPU queue，排在既有 GPU 工作後，不啟動平行訓練。

## GPU 執行順序

1. CPU contracts、unit tests、strict reload、transfer audit、loss/backward/step preflight。
2. `P2-Base-Staged`：B1R-A 10 epochs，再 B1R-B 90 epochs。
3. validation 診斷；若觸發條件，執行 `P2-Base-Direct` 100 epochs。
4. 只用 validation 凍結 `P2-Base-Selected`。
5. P2 五組依序做 3-epoch smoke：PaperFormula-Full、Lite-35、Lite-35-F7、Partial50-35、Partial25-35。
6. P2 五組依相同順序完成 formal 100 epochs。
7. P3 五組依相同順序完成 smoke，再依相同順序完成 formal 100 epochs。
8. 凍結所有 validation selections 後，統一執行 test 與同硬體 profile。
9. final audit、中文詳細報告與 GitHub push。

## 完成條件

- B1R 對所有 B0 source tensors 的成功 transfer coverage 為 100%，shape mismatch 與 unexpected source 為 0。
- destination missing 僅包含 manifest 白名單的新 P2/neck tensors。
- `PaperFormulaMFAM` 通過式 (1)–(6) 數值單元測試；Partial bypass 精確不變。
- CPU preflight 不使用 GPU 且所有測試通過。
- GPU pipeline 可安全續跑、direct fallback 條件可重現、舊 artifacts 不被修改。
- 報告清楚區分 B0 高分原因、B1R 修正、paper formula fidelity、P2/P3 placement adaptations 與 data exposure/單 seed 限制。
- 不預先宣稱 B1R 追平 B0；只以凍結後的 validation/test 證據判定。
