# 訓練指南

[EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md) 是唯一正式規格。本文件只解釋設定與階段轉換原因，不另立規則。

## Detect

D0 是獨立的 3–5 epoch 執行檢查，不參與精度排名。D1 是正式可比較訓練，最多 100 epochs、patience 20。D2 只有通過晚期改善門檻才可從 D1 `last.pt` 真正 resume，總 epochs 最多 140、patience 15。每個候選都從同一份已驗收 Float-PWL parent 獨立建立；shape 不相容張量採固定 seed 0 初始化，不能繼承另一候選。

Detect 每份完整正式 YAML 都明確保存 physical batch 16、`nbs=64`、imgsz 640、seed 0、AMP、deterministic 與 MuSGD；`config-check` 會比較共同欄位，禁止階段間漂移。本機 8.4.90 原本可能在首個 epoch CUDA OOM 後自動減半 batch；研究 trainer 禁止這項自動恢復，因為隱性 batch 差異會破壞 ablation 公平性。

Float EMA 維持可訓練格式。驗證時 deep-copy，再只把 inherited attention normalization 轉成 Bit-True PWL。Bit-True box mAP50-95 決定 best 與 early stopping；Float 指標只能另行記錄。

## MuSGD 與有效學習率群組

本機 Ultralytics 8.4.90 支援 `MuSGD`：rank 至少為 2 的參數進 Muon 群組，bias、normalization 與 scalar 參數保留 SGD-style；layer 23 的 `cv3` 與語意 auxiliary heads 另使用設定 LR 的 3 倍。每次完成都要記錄 param group、Muon 狀態、有效／初始 LR、weight decay 與參數量。禁止靜默改用 SGD、AdamW 或 `optimizer=auto`。

## Pose：預設停用，由使用者決定

資料分割與設定檢查不會建立、訓練或驗證 Pose 模型。只有明確提供 `--enable-pose --execute` 才允許進入 Pose 路線；Python 訓練入口也要求 `pose_opt_in=True`。每份 Pose 正式 YAML 都標示 `pose_opt_in_required=true` 與 `enabled_by_default=false`。

若啟用，PoseLoss26 使用 `box=7.5`、`cls=0.5`、`dfl=1.5`、`pose=12`、`kobj=1`、`rle=1`。僅接受 `rle` 參數不足以證明功能生效；materialized Pose head 沒有 `flow_model` 時會立即中止。

- P0：獨立 3 epoch smoke，不接入正式權重 lineage。
- P1：凍結 layers 0–22，只訓練 Pose head，最多 10 epochs、patience 5。
- P2：由 P1 `best.pt` 建立新 trainer/optimizer，凍結 layers 0–10，最多 20/8。
- P3：由 P2 `best.pt` 建立新 trainer/optimizer，完整 fine-tune，最多 100/20。
- P4：只有通過晚期改善門檻，才從 P3 `last.pt` resume，總 epochs 130、patience 10。

MASF 與 attention 的 parameters、BN buffers 及 train/eval 狀態全階段凍結。C0 與 C_best 使用相同 head seed、資料、stage、optimizer groups 與預算。研究選型使用 Pose mAP50-95，官方 Pose+Box combined fitness 另存但不改變 winner。

目前資料有兩個 keypoints 與 `flip_idx`，但沒有足以證明左右語意的 `kpt_names`，因此 `fliplr=0`。只有規格修訂並加入經測試的 mirror mapping 後才可開啟。

## 轉換與公平性

Detect 與 Pose 各階段使用一份完整正式 YAML，不再拼接 canonical、overlay 或 stage fragment。相同 task 的共同 learning fields 由 `config-check` 逐欄比較；runtime 只允許修改 `model/name/project/device/workers/cache`。P1→P2、P2→P3 是用 best weights 建立新 optimizer；只有 P3→P4、D1→D2 是從 `last.pt` 真正 resume。

## 失敗處理

- OOM：記錄模型、GPU、batch、imgsz、peak memory 與失敗位置；未修訂規格前不可只替單一候選縮小 batch。
- NaN/Inf：callback 立即中止；保存 manifest 與 logs，不可只替單一候選調 loss/LR。
- 中斷：fresh stage 從宣告輸入重跑；resume stage 只有通過 hash/provenance 檢查才可從自己的 `last.pt` 繼續。
- 權重缺失或 shape mismatch：以 transfer report 為準，不得宣稱完整 pretrained。
- Early stop：使用 `best.pt`，不執行 extension；跑滿 epochs 且通過門檻才可延長。
- Runtime overrides：只允許 I/O/runtime allowlist；CLI 偷改 learning fields 會在建立 trainer 前失敗。
- 正式 checkpoint：必須經 fresh-process reload、Bit-True validation 與一致 profiling 才可列入報告。
