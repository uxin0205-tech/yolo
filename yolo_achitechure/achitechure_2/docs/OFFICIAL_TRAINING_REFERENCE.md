# 官方訓練參考與本機實作對照

本文件區分公開官方原則與 pinned 本機環境的實際行為；專案決策仍以 [EXPERIMENT_SPEC.md](../EXPERIMENT_SPEC.md) 為準。

## 主要來源

- [Ultralytics Train mode](https://docs.ultralytics.com/modes/train/)：pretrained transfer、resume、early stopping、batch/imgsz、MuSGD、freeze、AMP 與 `nbs`。
- [Ultralytics Pose task](https://docs.ultralytics.com/tasks/pose/)：Pose 訓練、驗證與指標。
- [Ultralytics v8.4.90 default.yaml](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/cfg/default.yaml)：版本固定的公開參數基準。
- [Ultralytics v8.4.90 Pose trainer](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/models/yolo/pose/train.py)：task-specific trainer。
- [Ultralytics v8.4.90 loss](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/utils/loss.py)：Pose／YOLO26 loss 行為。
- [PyTorch AO QAT workflow](https://docs.pytorch.org/ao/stable/workflows/qat.html)：prepare、fake-quant training 與 conversion；本專案自訂 graph 只宣稱 simulation。

## 2026-08-18 本機對照

| 項目 | 公開原則 | 本機 8.4.90 觀察 | 本案決策 |
|---|---|---|---|
| Transfer | 將 pretrained weights 載入 task model | trainer 支援 checkpoint/model loading | 每候選由同一已驗收 Float-PWL parent graft |
| Resume | 還原 weights、optimizer、scheduler、epoch | `check_resume` 會還原 args/state | 只有 D2/P4 從 `last.pt` 真正 resume |
| Freeze | `freeze` 可接受前 N 層或 indices | Pose 共用 BaseTrainer freeze | P1 凍結 0–22；P2 凍結 0–10；P3 fresh 全開 |
| MuSGD | YOLO26 hybrid Muon/SGD | 本機 builder 建立 Muon/SGD-style groups，layer-23 cv3 使用 3× LR | 明確 MuSGD 並記錄各有效群組 |
| Nominal batch | `nbs` 正規化 accumulation/loss | default `nbs=64` | physical 16、nbs 64 |
| OOM | trainer 可能自動減少 batch | 本機 source 最多重試三次並減半 | 禁止恢復，fail closed |
| Pose loss | 存在 box/cls/dfl/pose/kobj/rle | 預設 7.5/.5/1.5/12/1/1；`flow_model` 才啟用 RLE | 同值，RLE inactive 即拒絕 |
| Best metric | validator 提供 task metrics/fitness | Pose 同時有 Pose、Box 與 combined fitness | Pose mAP50-95 選 research best；combined 另存 |
| 水平翻轉 | Pose data 可定義 `flip_idx` | 本資料缺 keypoint 語意證明 | `fliplr=0`，待規格修訂 |
| QAT | prepare→fake-quant train→convert/evaluate | custom attention graph 無已驗證自動 conversion | W8A8 fake-quant，`simulation_only=true` |

## 聲明邊界

Detect 正式 YAML 是受控專案 recipe，只是參考本機 YOLO26 設定；不宣稱重製 Ultralytics 未公開的 pretraining 資料與內部條件。Pose 的 P1–P4 是官方 transfer/freeze/resume 機制加上本案 head stabilization，不宣稱是官方通用 recipe。Pose 在本專案預設停用，是否執行由使用者明確決定。
