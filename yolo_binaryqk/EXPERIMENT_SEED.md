# Experiment seed record

本專案所有 26 個 canonical BinaryAttention 消融實驗統一使用：

```text
seed = 0
```

- E0、E1-S、E1、E2-DUAL：zero-training validation，`status.json` 與
  `training_args.json` 均記錄 seed 0。
- T0–T7 與 N4 共 22 個正式 fine-tuning runs：訓練 seed 0，且
  `deterministic=True`。
- COCO train manifest：seed 0，118,287 張影像，list SHA-256 為
  `aabf751c205f9620a0e050f4f83166dd0ed608ec2188f0ef0d3dd94f65926829`。
- 此批結果只有單一 seed，不可解讀為多 seed 的統計顯著性結果。

機器可讀紀錄位於
[`artifacts/reports/experiment_seed.json`](artifacts/reports/experiment_seed.json)。
若未來新增其他 seed，必須建立新的 runs，不可覆寫本批 seed 0 artifacts，並應在
variant/seed 層級分開彙整。

