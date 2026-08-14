# Clean Initializer 公平實驗

這條實驗線用 Ultralytics 官方 COCO80 `yolo11m.pt` 取代已接觸 BBT5 的 `yolo11m_bat_detect_init.pt`。本機權重已與官方 v8.3.0 release 逐 byte 比對，SHA-256 同為 `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`。

## 現在如何使用資料

| 資料 | 程式允許的用途 | 現況 |
|---|---|---|
| `train` | forward/backward、更新權重 | 模型會看；這是正常訓練。 |
| `validation` | 每輪驗證、checkpoint 與模型選擇 | 可以看；不可拿來回傳梯度。 |
| 現有 `test` | 選模凍結後的 historical report | 以前已反覆看過，因此不再稱 unseen。 |
| `final_holdout` | 最終一次 unseen claim | 尚未建立；程式目前會拒絕 unseen claim。 |

Clean worker 在開始訓練前會從 locked `data.yaml` 產生只含 `train`、`val` 的 YAML，實體移除 `test` key。因此 trainer 本身無法選到 historical test。這能限制程式可見範圍，但不能抹除研究者過去已看過 test 結果的事實。

## 實驗矩陣

嚴格公平組的 resolved training profile 完全相同：官方 clean initializer、640、batch 16、SGD、momentum 0.937、cosine LR、AMP、direct full-model 100 epochs、`lr0=0.001`，並跑 seeds 42/43/44。

| 最終模型 | 架構 | 分組 |
|---|---|---|
| B0-Clean | 原始 P3/P4/P5 三尺度 Detect | strict fair |
| P3-PaperFormula-Clean | 三尺度 Detect，P3 完整論文公式 MFAM | strict fair |
| P3-Partial25-Clean | 三尺度 Detect，P3 前 25% channels 使用 DW3+DW5 | strict fair |
| P2-Direct-Clean | P2/P3/P4/P5 四尺度 Detect | strict fair |
| P2-Control-Clean | 同一四尺度架構，head-only 20 epochs 再 full 80 epochs | optimization control；不混入 strict ranking |

## CPU feasibility 結果

執行 [`masf_yolo.clean.inspect`](../../masf_yolo/clean/inspect.py) 已確認：

| 代表模型 | Strides | Params | 轉移 tensors | 新 tensors | Shape mismatch |
|---|---|---:|---:|---:|---:|
| B0-Clean | 8/16/32 | 20,054,550 | 643 | 0 | 6 |
| P3-PaperFormula-Clean | 8/16/32 | 20,206,614 | 643 | 48 | 6 |
| P3-Partial25-Clean | 8/16/32 | 20,065,430 | 643 | 24 | 6 |
| P2-Direct/Control-Clean | 4/8/16/32 | 20,868,696 | 643 | 154 | 6 |

6 個 shape mismatches 都是 COCO 80 類輸出轉成 BBT5 2 類時必須重新初始化的 classification tensors；不是 transfer failure。所有代表模型的 unexpected source tensors 都是 0。

## 狀態與執行入口

- 設定：[`clean_ablation.yaml`](clean_ablation.yaml)
- CPU 證據：[`FEASIBILITY.json`](FEASIBILITY.json)
- 18 個 jobs：6 stages × 3 seeds，全部為 `prepared_not_queued`。
- 尚未啟動 trainer、尚未使用 GPU。

CPU 重新檢查：

```bash
CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m masf_yolo.clean.inspect \
  --config configs/clean/clean_ablation.yaml \
  --output configs/clean/FEASIBILITY.json
```

GPU 空閒後，才以 `masf_yolo.clean.worker` 明確啟動單一 job。P2-Control-Full 必須提供相同 seed 的 P2-Control-Head best checkpoint；其餘工作不接受 parent checkpoint。目前沒有自動 queue，因此閱讀或執行 inspect 都不會開始訓練。

## 尚未解決的限制

1. 現有 test 是 historical test；真正 unseen claim 仍需新增 final holdout。
2. 官方 COCO initializer 解決 initializer data exposure，但 COCO 本身仍是一般預訓練資料；這是正常 transfer learning，不是 random initialization。
3. 三個 seeds 完成前，不報告 mean ± std，也不宣稱 0.x AP 差異穩定。
