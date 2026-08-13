# p3-partial50-35

## 做法

維持 B0 三尺度 Detect，只讓 P3 前 50% channels 通過 DW3+DW5。

- 訓練：先完成 3-epoch smoke 工程檢查，再從指定 parent 初始化，全模型 formal 100 epochs（lr0=0.001）。
- 共通條件：imgsz=640、batch=16、seed=42、SGD、momentum=0.937、cosine LR、AMP、deterministic。
- 資料：固定 BBT5 80/10/10 group/hash-audited split。
- 權重狀態：**缺失**；lineage 保留原路徑與 SHA-256，但 runtime checkpoint 已不在磁碟，無法誠實上傳。
- 記錄 SHA-256：`3c599802446c6950742c2e50b151b21a31cc5ea6da5c38fbf77bfaa8d2e2bf89`

## 精確結果

| Split | mAP50–95 | mAP50 | AP75 | AP_S | AP_M | AP_L | Ball AP | Ball AP_S | Ball Recall | Bat AP | Ball FP | Bat FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.740887 | 0.981598 | 0.826012 | 0.651320 | 0.844477 | 0.939816 | 0.618524 | 0.603161 | 0.983092 | 0.863249 | 1,968 | 2,210 |
| test | 0.743664 | 0.925931 | 0.816730 | 0.571497 | 0.851336 | 0.825575 | 0.640538 | 0.461478 | 0.968191 | 0.846790 | 3,511 | 3,295 |

## 硬體靜態成本

| Params | GFLOPs | P2 activation bytes | Peak live activation bytes | Feature traffic bytes |
|---:|---:|---:|---:|---:|
| 20,092,694 | 68.121 | 6,553,600 | 26,214,400 | 666,006,400 |

## 檔案

- [`val_metrics.json`](val_metrics.json)
- [`test_metrics.json`](test_metrics.json)
- [`profile.json`](profile.json)
- [`checkpoint.json`](checkpoint.json)
