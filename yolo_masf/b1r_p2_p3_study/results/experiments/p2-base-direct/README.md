# p2-base-direct

## 做法

B1R 四尺度 P2/P3/P4/P5 baseline；從來源初始化權重全模型直接訓練 100 epochs。

- 訓練：全模型 100 epochs，SGD，lr0=0.001。
- 共通條件：imgsz=640、batch=16、seed=42、SGD、momentum=0.937、cosine LR、AMP、deterministic。
- 資料：固定 BBT5 80/10/10 group/hash-audited split。
- 權重狀態：**缺失**；lineage 保留原路徑與 SHA-256，但 runtime checkpoint 已不在磁碟，無法誠實上傳。
- 記錄 SHA-256：`288a3c738b3cb772df0eb5be6d5448a7bf6ef5ce544242d056ae12e3f70ea2a5`

## 精確結果

| Split | mAP50–95 | mAP50 | AP75 | AP_S | AP_M | AP_L | Ball AP | Ball AP_S | Ball Recall | Bat AP | Ball FP | Bat FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.713362 | 0.985531 | 0.743976 | 0.589325 | 0.819962 | 0.957063 | 0.566564 | 0.543665 | 0.985507 | 0.860159 | 1,205 | 1,552 |
| test | 0.739114 | 0.925054 | 0.799827 | 0.548466 | 0.848344 | 0.821833 | 0.627660 | 0.434645 | 0.968191 | 0.850568 | 2,968 | 2,495 |

## 硬體靜態成本

| Params | GFLOPs | P2 activation bytes | Peak live activation bytes | Feature traffic bytes |
|---:|---:|---:|---:|---:|
| 20,868,696 | 88.962 | 13,107,200 | 26,214,400 | 1,046,729,600 |

## 檔案

- [`val_metrics.json`](val_metrics.json)
- [`test_metrics.json`](test_metrics.json)
- [`profile.json`](profile.json)
- [`checkpoint.json`](checkpoint.json)
