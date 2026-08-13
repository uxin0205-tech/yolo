# p2-partial50-35

## 做法

在 P2 slot 只讓前 50% channels 通過 DW3+DW5，其餘 identity bypass。

- 訓練：先完成 3-epoch smoke 工程檢查，再從指定 parent 初始化，全模型 formal 100 epochs（lr0=0.001）。
- 共通條件：imgsz=640、batch=16、seed=42、SGD、momentum=0.937、cosine LR、AMP、deterministic。
- 資料：固定 BBT5 80/10/10 group/hash-audited split。
- 權重狀態：**缺失**；lineage 保留原路徑與 SHA-256，但 runtime checkpoint 已不在磁碟，無法誠實上傳。
- 記錄 SHA-256：`e59241e7accf1e18f22a8f68c6b050a1a8983ebbd10659e3dd1747fe5bd000d6`

## 精確結果

| Split | mAP50–95 | mAP50 | AP75 | AP_S | AP_M | AP_L | Ball AP | Ball AP_S | Ball Recall | Bat AP | Ball FP | Bat FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.709980 | 0.971308 | 0.765888 | 0.589673 | 0.828643 | 0.928112 | 0.563764 | 0.543003 | 0.978261 | 0.856196 | 1,897 | 2,004 |
| test | 0.741984 | 0.933336 | 0.803077 | 0.549337 | 0.841836 | 0.827648 | 0.642847 | 0.447519 | 0.988072 | 0.841120 | 4,677 | 2,892 |

## 硬體靜態成本

| Params | GFLOPs | P2 activation bytes | Peak live activation bytes | Feature traffic bytes |
|---:|---:|---:|---:|---:|
| 20,879,576 | 89.493 | 13,107,200 | 26,214,400 | 1,099,158,400 |

## 檔案

- [`val_metrics.json`](val_metrics.json)
- [`test_metrics.json`](test_metrics.json)
- [`profile.json`](profile.json)
- [`checkpoint.json`](checkpoint.json)
