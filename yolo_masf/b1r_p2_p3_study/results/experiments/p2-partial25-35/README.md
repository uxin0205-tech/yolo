# p2-partial25-35

## 做法

在 P2 slot 只讓前 25% channels 通過 DW3+DW5，其餘 identity bypass。

- 訓練：先完成 3-epoch smoke 工程檢查，再從指定 parent 初始化，全模型 formal 100 epochs（lr0=0.001）。
- 共通條件：imgsz=640、batch=16、seed=42、SGD、momentum=0.937、cosine LR、AMP、deterministic。
- 資料：固定 BBT5 80/10/10 group/hash-audited split。
- 權重狀態：**缺失**；lineage 保留原路徑與 SHA-256，但 runtime checkpoint 已不在磁碟，無法誠實上傳。
- 記錄 SHA-256：`852f7d2151ca850c93ad839f317fd1bd6a6b6ff339b495ea9ff9d3444e3f207c`

## 精確結果

| Split | mAP50–95 | mAP50 | AP75 | AP_S | AP_M | AP_L | Ball AP | Ball AP_S | Ball Recall | Bat AP | Ball FP | Bat FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.709068 | 0.982628 | 0.764568 | 0.585228 | 0.818818 | 0.943505 | 0.556807 | 0.535125 | 0.987923 | 0.861329 | 1,529 | 1,636 |
| test | 0.742383 | 0.936557 | 0.791713 | 0.545146 | 0.861847 | 0.830756 | 0.635180 | 0.428112 | 0.988072 | 0.849587 | 3,885 | 2,442 |

## 硬體靜態成本

| Params | GFLOPs | P2 activation bytes | Peak live activation bytes | Feature traffic bytes |
|---:|---:|---:|---:|---:|
| 20,872,088 | 89.122 | 13,107,200 | 26,214,400 | 1,072,944,000 |

## 檔案

- [`val_metrics.json`](val_metrics.json)
- [`test_metrics.json`](test_metrics.json)
- [`profile.json`](profile.json)
- [`checkpoint.json`](checkpoint.json)
