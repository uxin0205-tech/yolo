# p2-lite-35-f7

## 做法

在 P2 slot 使用 DW3+DW5+factorized DW7，全通道。

- 訓練：先完成 3-epoch smoke 工程檢查，再從指定 parent 初始化，全模型 formal 100 epochs（lr0=0.001）。
- 共通條件：imgsz=640、batch=16、seed=42、SGD、momentum=0.937、cosine LR、AMP、deterministic。
- 資料：固定 BBT5 80/10/10 group/hash-audited split。
- 權重狀態：**缺失**；lineage 保留原路徑與 SHA-256，但 runtime checkpoint 已不在磁碟，無法誠實上傳。
- 記錄 SHA-256：`47b82f0a3a37d2e7c50dc0df92b35b04cfe40b468b378137813e86441f9e176f`

## 精確結果

| Split | mAP50–95 | mAP50 | AP75 | AP_S | AP_M | AP_L | Ball AP | Ball AP_S | Ball Recall | Bat AP | Ball FP | Bat FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.710082 | 0.968672 | 0.781921 | 0.583057 | 0.842632 | 0.945297 | 0.558617 | 0.538417 | 0.987923 | 0.861548 | 4,854 | 1,383 |
| test | 0.736631 | 0.929353 | 0.806974 | 0.530559 | 0.834063 | 0.823871 | 0.624230 | 0.397513 | 0.976143 | 0.849031 | 9,549 | 2,827 |

## 硬體靜態成本

| Params | GFLOPs | P2 activation bytes | Peak live activation bytes | Feature traffic bytes |
|---:|---:|---:|---:|---:|
| 20,909,144 | 90.954 | 13,107,200 | 26,214,400 | 1,204,016,000 |

## 檔案

- [`val_metrics.json`](val_metrics.json)
- [`test_metrics.json`](test_metrics.json)
- [`profile.json`](profile.json)
- [`checkpoint.json`](checkpoint.json)
