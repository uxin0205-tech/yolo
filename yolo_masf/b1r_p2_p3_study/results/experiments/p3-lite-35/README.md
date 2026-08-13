# p3-lite-35

## 做法

維持 B0 三尺度 Detect，只在 P3 feature slot 使用 DW3+DW5。

- 訓練：先完成 3-epoch smoke 工程檢查，再從指定 parent 初始化，全模型 formal 100 epochs（lr0=0.001）。
- 共通條件：imgsz=640、batch=16、seed=42、SGD、momentum=0.937、cosine LR、AMP、deterministic。
- 資料：固定 BBT5 80/10/10 group/hash-audited split。
- 權重狀態：**缺失**；lineage 保留原路徑與 SHA-256，但 runtime checkpoint 已不在磁碟，無法誠實上傳。
- 記錄 SHA-256：`a7ae19d1ec875ba8d14cf72b87823aac0457e185247e7755c850aa06a049dc86`

## 精確結果

| Split | mAP50–95 | mAP50 | AP75 | AP_S | AP_M | AP_L | Ball AP | Ball AP_S | Ball Recall | Bat AP | Ball FP | Bat FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.705104 | 0.983540 | 0.734492 | 0.579687 | 0.827268 | 0.928012 | 0.562801 | 0.543305 | 0.987923 | 0.847406 | 1,924 | 7,448 |
| test | 0.727651 | 0.922053 | 0.773276 | 0.525245 | 0.826482 | 0.805815 | 0.620818 | 0.424938 | 0.970179 | 0.834484 | 4,676 | 4,636 |

## 硬體靜態成本

| Params | GFLOPs | P2 activation bytes | Peak live activation bytes | Feature traffic bytes |
|---:|---:|---:|---:|---:|
| 20,196,374 | 69.435 | 6,553,600 | 26,214,400 | 692,220,800 |

## 檔案

- [`val_metrics.json`](val_metrics.json)
- [`test_metrics.json`](test_metrics.json)
- [`profile.json`](profile.json)
- [`checkpoint.json`](checkpoint.json)
