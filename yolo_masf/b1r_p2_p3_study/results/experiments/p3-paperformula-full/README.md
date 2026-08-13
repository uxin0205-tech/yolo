# p3-paperformula-full

## 做法

維持 B0 三尺度 Detect，只在 P3 feature slot 使用完整 PaperFormulaMFAM。

- 訓練：先完成 3-epoch smoke 工程檢查，再從指定 parent 初始化，全模型 formal 100 epochs（lr0=0.001）。
- 共通條件：imgsz=640、batch=16、seed=42、SGD、momentum=0.937、cosine LR、AMP、deterministic。
- 資料：固定 BBT5 80/10/10 group/hash-audited split。
- 權重狀態：**缺失**；lineage 保留原路徑與 SHA-256，但 runtime checkpoint 已不在磁碟，無法誠實上傳。
- 記錄 SHA-256：`e4a62d9b4e30f3012b4b4bbe77b0677e0b3c9b29b3d7a0132067eae17eac3fca`

## 精確結果

| Split | mAP50–95 | mAP50 | AP75 | AP_S | AP_M | AP_L | Ball AP | Ball AP_S | Ball Recall | Bat AP | Ball FP | Bat FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.712205 | 0.980154 | 0.775745 | 0.604751 | 0.824485 | 0.880339 | 0.567014 | 0.550607 | 0.985507 | 0.857395 | 754 | 1,124 |
| test | 0.750617 | 0.943264 | 0.796959 | 0.557967 | 0.871743 | 0.827096 | 0.643119 | 0.423240 | 0.968191 | 0.858115 | 2,387 | 2,012 |

## 硬體靜態成本

| Params | GFLOPs | P2 activation bytes | Peak live activation bytes | Feature traffic bytes |
|---:|---:|---:|---:|---:|
| 20,206,614 | 69.540 | 6,553,600 | 26,214,400 | 744,649,600 |

## 檔案

- [`val_metrics.json`](val_metrics.json)
- [`test_metrics.json`](test_metrics.json)
- [`profile.json`](profile.json)
- [`checkpoint.json`](checkpoint.json)
