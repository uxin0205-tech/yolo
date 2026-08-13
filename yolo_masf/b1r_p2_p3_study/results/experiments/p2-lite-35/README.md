# p2-lite-35

## 做法

在 P2 slot 使用 DW3+DW5 的 PaperFormulaMFAM，全通道。

- 訓練：先完成 3-epoch smoke 工程檢查，再從指定 parent 初始化，全模型 formal 100 epochs（lr0=0.001）。
- 共通條件：imgsz=640、batch=16、seed=42、SGD、momentum=0.937、cosine LR、AMP、deterministic。
- 資料：固定 BBT5 80/10/10 group/hash-audited split。
- 權重狀態：**缺失**；lineage 保留原路徑與 SHA-256，但 runtime checkpoint 已不在磁碟，無法誠實上傳。
- 記錄 SHA-256：`2eb66c558b73c4471196fa309ff1fe9dc7f44d7a6bd6741de2f06e5141fbc9fc`

## 精確結果

| Split | mAP50–95 | mAP50 | AP75 | AP_S | AP_M | AP_L | Ball AP | Ball AP_S | Ball Recall | Bat AP | Ball FP | Bat FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.716581 | 0.980712 | 0.770013 | 0.586931 | 0.828435 | 0.950495 | 0.577680 | 0.561102 | 0.997585 | 0.855482 | 2,986 | 1,490 |
| test | 0.739476 | 0.930979 | 0.796507 | 0.543006 | 0.847917 | 0.827503 | 0.637168 | 0.430819 | 0.986083 | 0.841784 | 6,841 | 2,524 |

## 硬體靜態成本

| Params | GFLOPs | P2 activation bytes | Peak live activation bytes | Feature traffic bytes |
|---:|---:|---:|---:|---:|
| 20,906,840 | 90.862 | 13,107,200 | 26,214,400 | 1,151,587,200 |

## 檔案

- [`val_metrics.json`](val_metrics.json)
- [`test_metrics.json`](test_metrics.json)
- [`profile.json`](profile.json)
- [`checkpoint.json`](checkpoint.json)
