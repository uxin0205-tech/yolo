# p2-paperformula-full

## 做法

在 P2 slot 使用 PaperFormulaMFAM：DW3、DW5、factorized DW7、factorized DW9，全通道。

- 訓練：先完成 3-epoch smoke 工程檢查，再從指定 parent 初始化，全模型 formal 100 epochs（lr0=0.001）。
- 共通條件：imgsz=640、batch=16、seed=42、SGD、momentum=0.937、cosine LR、AMP、deterministic。
- 資料：固定 BBT5 80/10/10 group/hash-audited split。
- 權重狀態：**缺失**；lineage 保留原路徑與 SHA-256，但 runtime checkpoint 已不在磁碟，無法誠實上傳。
- 記錄 SHA-256：`ae7d2c079b0a6c0e1631f50974b3045c3a0225599e08a9daa4bb0ee58e863589`

## 精確結果

| Split | mAP50–95 | mAP50 | AP75 | AP_S | AP_M | AP_L | Ball AP | Ball AP_S | Ball Recall | Bat AP | Ball FP | Bat FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.716556 | 0.981565 | 0.777770 | 0.584432 | 0.812582 | 0.937765 | 0.577014 | 0.563443 | 0.983092 | 0.856097 | 4,091 | 1,958 |
| test | 0.743312 | 0.937728 | 0.799498 | 0.535394 | 0.868195 | 0.831479 | 0.637197 | 0.416875 | 0.974155 | 0.849428 | 8,525 | 3,868 |

## 硬體靜態成本

| Params | GFLOPs | P2 activation bytes | Peak live activation bytes | Feature traffic bytes |
|---:|---:|---:|---:|---:|
| 20,911,960 | 91.072 | 13,107,200 | 26,214,400 | 1,256,444,800 |

## 檔案

- [`val_metrics.json`](val_metrics.json)
- [`test_metrics.json`](test_metrics.json)
- [`profile.json`](profile.json)
- [`checkpoint.json`](checkpoint.json)
