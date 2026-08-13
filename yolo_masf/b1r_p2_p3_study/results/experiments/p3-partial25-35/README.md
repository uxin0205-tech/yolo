# p3-partial25-35

## 做法

維持 B0 三尺度 Detect，只讓 P3 前 25% channels 通過 DW3+DW5。

- 訓練：先完成 3-epoch smoke 工程檢查，再從指定 parent 初始化，全模型 formal 100 epochs（lr0=0.001）。
- 共通條件：imgsz=640、batch=16、seed=42、SGD、momentum=0.937、cosine LR、AMP、deterministic。
- 資料：固定 BBT5 80/10/10 group/hash-audited split。
- 權重狀態：**缺失**；lineage 保留原路徑與 SHA-256，但 runtime checkpoint 已不在磁碟，無法誠實上傳。
- 記錄 SHA-256：`1cecd17f8f98b838c4a0af974571c7494bdd39a2dd0a4dc36ec275f3e18d7e16`

## 精確結果

| Split | mAP50–95 | mAP50 | AP75 | AP_S | AP_M | AP_L | Ball AP | Ball AP_S | Ball Recall | Bat AP | Ball FP | Bat FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.730748 | 0.980231 | 0.799209 | 0.612932 | 0.849366 | 0.954855 | 0.599314 | 0.580981 | 0.990338 | 0.862181 | 935 | 1,341 |
| test | 0.754951 | 0.926158 | 0.821969 | 0.576821 | 0.877733 | 0.845235 | 0.651076 | 0.468004 | 0.964215 | 0.858826 | 2,277 | 1,889 |

## 硬體靜態成本

| Params | GFLOPs | P2 activation bytes | Peak live activation bytes | Feature traffic bytes |
|---:|---:|---:|---:|---:|
| 20,065,430 | 67.779 | 6,553,600 | 26,214,400 | 652,899,200 |

## 檔案

- [`val_metrics.json`](val_metrics.json)
- [`test_metrics.json`](test_metrics.json)
- [`profile.json`](profile.json)
- [`checkpoint.json`](checkpoint.json)
