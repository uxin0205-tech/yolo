# p3-lite-35-f7

## 做法

維持 B0 三尺度 Detect，只在 P3 feature slot 使用 DW3+DW5+factorized DW7。

- 訓練：先完成 3-epoch smoke 工程檢查，再從指定 parent 初始化，全模型 formal 100 epochs（lr0=0.001）。
- 共通條件：imgsz=640、batch=16、seed=42、SGD、momentum=0.937、cosine LR、AMP、deterministic。
- 資料：固定 BBT5 80/10/10 group/hash-audited split。
- 權重狀態：**缺失**；lineage 保留原路徑與 SHA-256，但 runtime checkpoint 已不在磁碟，無法誠實上傳。
- 記錄 SHA-256：`629f5abece8b4c1fcd7a713c1e12edc1784ce88e12ea00f53f7db3984df3ecb7`

## 精確結果

| Split | mAP50–95 | mAP50 | AP75 | AP_S | AP_M | AP_L | Ball AP | Ball AP_S | Ball Recall | Bat AP | Ball FP | Bat FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.710517 | 0.980753 | 0.772772 | 0.599080 | 0.810840 | 0.892349 | 0.567198 | 0.546834 | 0.983092 | 0.853836 | 838 | 876 |
| test | 0.740361 | 0.927071 | 0.810478 | 0.537943 | 0.847150 | 0.825202 | 0.633581 | 0.425883 | 0.958250 | 0.847142 | 3,273 | 1,744 |

## 硬體靜態成本

| Params | GFLOPs | P2 activation bytes | Peak live activation bytes | Feature traffic bytes |
|---:|---:|---:|---:|---:|
| 20,200,982 | 69.481 | 6,553,600 | 26,214,400 | 718,435,200 |

## 檔案

- [`val_metrics.json`](val_metrics.json)
- [`test_metrics.json`](test_metrics.json)
- [`profile.json`](profile.json)
- [`checkpoint.json`](checkpoint.json)
