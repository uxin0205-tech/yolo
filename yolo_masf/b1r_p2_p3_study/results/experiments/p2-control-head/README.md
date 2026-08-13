# p2-control-head

## 做法

B1R 的 P2 對照第一段；凍結 inherited backbone/neck 與 P3–P5 towers，只訓練 P2 slot/head 20 epochs。

- 訓練：P2 slot/head 20 epochs，SGD，lr0=0.01；其他 inherited layers 凍結。
- 共通條件：imgsz=640、batch=16、seed=42、SGD、momentum=0.937、cosine LR、AMP、deterministic。
- 資料：固定 BBT5 80/10/10 group/hash-audited split。
- 權重狀態：**缺失**；lineage 保留原路徑與 SHA-256，但 runtime checkpoint 已不在磁碟，無法誠實上傳。
- 記錄 SHA-256：`73238476c2530464efaf50259b81f7182c8064a6d21afa3d5b96c929f3caf4b1`

## 精確結果

| Split | mAP50–95 | mAP50 | AP75 | AP_S | AP_M | AP_L | Ball AP | Ball AP_S | Ball Recall | Bat AP | Ball FP | Bat FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.714415 | 0.985733 | 0.748020 | 0.590781 | 0.819962 | 0.957063 | 0.568670 | 0.546578 | 0.985507 | 0.860159 | 1,844 | 2,058 |
| test | 0.743269 | 0.931690 | 0.802777 | 0.557752 | 0.848344 | 0.821833 | 0.635860 | 0.453056 | 0.986083 | 0.850678 | 4,015 | 2,803 |

## 硬體靜態成本

| Params | GFLOPs | P2 activation bytes | Peak live activation bytes | Feature traffic bytes |
|---:|---:|---:|---:|---:|
| 20,868,696 | 88.962 | 13,107,200 | 26,214,400 | 1,046,729,600 |

## 檔案

- [`val_metrics.json`](val_metrics.json)
- [`test_metrics.json`](test_metrics.json)
- [`profile.json`](profile.json)
- [`checkpoint.json`](checkpoint.json)
