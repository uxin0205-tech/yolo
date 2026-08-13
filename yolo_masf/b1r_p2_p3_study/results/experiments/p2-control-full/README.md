# p2-control-full

## 做法

承接 P2-Control-Head best，解除凍結後全模型訓練 80 epochs。

- 訓練：承接 control-head best，全模型 80 epochs，SGD，lr0=0.001。
- 共通條件：imgsz=640、batch=16、seed=42、SGD、momentum=0.937、cosine LR、AMP、deterministic。
- 資料：固定 BBT5 80/10/10 group/hash-audited split。
- 權重狀態：**缺失**；lineage 保留原路徑與 SHA-256，但 runtime checkpoint 已不在磁碟，無法誠實上傳。
- 記錄 SHA-256：`0c00db7370f15fc7d8784ccbddfdb39274f092077f2d90f987cc7ad5fc4f2dd3`

## 精確結果

| Split | mAP50–95 | mAP50 | AP75 | AP_S | AP_M | AP_L | Ball AP | Ball AP_S | Ball Recall | Bat AP | Ball FP | Bat FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.716379 | 0.982255 | 0.783925 | 0.596794 | 0.822491 | 0.950065 | 0.575507 | 0.555532 | 0.980676 | 0.857250 | 1,497 | 1,602 |
| test | 0.747240 | 0.939041 | 0.811402 | 0.562834 | 0.856095 | 0.831980 | 0.645597 | 0.461209 | 0.988072 | 0.848883 | 3,956 | 2,235 |

## 硬體靜態成本

| Params | GFLOPs | P2 activation bytes | Peak live activation bytes | Feature traffic bytes |
|---:|---:|---:|---:|---:|
| 20,868,696 | 88.962 | 13,107,200 | 26,214,400 | 1,046,729,600 |

## 檔案

- [`val_metrics.json`](val_metrics.json)
- [`test_metrics.json`](test_metrics.json)
- [`profile.json`](profile.json)
- [`checkpoint.json`](checkpoint.json)
