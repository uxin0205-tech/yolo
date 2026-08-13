# b0-original-3scale

## 做法

既有 B0 三尺度 P3/P4/P5 operational reference；未加入 P2 或 MFAM。

- 訓練：不在本輪重訓；直接評估來源初始化權重。
- 共通條件：imgsz=640、batch=16、seed=42、SGD、momentum=0.937、cosine LR、AMP、deterministic。
- 資料：固定 BBT5 80/10/10 group/hash-audited split。
- 權重狀態：可用；repo 內位置：[`../../../../bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`](../../../../bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt)。
- 記錄 SHA-256：`9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d`

## 精確結果

| Split | mAP50–95 | mAP50 | AP75 | AP_S | AP_M | AP_L | Ball AP | Ball AP_S | Ball Recall | Bat AP | Ball FP | Bat FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.750816 | 0.990140 | 0.827157 | 0.646280 | 0.843561 | 0.916396 | 0.632539 | 0.614263 | 0.990338 | 0.869092 | 579 | 344 |
| test | 0.770812 | 0.938334 | 0.839527 | 0.595169 | 0.879066 | 0.853723 | 0.678444 | 0.472178 | 0.966203 | 0.863179 | 1,204 | 485 |

## 硬體靜態成本

| Params | GFLOPs | P2 activation bytes | Peak live activation bytes | Feature traffic bytes |
|---:|---:|---:|---:|---:|
| 20,054,550 | 67.646 | 6,553,600 | 26,214,400 | 639,792,000 |

## 檔案

- [`val_metrics.json`](val_metrics.json)
- [`test_metrics.json`](test_metrics.json)
- [`profile.json`](profile.json)
- [`checkpoint.json`](checkpoint.json)
