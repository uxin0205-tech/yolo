# 2026-09-15：原始 YOLO11m 與棒球 Pose 權重評估

## 任務與範圍

核對 `/home/uxin/single_view_pipeline/single_view_pipeline/weights/yolo11m.pt` 在 COCO2017 的整體與 person AP50:95，以及同目錄 `yolo11m_bat.pt` 在 BBAT5 的框與關鍵點指標。

## 變更內容與原因

新增可重現評估腳本與原始結果；使用使用者指定的原始權重，不以其他訓練或量化模型代替。兩份權重 SHA-256 分別為 `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95` 與 `7a8c720b8f5acef3a3cb465fbffb1b41b2dd3fee4519600389cdcd0d58c54009`，與本專案 `original/` 中對應檔案一致。

COCO Detect 使用 `/home/uxin/yolo/coco2017.yaml`。BBAT Pose 使用 `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml`，dataset ID 為 `bbat5-v1`，registry 為 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`；split 為 formal val，共 683 張。樣本、標註與 assignment 維持不變；停用資料 cache 寫入，避免修改不可變資料目錄。

## 驗證方式與結果

以 Ultralytics 8.4.90、imgsz=640、batch=16、FP32、conf=0.001、NMS IoU=0.7、max_det=300、rect=True 重跑。COCO 額外使用 faster-coco-eval 1.7.2 的 COCO 標準協定 計算 maxDets=100 的官方 AP，與 Ultralytics 內部 AP 分開保存。BBAT 關鍵點使用此版本的預設 OKS sigma=[0.5, 0.5]。

腳本：[evaluate.py](assets/2026-09-15-original-weights-evaluation/evaluate.py)。完整結果：[results.json](assets/2026-09-15-original-weights-evaluation/results.json)。執行紀錄：[evaluation.log](assets/2026-09-15-original-weights-evaluation/evaluation.log)。


## 最終數值（百分比）


| COCO 計算口徑 | mAP50:95 | person AP50:95 |
| --- | ---: | ---: |
| COCO 標準協定 | 51.5144 | 63.7686 |
| Ultralytics 內部 | 50.9854 | 62.2683 |

| BBAT 任務 | 類別 | Precision | Recall | F1 | AP50 | AP75 | AP50:95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| box | all | 96.2238 | 94.7789 | 95.4955 | 96.2792 | 82.9094 | 74.8700 |
| box | ball | 92.9965 | 91.2276 | 92.1036 | 93.0911 | 68.4584 | 62.8783 |
| box | bat | 99.4510 | 98.3302 | 98.8875 | 99.4673 | 97.3604 | 86.8617 |
| pose | all | 96.7067 | 95.2534 | 95.9742 | 96.5016 | 96.5016 | 96.4810 |
| pose | ball | 93.7747 | 91.9910 | 92.8743 | 93.5161 | 93.5161 | 93.5161 |
| pose | bat | 99.6387 | 98.5158 | 99.0740 | 99.4871 | 99.4871 | 99.4458 |

完整 COCO val 共 5,000 張，BBAT formal val 共 683 張／932 個物件。兩次推論及 COCO 標準 AP 累積均已完成，未重複執行已完成的推論。

## 困難與解法

一般沙箱因 bwrap loopback 初始化失敗而無法執行，改由核准的提升權限命令執行。既有報告包含不同權重與不同 AP 計算口徑，因此本次直接評估指定權重並同時保存兩種 COCO AP。

停用 cache 寫入的初版函式未保留 version metadata，造成 BBAT 載入時 KeyError；修正為只在記憶體補上 version，保留已完成的 COCO 結果並續跑。系統沒有 python 別名，文件索引更新改用專案 .venv/bin/python。

環境沒有 pycocotools，改用已安裝的 faster-coco-eval 1.7.2 計算 COCO 標準協定 AP，無須重新推論或安裝套件。

## 未解事項或風險

BBAT 的結果是 formal val 評估，不是獨立 test；不據此推論原始訓練資料與此 val 之間沒有重疊。GPU 與現有工作共用，本次不以速度判斷模型效能。
