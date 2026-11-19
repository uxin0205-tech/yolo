# 2026-09-15：原始模型與最新模型 Recall 比較

## 任務與範圍

整理 YOLO26m、YOLO11m 與最新 Rep17＋20 E5 BitTrue 的 Recall。

## 變更內容與原因

新增結果整理與索引；不改模型與資料，不重新推論。

## 驗證方式與結果

YOLO26m 原 baseline-fp/summary.json 僅保存 AP；其 artifacts/logs/baseline-fp.log 第 7、8 行保存 COCO 整體 R=0.628、person R=0.779，因此只報約 62.8%／77.9%，不補造精度。

YOLO11m 取本次原始權重評估 results.json，COCO 整體 Recall=61.4591%，person=74.5699%。最新模型取 yolo_optimize/model/metrics.json 的 bittrue，整體=61.2056%，person=77.4837%。

BBAT5-v1 formal val 上，最新模型的框 Recall：整體 82.7241%、ball 80.6616%、bat 84.7866%；關鍵點 Recall：整體 83.6041%、ball 81.6794%、bat 85.5288%。原始 yolo11m_bat.pt 對應框 Recall 為 94.7789%、91.2276%、98.3302%；關鍵點為 95.2534%、91.9910%、98.5158%。

全部採 Ultralytics 內部 Recall，各模型取平均 F1 最佳門檻；COCO 整體為類別平均 Recall，不是 COCO AR@100。資料與標註均未改動。

## 困難與解法

YOLO26m JSON 未保存 Recall；從同一輪完整驗證 log 取得三位有效數字，明示約值，避免不必要重跑。

## 未解事項或風險

未取得原 YOLO26m 該輪 Recall 的更多小數位。不同模型的最佳 F1 confidence 門檻可能不同，本表不表示同一固定門檻下的 Recall。
