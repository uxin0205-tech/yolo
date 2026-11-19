# 2026-09-15：Rep17＋20 E5 與 YOLO26m 評估口徑稽核

## 任務與範圍

確認目前交付權重與原生 YOLO26m 已有成績的計算口徑；本次沒有重新推論。

## 變更內容與原因

新增本紀錄與索引，避免將 Ultralytics 內部 AP 與 COCO 標準協定 AP 混用。

## 驗證方式與結果

核對兩份權重 SHA-256，分別為 9d9306f3574eb584c5adbe43c303afb2865109234ddf6dcb4c866b63901fc161（Rep17＋20）與 401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7（yolo_attention/weights/yolo26m.pt），皆符合結果來源。

Rep17＋20 的 run_arm.py 呼叫 pose_masf_training_v1/training_b.py 的 validate；該函式將 Detect validator 替換為 InternalValidator，停用 COCO JSON 評估。JointValidator 從 metrics.box.map 與 class_result 取得 AP。YOLO26m baseline-fp 的 validate.py 同樣使用 InternalValidator，eval_json 直接回傳內部結果。

共同 COCO 設定為完整 val 5,000 張、imgsz=640、batch=32、Ultralytics 內部 AP。YOLO26m 為 Float；Rep17＋20 同時保存 Float／BitTrue，交付預設 BitTrue。BitTrue 表示模型計算路徑，不代表另一套 AP 計算方式。

| 模型 | COCO mAP50:95（%） | person AP50:95（%） |
| --- | ---: | ---: |
| YOLO26m Float | 51.8019 | 63.0795 |
| Rep17＋20 E5 BitTrue | 50.5869 | 62.6986 |
| Rep17＋20 E5 Float | 50.5919 | 62.7037 |

BBAT 使用 registry /home/uxin/yolo/configs/datasets/bbat5-v1.yaml 對應的不可變 formal val 683 張，Pose runtime view 保留原 assignment 與 labels，batch=16。框 AP 採 IoU 0.50:0.05:0.95，關鍵點 AP 採 OKS 0.50:0.05:0.95，兩個 keypoints 的 sigma=[0.5,0.5]，area=GT box area×0.53。整體為兩類平均，ball／bat 另列。Precision／Recall 從框或 Pose 的平均 F1 曲線最大值對應門檻取得，不是固定 conf=0.25。

來源：yolo_optimize/model/metrics.json、model/runtime/yolo_combine/validation.py、model/runtime/ultralytics/models/yolo/pose/val.py、experiments/pose_masf_training_v1/training_b.py、experiments/studies/pre-fusion-full35-b100/scripts/validate.py 及 artifacts/baseline-fp/summary.json。

## 困難與解法

無。僅讀取現有結果、追蹤程式與核對雜湊；未修改模型或資料集。

## 未解事項或風險

本次未補跑兩模型的 COCO 標準協定 AP，不能將現有內部 AP 直接當成該協定結果。與上一輪 YOLO11m 比較應使用其內部 AP；不同模型仍保留各自原生後處理。
