# Partial75 不融合基線

這份報告只整理既有 artifact，建立時沒有重新跑 validation 或訓練，也沒有使用 GPU。

## 原始 BBT5 Pose（YOLO11m）

| 項目 | 數值 |
|---|---:|
| Parameters | 20,904,360 |
| Train / val images | 6,080 / 567 |
| Epochs | 30 |
| Physical batch / nbs | 16 / 64 |
| Box mAP50-95 | 0.63036 |
| Pose mAP50-95 | 0.88204 |
| Pose mAP50 | 0.88697 |

Checkpoint：`/home/uxin/yolo/original/pose/weight/yolo11m_bat.pt`；SHA256
`7a8c720b8f5acef3a3cb465fbffb1b41b2dd3fee4519600389cdcd0d58c54009`。
它是 YOLO11m 歷史 baseline/teacher，不可直接當 Pose26 head 權重；原始 split
另有 10 個 source-group overlap。

## COCO Detect（Partial75-A2）

COCO2017 有 118,287 train／5,000 val images；train person/sports-ball/baseball-bat
instances 分別為 257,252／6,299／3,273，val 為 10,777／260／145。此 baseline
保留完整 COCO80 head，應用層才只取 person。

| 評估 | Overall mAP50-95 | Person AP | Ball AP | Bat AP |
|---|---:|---:|---:|---:|
| COCO internal Bit-True | 0.506754 | 0.626446 | 0.515439 | 0.486951 |
| COCO canonical API | 0.512689 | — | 0.518819 | 0.488661 |
| BBT5 Detect internal | 0.401554 | — | 0.313672 | 0.489437 |

Partial75 Detect 有 21,903,917 parameters、75.4794 GFLOPs（640）。Float checkpoint
供續訓；上表 AP 均來自 Bit-True PWL。既有測速環境是 RTX 5060 Ti、batch 8，
COCO inference 約 10.932 ms/image，不能當 RTX 5090 或 FPGA 結論。

## 架構匹配 Partial75 Pose26

目前只有 graph 建構與參數量 23,538,605 的證據；P1/P2/P3 都尚未跑，所以沒有
可宣稱的 Partial75 Pose 精度。Full35 P1 的 0.81622 不得借用到本變體。

## 參數比較的正確解讀

Partial75 架構匹配的兩個完整 task graph 合計 45,442,522 parameters；初始化時共享
trunk 的雙 head graph 為 26,460,581，少 18,981,941（41.7713%）。這是刪除
重複 trunk 的圖結構事實，不代表精度、MAC、latency 或 FPGA 資源也下降 41.8%。
