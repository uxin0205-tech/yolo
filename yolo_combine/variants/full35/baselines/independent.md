# Full35 不融合基線

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

## COCO Detect（Full35-A2）

COCO2017 有 118,287 train／5,000 val images；train person/sports-ball/baseball-bat
instances 分別為 257,252／6,299／3,273，val 為 10,777／260／145。此 baseline
保留完整 COCO80 head，應用層才只取 person。

| 評估 | Overall mAP50-95 | Person AP | Ball AP | Bat AP |
|---|---:|---:|---:|---:|
| COCO internal Bit-True | 0.506391 | 0.625917 | 0.510611 | 0.489142 |
| COCO canonical API | 0.512358 | — | 0.514185 | 0.490896 |
| BBT5 Detect internal | 0.399448 | — | 0.314137 | 0.484759 |

Full35 Detect 有 21,973,037 parameters、76.3789 GFLOPs（640）。Float checkpoint
供續訓；上表 AP 均來自 Bit-True PWL。既有測速環境是 RTX 5060 Ti、batch 8，
COCO inference 約 11.005 ms/image，不能當 RTX 5090 或 FPGA 結論。

## 架構匹配 Full35 Pose26 P1

| 項目 | 數值 |
|---|---:|
| Epochs | 10 |
| Physical batch / nbs / 穩態累積 | 16 / 64 / 約 4 |
| Trainable scope | 只有 fresh Pose26 head |
| Box mAP50-95 | 0.52053 |
| Pose mAP50-95 | 0.81622 |
| Ball / bat Pose AP（既有報告） | 0.796 / 0.836 |

因此 P1 有重訓，但不是 batch-128 的官方完整預訓練，也不是已完成的 P2/P3。
資料仍用 basic split，結果必須標為 provisional。

## 參數比較的正確解讀

Full35 架構匹配的兩個完整 task graph 合計 45,580,762 parameters；初始化時共享
trunk 的雙 head graph 為 26,529,701，少 19,051,061（41.7963%）。這是刪除
重複 trunk 的圖結構事實，不代表精度、MAC、latency 或 FPGA 資源也下降 41.8%。
