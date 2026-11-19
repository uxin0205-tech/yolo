# 2026-11-11：Stage9 完成、最終形狀選型與交付

## 任務與範圍

使用者詢問目前狀況。本次核對背景 queue 的最終結果，完成既有 Stage9 的分析與交付，沒有新增候選、GPU 工作或下一階段。初輪 40E 與擴充 60E 於 **2026-11-11 06:25:37（Asia/Taipei）** 全部完成；上午實際核對 supervisor 已退出、GPU 無計算程序。

## 研究結果與決策

五組均完成 10E＋半 LR 10E，全部按預算結束。正式保留使用者指定的 **qSiLU 訓練版 Rep17＋20 E1**，搭配原 batch rect640 letterbox 流程。沒有固定尺寸通過全部採用條件；沒有把單項最高 AP 拼成一個不存在的模型。

| 尺寸／模型 | 最佳段落 | COCO AP50–95 | person AP50–95 | BBAT box AP50–95 | Pose AP50–95 |
|---|---|---:|---:|---:|---:|
| 原 qSiLU Rep17＋20／rect640 | Stage8 E1 | 51.4347% | 64.2796% | 61.4371% | 90.1389% |
| 384×768 | adapt E4 | 50.2160% | 62.4010% | 61.7208% | 89.8808% |
| 512×1024 | adapt E4 | 51.8553% | 64.6954% | 61.6512% | 89.9204% |
| 256×512 | low-LR E5 | 44.9273% | 56.2016% | 55.6881% | 86.0849% |
| 384×512 | low-LR E1 | 49.4749% | 61.6394% | 60.6093% | 89.8889% |
| 512×768 | adapt E2 | 51.9632% | 64.5865% | 62.4501% | 89.9026% |

512×768 的 COCO AP 提高 0.5285 個百分點，但 COCO 球棒 AP 由 46.9576% 降為 45.3351%，Pose 亦下降 0.2364 個百分點，未通過採用條件。半 LR 10E 沒有超過同組 adapt E2 的綜合選型。完整十項 AP、E0、成本與負結果見 [正式報告](../../yolo_actshape/result/stage9/README.md)。

長方形與二次冪尺寸都能執行，不表示一定更省或精度更好。512×768 的分析 Conv＋P.V MAC 較**固定 640×640** 減少約 4.01%，512×1024 反而增加約 28.06%；不能拿這兩個數字代表相對原動態 rect640 流程的成本差。GPU 同步中位時間分別為 29.6809 ms 與固定 640×640 的 29.6725 ms，未量到前者加速。

## 交付整理與清理

正式入口為 [result/stage9](../../yolo_actshape/result/stage9/README.md)，[handoff](../../yolo_actshape/handoff/README.md) 指向其中的交接包。權重 SHA256 為 `2454548104ed64768eb9ecdb99d44804ee9d8e1e371e2a508cf052c0911fdd35`；qSiLU／D1 整數 PWL exp＋LUT256 reciprocal／B2／S1／folded17+20 保持。

背景 finalizer 的兩輪清理共移除 14 份 optimizer resume／未採用權重，合計 **3,832,988,886 bytes（3.56975 GiB）**。本次逐項核對皆已不存在，沒有新增刪除、沒有 archive／backup。初輪兩個適應權重已刪除，保留數據；新增三個尺寸的最佳權重保留，供硬體取捨。清單見 [初輪](../../yolo_actshape/result/stage9/cleanup.json) 與 [擴充](../../yolo_actshape/result/stage9/extension/cleanup.json)。

同步根層 README、整合計畫、子專案入口與硬體分析的完成狀態，修正計畫內的結果相對連結。交付版 `redraw.py` 改為共用尺寸列的三欄圖，解決原散點標籤重疊，新增 SVG；只改結果目錄的展示程式，不修改已鎖定實驗程式、protocol、原始 metrics 或 checkpoint。另以小型索引連結校準恢復的政策、來源 SHA 與驗證證據。

## 驗證方式與結果

- 五組各 20 個完整 epoch，總計 100E；五組最佳重載後的十項 AP 最大誤差均為 0，算子契約通過。
- 實際核對交接 manifest 的 **372 個檔案**大小及 SHA、14 列 CSV 與完整 JSON，以及仍可用候選的 checkpoint SHA；均相符。
- 七組尺寸各 42 項、共 **294 項 GPU replay** 通過，模型 state 不變；每任務一張的 replay 不能代替 COCO 5000／BBAT5 v1 683 張完整 AP。
- 既有可攜包獨立 CPU 驗證通過，bitwise equal，故意替換 SiLU 會被拒絕。沒有重跑已通過的模型測試或 GPU 評估。
- 本次僅將 `redraw.py` 與 `metrics.csv` 放入獨立暫存目錄、禁用 CUDA，成功重畫 PNG／SVG；目錄無 checkpoint，CSV SHA 保持 `c06e136601ab41c15f9d83c1aa78672a16ae449a1fd369bb66de04cf316c79bf`。視覺核對標籤可讀、圖例與參考線完整。
- 文件連結與交付展示檔案另作核對，結果收錄 [final-audit.json](../../yolo_actshape/result/stage9/final-audit.json)。

## 困難與解法

本次沒有新的訓練故障。自動交付後根層與計畫仍有 88／100 輪的進度描述，現已更新；圖表的散點標籤過密，改用對齊尺寸列解決，數據保持不變。文件檢查發現報告內兩個工作紀錄連結多退一層，已修正相對路徑並重新查核。

先前電腦重啟從有效 checkpoint 接續；CUDA 原生崩潰核查後只作一次有界恢復。11 月 10 日的 256×512 低 LR 梯度校準先重現原失敗，再參照 optimize 的 16 批量測／8 批確認，保留原 target／band，以獨立來源鎖補充，Pose weight 0.04 通過；其他組未需要補充恢復。詳見 [重啟](2026-11-09-actshape-reboot-recovery.md)、[CUDA](2026-11-09-actshape-cuda-segv.md) 與 [校準](2026-11-10-actshape-calibration-recovery.md) 紀錄。沒有增加 100E 上限。

## 未解事項與界線

CUDA 原生崩潰根因尚未確認；完成這輪訓練不是驅動已修復的證明。硬體型號與資源限制尚未指定，MAC／容量估算、GPU replay 與 PyTorch 時間均不等於 RTL／HLS 或板端效能。Conv、QKV、P.V、projection 及 head／decode 仍有浮點路徑，未宣稱整網整數化。沒有實作固定輸入下另改內部 stride 的架構變體。本輪單 seed、validation 選型，未增加原尺寸 control 加訓。

資料、標註、BBAT5 v1 assignment 與唯讀 `yolo_optimize` 沒有變更。本階段已停止，未啟動下一階段，也未 commit／push 或發布模型。交付過程一次 GPU 查詢見外部 `/home/uxin/mambapose/.venv/bin/python` 程序（PID 82200、約 2736 MiB），最後查詢已無計算程序；本研究未操作該外部工作。「已釋放 GPU」指本研究沒有再使用 GPU，外部使用狀態可能另有變化。
