# Stage8 原生 RepConv 訓練與硬體部署結果

**使用者最終採用：`rep17_20` E1，訓練與部署皆 qSiLU。** 預設交接模型已更新，選擇、SHA 與十項 AP 見 [selection.json](selection.json)。下方原自動推薦與 AP 保留為實驗事實；目前採用以使用者選擇為準。

訓練使用 Float PWL，四組保留 qSiLU，一組僅 Rep 使用 SiLU；表中 AP 全為折疊且轉回 qSiLU／D1 的部署結果。兩種算子的 AP 差距記錄於 handoff.json。

原自動研究建議：`rep17_20_silu`。目前可用且最終選定為頁首的 `rep17_20`；未採用模型的權重已依使用者要求刪除，表格保留歷史指標。

原訓練前採用規則及 gate 結果保留；使用者最終選擇另記於 selection.json。

| 模型 | COCO AP | 球 AP | 球棒 AP | Pose box AP | Pose AP | 輪數 |
|---|---:|---:|---:|---:|---:|---:|
| parent | 0.513951 | 0.514662 | 0.469036 | 0.613904 | 0.900762 | 0 |
| previous_rep17_20 | 0.513738 | 0.515436 | 0.469469 | 0.613471 | 0.901826 | 0 |
| rep17 | 0.514681 | 0.516807 | 0.474786 | 0.613637 | 0.900236 | 5 |
| rep20 | 0.514086 | 0.514966 | 0.466321 | 0.614905 | 0.899160 | 5 |
| rep17_20 | 0.514347 | 0.515782 | 0.469576 | 0.614371 | 0.901389 | 5 |
| rep17_20_silu | 0.513980 | 0.515215 | 0.475120 | 0.614043 | 0.901161 | 8 |
| acb17_20 | 0.514244 | 0.516787 | 0.473716 | 0.615579 | 0.901154 | 5 |

完整十項 AP 見 [metrics.csv](metrics.csv)，完整指標／血緣見 [handoff.json](handoff.json)。

五組從同一份 qSiLU E8 獨立開始；保留 B2 sign＋Hadamard、S1 Q10／signed16 bias、D1 整數 PWL exp＋LUT256 reciprocal。Rep／ACB 部署都是單 Conv3×3+bias → qSiLU，沒有分支或 BN；其他原生 ConvBN、head sigmoid 與浮點路徑詳見 evidence/verify-*/operators.json。

[可攜推論包](handoff/README.md) 已用獨立 CPU 程序比對 160／640 完整輸出。training/ 僅保留選定模型的最佳多分支 EMA；舊 optimizer resume 已清理，不能再使用 handoff.json 的歷史續訓路徑。該 JSON 的舊路徑是實驗血緣證據，不是目前檔案清單。清理見 [cleanup-final-20261108.json](cleanup-final-20261108.json)。

完整 D1 部署 AP 與 Float PWL 診斷分列；沒有將 Float 診斷路徑寫入部署模型。所有訓練均使用同一 BBAT5 v1，未重切資料。

可用 `python redraw.py` 重畫 accuracy／curves，不需 GPU 或 checkpoint。CPU smoke、GPU 訓練、完整 AP、定點 reference 與未執行的硬體綜合／實板量測分開看待。

限制：單 seed、validation 選型，無獨立 test 與硬體量測；本輪沒有未改方法的 control 加訓。第九階段已獲授權，在 [yolo_actshape](../../../yolo_actshape/README.md) 接續形狀分析與 GPU queue。
