# 第七階段結果與交接：Activation／qSiLU

**Stage7 已於 2026-11-03 01:00:29（Asia/Taipei）完成；後續 Stage8／9 已完成，現行最終成果見 [統一交接](../../../result/README.md)。** 四候選各完成 10 輪，共 40 輪；四份最佳模型皆重載後重驗完整 AP，十項指標與選定時完全相同。

可直接使用的模型在 [handoff/](handoff/README.md)。**qSiLU E8 是下一階段唯一共同 parent**；PolyQuality E10 是本輪綜合精度最佳的比較模型。Hardswish／PolyShift 的最佳 checkpoint 與完整數據保留於 `candidates/`，各只有一份實體權重。根層 `handoff` 與舊路徑是相容連結。

## 精度與選型

所有候選由同一 Stage6 E1 獨立出發，未新增 SiLU control 加訓。資料為完整 COCO val 5,000、BBAT5 v1 val 683，FP32 E2E、imgsz=640、rect validation、batch8、conf0.001、iou0.7、max_det300。BBAT5 v1 assignment／來源／標註未改。

| AP50–95 | Stage6 E1 | qSiLU E8 | Hardswish E8 | PolyShift E10 | PolyQuality E10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| COCO AP | 0.512676 | 0.513951 | 0.498474 | 0.514996 | 0.515610 |
| COCO person | 0.641255 | 0.643234 | 0.629305 | 0.643457 | 0.643024 |
| COCO sports ball | 0.515161 | 0.514662 | 0.495165 | 0.514859 | 0.515465 |
| COCO baseball bat | 0.470841 | 0.469036 | 0.426266 | 0.472367 | 0.466103 |
| BBAT box | 0.609202 | 0.613904 | 0.609630 | 0.613729 | 0.616505 |
| BBAT Pose | 0.899850 | 0.900762 | 0.905294 | 0.901011 | 0.903934 |
| BBAT ball box | 0.493883 | 0.497430 | 0.504761 | 0.495755 | 0.500225 |
| BBAT ball Pose | 0.863474 | 0.864731 | 0.878103 | 0.863558 | 0.867117 |
| BBAT bat box | 0.724521 | 0.730378 | 0.714499 | 0.731704 | 0.732786 |
| BBAT bat Pose | 0.936227 | 0.936792 | 0.932484 | 0.938464 | 0.940751 |

相對門檻是每項 AP 不低於 E1 超過 0.015；通過後，以 `0.5 × (COCO 四項平均 + BBAT 六項平均)` 選型。qSiLU、PolyShift、PolyQuality 通過；Hardswish 在 COCO ball／bat 等指標退化，未通過。PolyQuality 的綜合分數最高，不代表每項都最高，例如 Hardswish 的 Pose AP 更高。

qSiLU 相較 E1，COCO／BBAT box／Pose 分別提高約 0.128／0.470／0.091 個百分點；COCO sports ball／bat 仍略低。qSiLU 已完成必要適應，不觸發額外 5 輪。它保留硬體友善分段二次式與指定的 RepConv 接續角色；不因另一 activation 較高分而換掉主線。原獨立模型的四項融合目標仍未全部達成，不能把 activation 相對門檻混稱為原目標通過。

## 訓練與來源

動手前唯讀核對 optimize activation bridge 的實際程式、10 輪 run、parent、結果與上游 J3；PolyQuality 另參考原 Full35 完整配方與結果。來源與差異見 [reference-review.json](reference-review.json)、[protocol.json](protocol.json)。

共同配方：AdamW，warmup1、cosine final0.5；backbone／neck／heads LR 分別 3.8e-7／1.9e-6／5e-6，Attention 固定。Detect physical16 × accumulate16、Pose physical16、reference64。shared BN running 固定、affine 可訓練，head BN 可更新；各候選 fresh optimizer／EMA，Pose weight 以 train-only 梯度校準。每組先做完整零樣本 AP、梯度校準與兩個 AMP smoke updates，正式訓練重新載入共同起點。實際 pose weight、EMA 與 lineage 存在每個 checkpoint manifest。

## 敏感度與硬體契約

qSiLU／PolyQuality 各排 6 個區域驗證；其中 `attention` 沒有 activation 可替換，兩筆結果恰等於 E1，已標示為 **0 sites／no replacement**。有效替換比較共 10 筆。qSiLU 單換 neck 時 BBAT ball Pose 約下降 1.085 個百分點，是此次較敏感項；完整替換經恢復後已通過門檻，因此本輪不另展開 selective 訓練矩陣。163 個部署位置、24 個 RealNVP loss 專用位置與共享 alias 分開記錄。

B2 sign＋Hadamard 二值 QK、兩處四 heads、16 個 S1 Q10 scale、signed16/1024 分解式位置 bias 均保留。D1 為整數 PWL exp `[-10,0]`／20 段＋LUT256 reciprocal，UQ1.16 倒數、uint26 分母、UQ1.24 probability，max_tokens1024；沒有最後浮點除法重正規化。

qSiLU 選定權重的兩處 Attention 前 16 次完整 val forward 診斷：最大 row-sum 誤差約 0.001946／0.001949，尾端截斷比例約 0.668%／2.804%，固定 normalization 契約通過。這是指定採樣範圍的診斷，不是所有資料的極值。完整來源見 [runtime-diagnostics](evidence/activation-20261102/verify-qsilu_pq/evaluation/runtime-diagnostics.json)。實際 val 已有長方形張量；後續長方形／二次冪尺寸比較已於第九階段完成 120E，詳見根層總報告。

| Activation | 資料相乘 proxy | 固定係數與額外處理 |
| --- | ---: | --- |
| qSiLU | 1 次共用平方 | 二次冪分段點、dyadic 係數，可用移位加法；另有分段選擇 |
| Hardswish | 1 次 | clamp 與固定 1/6，需規定定點近似／捨入 |
| PolyShift | 4 次產生資料冪 | 固定係數可用移位加法，資料乘法仍存在 |
| PolyQuality | 4 次產生資料冪 | 一般固定係數乘法，量化與資源通常較複雜 |

以上是實作的運算 proxy，不等於 DSP 數量或實測加速。四者成本需在相同位寬、吞吐與資源共用條件下比較。Conv、QKV、P·V、projection 與 activation 仍為浮點任務運算；數值報告的 qSiLU／PolyShift Q10 emulator 是局部 reference。尚未做 RTL／HLS／板端驗證。

## 檔案、驗證與清理

- [handoff](handoff/README.md)：兩模型、完整設定、載入程式與凍結 runtime，可不讀舊研究目錄推論。CPU 160／640 各 24 個輸出 tensors 與原模型逐值相同；routing、錯誤 activation 與常數拒絕均通過，見 [交付驗證](handoff/validation/result.json)。
- [comparison.csv](comparison.csv)、[curves.json](curves.json)、[sensitivity.json](sensitivity.json)：零樣本、選定十項 AP、40 輪曲線與區域數據。
- [comparison.png](comparison.png)、[sensitivity.png](sensitivity.png)：現行圖表；執行 `.venv/bin/python yolo_activation/result/stage7/redraw.py` 可只用小型數據重畫，不需 GPU／checkpoint。
- [evidence](evidence/activation-20261102/)：完整本輪小型配置、逐輪／驗證結果、監測與不可變 source lock。370 個實體檔案搬移前後 SHA 一致，舊 queue 路徑只留相容連結。
- [cleanup-20261103.json](cleanup-20261103.json)：本次刪除 810,934 個 runtime 連結、4,844 個檔案等，移除配置空間 5.2484 GiB；有效內容約 2.1547 GiB，差異主要來自大量連結占用的區塊。另移除 721 個空資料夾。48 個受保護檔案 SHA 一致，無 archive／backup。
- [舊 Full35 小型證據](../reference-full35/README.md)：已抽取舊指標與設定；外部量化仍使用的四份舊權重留原路徑。本次不更動 optimize、正式資料或其他研究階段。

Stage7 開始前曾另清理 26.137 GiB；queue 完成時另刪除四份 optimizer resume。本次 5.2484 GiB 不重複計入那兩批。qSiLU checkpoint SHA 為 `2f91ec1862ad62304f1e2727e9b9ee64a27c5a46a45137f83a74dc7f14c46a69`；完整交接記錄在 [handoff.json](handoff.json)。沒有精確 optimizer resume，但可從選定權重開始新訓練段。

單 seed、原融合目標缺口、未做整網定點／硬體量測與新機安裝測試等限制仍保留。Stage7 本身未進行 RepConv；後續 Stage8 已執行並交付，現行選擇以根層 result 為準。
