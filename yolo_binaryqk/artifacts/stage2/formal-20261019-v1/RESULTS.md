# 第 2 階段完成，等待檢視

S1／S2 各從同一 B2 parent 獨立執行 5 epochs；沒有 baseline/control 加訓。

| 候選 | Best epoch | 官方 AP50–95 | internal AP50–95 | 相對 B2 差值 |
| --- | ---: | ---: | ---: | ---: |
| S1 | 5 | 0.514152 | 0.507067 | +0.004263 |
| S2 | 5 | 0.513711 | 0.506395 | +0.003822 |

依事前規則推薦 S1，不預設二次冪勝出；第 3 階段未開始。
Best 包含 epoch 0；若 best 為 0，表示該候選的適應訓練沒有改善初始替換結果。

兩組均為 16 個 scale、1008 個 signed16 bias entries、最大 H/W32；Q10 score 累加需 signed18，reference 用 int32。
每張640×640：2.56M次尺度計算，S1為整數乘法、S2為固定shift；共同bias表合計2016 bytes，未存完整NxN bias表。
未打包二值reference、QKV/PV/projection及native Softmax仍保留；不宣稱整網定點或實測加速。
完整八項AP、參數、檔案大小、訓練時間及顯存見 comparison.csv；部署整數常數見各候選 deployment-constants.json。
載入必須依manifest重建，不得退回Stage1固定浮點scale；下一階段parent與SHA見 handoff.json。
限制：單seed、沒有未改方法加訓control、bias從零開始；只有共同五輪預算，未做硬體綜合或板端量測。

## 完成核對與選型解讀

2026-10-19 04:58（Asia/Taipei）完成，總耗時 15799.84 秒，約 4 小時 23 分鐘。兩組每輪均完整訓練 118287 張、463 次 optimizer 更新，epoch 0 至 5 各完整評估 5000 張，AMP retries 全部為 0。沒有追加訓練，目前 GPU 已釋放。

S1 較 B2 官方整體 AP 提高 **0.4263 個百分點**，較 S2 高 **0.0441 個百分點**。這是單 seed 的觀察結果，不能宣稱差距具有統計顯著性；推薦 S1 是依事前整體 AP 規則，S2 的 shift 成本優勢仍保留為比較。

| 模型 | Person AP50–95 | Ball AP50–95 | Bat AP50–95 |
| --- | ---: | ---: | ---: |
| B2 parent | 0.639523 | 0.514560 | 0.487651 |
| S1 Q10 | 0.643376 | 0.512632 | 0.485961 |
| S2 二次冪 | 0.644395 | 0.512593 | 0.473084 |

兩組 ball／bat 皆未超過 B2。S1 的 ball／bat 分別下降 0.1928／0.1689 個百分點，S2 的 bat 下降 1.4567 個百分點，不能把整體提升解讀成所有類別改善。S1 距原始 native parent 官方 AP 0.524686 仍低約 1.0534 個百分點；internal AP 不混用。沒有 control 加訓，無法單獨分離訓練、scale 與 bias 的收益。

實際最佳模型均為 21,897,272 parameters，比 B2 增加 1024 個；checkpoint 約 88.08 MB。兩組訓練的 peak allocated memory 均為 6315927040 bytes，這是軟體訓練量測。每張影像的乘法／shift 次數及 2016-byte bias 表是解析成本，不是硬體速度。

交付權重為 `S1/epoch-05.pt`，SHA `9cacbd802f089a6e0aca87c2e0d1743d74458e7c030029be3076dd3090b48dea`；固定常數見 [deployment-constants.json](S1/deployment-constants.json)，完整契約見 [handoff.json](handoff.json)，核對證據見 [handoff-verification.json](handoff-verification.json)。切換第 3 階段前，依使用者要求整理成 `result/stage2/`，由 `yolo_attention` 接手；目前沒有切換階段。
