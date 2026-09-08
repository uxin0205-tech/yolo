# 量化階段收尾：已完成結果與延後事項

2026-09-08，依使用者及老師的新決定：完成當前有限 PTQ 批次後收尾，量化延後至模型最後階段。**六組 5-epoch QAT 與十二個累積 PTQ 候選均已完成；不啟動原先考慮的額外兩組 QAT，不做新的 formal／部署測試。這是階段預研成果，不是全模型整數部署完成。**

本輪 supervisor 已自然退出（exit code 0），600 秒 shell monitor 收到終點事件後退出，未中斷正在運算的 GPU child。原始狀態 `decision_required` 保留作執行證據；新的[交接決策](../../artifacts/queues/full-model-cumulative-0908/phase-hold.json)明定量化延後，不能把原計畫名額當作繼續訓練的授權。

## 1. 已經做了多少，而不是還要訓練多少次

| 證據 | 已完成 | 能回答什麼／不能回答什麼 |
| --- | ---: | --- |
| 單層輸出 sensitivity | 148 層 × 4 格式＝592 probes | 同 parent 的小樣本輸出變化；不是 592 次訓練，也不是逐層 mAP |
| 十區隔離 PTQ | 10 區 × 4 格式＝40 候選 | 各大區替換的任務精度；不訓練 |
| 特殊格式短 QAT | 6 組 × 5 epochs＝30 組內回合 | 尺度學習、三元格式與有限恢復試驗；不是一個模型訓練 30 epochs |
| 逐區累積 PTQ | 11 階段、12 候選 | 已接受路由加下一區的實際交互影響；不訓練 |
| 額外累積 QAT、finalists、formal、整數部署 | 未做，延後 | 不能由以上搜尋數字直接宣稱完成 |

之前的權重分布與輸出誤差用來縮小候選，不丟棄重做。仍需累積驗證，是因為「各區單獨合格」不保證組合後合格。本輪沒有逐層、逐格式窮舉 QAT。歷史 V35 的 1,332 筆權重重建誤差與 V36 這輪不同 parent，不納入同模型隔離 mAP 統計。

## 2. 控制條件與全模型覆蓋

- 架構：`yolo_combine/final/full35` 的 Detect＋Pose 模型；BinaryQK／MASF 保留。
- 共同 parent：V36 `v36-qsilu-a8-full-coverage-epoch1`，由[manifest](../../artifacts/queues/full-model-continuous-0907/parent-manifest.json)釘住 export 與對應 EMA full-resume。PTQ 都重新從此 export 建構，不拼接六個獨立 QAT 模型的權重。
- Activation：qSiLU，124 個 LSQ+ A8 輸出量化器。十二個實際 PTQ build 均核對為 124 個。
- Weight：148 個部署 Conv/Linear 路徑均有量化配置，十二個實際候選均確認覆蓋 148。未新增替換的路徑保留 parent 的 W8／LS-SD4／W6／W4 混合配置，不是退回全 FP32。
- 本輪仍使用 fake quant／浮點運算；保護算子、bias、殘差和算子間邊界不等於全部變成整數 kernel。148 路徑覆蓋不是「所有算子已整數部署」。
- BBAT5：不可變 `/home/uxin/yolo/original/pose/derived/bbat5-v1/`，registry `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。Pose／ball-bat Detect 使用各自正式 `configs/pose.yaml`／`configs/detect.yaml`，沿用既有 search assignment，未改影像、標註或 split；未使用 formal val 搜尋。
- COCO80／Person：`/home/uxin/yolo/coco2017.yaml`，沿用既有搜尋驗證契約；不是 BBAT5 二類 Detect YAML。
- 六組短 QAT：seed 1、AdamW、patience 5，Detect logical batch 128／microbatch 16、Pose 16，warmup 1、scale-only 1、不加新影像雜訊；不是額外訓練新 baseline。
- 總門檻：相對既有 accepted reference，8 項 mAP50 各下降 ≤1.5 pp、8 項 mAP50–95 各下降 ≤4 pp，包含 activation 與 COCO Person；不是只計相對 V36 的新增損失。門檻是上限，不是目標。

Accepted reference 為 `artifacts/runs/v4-qsilu-backbone-early-w8-search-v1/validation/accepted/epoch-0000/bittrue/metrics.json`，SHA-256 `2e97e4a60039741cc1fc45a03d67615bbc5b851bcf60a56cd153251366c4a355`。V36 export SHA-256 `8c1f3652fd21c0e38221cfc3acba7ee013c5e88df641b9331c33796f6f11b66c`；EMA full-resume SHA-256 `f83e1ba22adc737ed05abf68152f621a67915a2104ace3bfd5a3b630d66dcb78`。不能用另一個 checkpoint 的敏感度當成本輪結果。

## 3. 六組 QAT 留下的有效證據

下表統一顯示第 5 回合，下降單位為百分點（pp）；每列各取八項指標中的最差總下降。第 5 回合不一定是各組 `best_joint.pt` 選中的回合，不能把本表當作 checkpoint 選定結果。

| 組別 | 最差 mAP50 下降 | 最差 mAP50–95 下降 | 第 5 回合雙門檻 |
| --- | ---: | ---: | --- |
| Pose Fixed-SD4 | 0.834 | 1.008 | 通過 |
| Pose LS-SD4 | 0.896 | 1.043 | 通過 |
| MASF exact ternary | 0.949 | 1.069 | 通過 |
| MASF Paper-TWN | 0.874 | 1.064 | 通過 |
| MASF filterwise TWN | 0.826 | 0.981 | 通過 |
| Detect predictor LS-SD4 recovery | 1.016 | 1.147 | 通過 |

Pose 的 Fixed／LS-SD4 同 parent、同 24 路徑，分別固定／學習所選權重尺度，末回合各任務互有優劣；不足以宣稱 LS-SD4 必然優於 Fixed。MASF 三種三元使用同三路，但格式、尺度粒度／metadata 成本有差異，不宜只用「都是三元」視為同硬體成本。

Detect predictor LS-SD4 在第 1 回合未通過，第 2–5 回合通過。第 5 回合的 COCO／Person mAP50–95 相對其匹配 PTQ 分別回升 **4.192／4.461 pp**，支持「部分 PTQ 失敗可由短 QAT 恢復」，不支持「所有 PTQ 失敗都可恢復」。沒有新增同 parent sham，不能把所有改善單獨歸因於尺度學習。

完整 30 回合與 16 指標見[六組結果](2026-09-07-continuous-qat-recovery-results.md)及[原始摘要](../../artifacts/queues/full-model-continuous-0907/qat-recovery-summary.json)。分布只能作提名依據；「接近零／越分散」不是足以決定 SD4／三元任務精度的單一規則，判讀見[方法說明](2026-09-08-quantization-method-and-decisions.md)。

## 4. 累積 PTQ 的全部結果

每一步是在當時已接受前綴上追加；未接受者不進入下一步。以下不是十二個相同隔離配置的格式排行榜。W6、W5 是從同一個特殊格式前綴分叉的兩個替代配置。所有數字是搜尋驗證，不是 formal。

| 追加候選 | 最差 mAP50 總下降（pp） | 最差 mAP50–95 總下降（pp） | 該候選 codes＋scales（bytes） | 累積接受 |
| --- | ---: | ---: | ---: | --- |
| Backbone early：SD4 | 1.360 | 1.432 | 19,995,456 | 是 |
| Backbone deep：SD4 | 1.584 | 1.582 | 18,815,808 | 否 |
| Backbone attention safe：SD4 | 2.704 | 4.110 | 19,864,384 | 否 |
| Neck：SD4 | 1.405 | 1.523 | 19,405,632 | 是 |
| Neck attention safe：SD4 | 1.866 | 2.222 | 19,241,792 | 否 |
| MASF：filterwise TWN | 1.403 | 1.553 | 19,354,304 | 是 |
| Detect tower：SD4 | 1.892 | 2.036 | 19,288,768 | 否 |
| Pose tower：SD4 | 1.403 | 2.954 | 18,382,528 | 是 |
| Detect predictor：SD4 | 2.135 | 5.617 | 18,361,664 | 否 |
| Pose predictor：Paper-TWN | 1.403 | 5.703 | 18,382,140 | 否 |
| 其餘指定 backbone／neck 路徑：W6 | 8.637 | 17.183 | 15,341,840 | 否 |
| 相同前綴、相應剩餘路徑：W5 | 43.061 | 43.782 | 13,821,496 | 否 |

W6／W5 的這種廣泛追加 PTQ 明顯不合格，不代表每個單層 W6／W5 都不行。Pose predictor 的 Paper-TWN 幾乎不影響最差 mAP50，卻使最差 mAP50–95 下降 5.703 pp，是不能只看 mAP50 的具體負結果。Backbone deep 只略超 mAP50 上限，原可作有限 QAT 恢復候選；現在依使用者決定延後，不自行追加。

## 5. 保留兩個取捨點，不宣布 final winner

Parent 的 codes＋scales 估算為 20,290,368 bytes。以下保留的是可重建的配置與搜尋結果，不是已完成 formal 的新部署 checkpoint。

| 配置 | 新增路由 | 最差總下降 mAP50／mAP50–95（pp） | codes＋scales | 較 V36 估算減少 |
| --- | --- | ---: | ---: | ---: |
| 較保守 | early SD4＋neck SD4＋MASF TWN | 1.403／1.553 | 19,354,304 bytes | 4.61% |
| 較壓縮 | 上述配置＋Pose tower SD4 | 1.403／2.954 | 18,382,528 bytes | 9.40% |

較壓縮配置在 parent 上新增 29 個路由指定（26 個 Fixed-SD4、3 個 filterwise TWN），**不是只有 29 層量化**，其餘 119 路徑保留已量化 parent。精確路徑與各步前綴見[累積決策原始 JSON](../../artifacts/queues/full-model-cumulative-0908/cumulative-selection.json)。

較壓縮配置的完整 16 指標如下。mAP 為百分比；Δ 為百分點，正值是增加，負值是下降。Δ accepted 用來判總門檻，Δ V36 分開顯示本次累積替換的增量。

| 任務 | mAP50（%） | Δ accepted | Δ V36 | mAP50–95（%） | Δ accepted | Δ V36 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO box | 65.669 | -1.403 | -0.568 | 48.249 | -1.553 | -0.396 |
| COCO Person box | 83.278 | -0.663 | -0.163 | 60.945 | -1.093 | -0.228 |
| BBAT box | 97.773 | +0.150 | -0.425 | 80.770 | -2.874 | -2.272 |
| BBAT pose | 97.775 | +0.096 | -0.432 | 97.744 | +0.066 | -0.439 |
| BBAT ball box | 96.449 | +0.611 | -0.716 | 74.950 | -2.954 | -2.303 |
| BBAT bat box | 99.096 | -0.312 | -0.134 | 86.590 | -2.795 | -2.242 |
| BBAT ball pose | 96.453 | +0.504 | -0.713 | 96.413 | +0.464 | -0.735 |
| BBAT bat pose | 99.096 | -0.312 | -0.151 | 99.075 | -0.333 | -0.142 |

加入 Pose tower 後雖仍過門檻，BBAT box mAP50–95 有明顯代價，因此目前同時保留較保守點，避免把「成本最低且過門檻」誤稱「整體最好」。本輪單 seed／搜尋驗證，未量測不同 seed 的不確定性。

成本公式為 `inherited_unmodified_bytes + weight_code_bytes + scale_bytes`。減少比例為 `1 - candidate_bytes / parent_bytes`。未含 bias、protected、alignment、kernel 或額外部署開銷；**不是實際模型檔案縮小 9.40%，也不是速度提升 9.40%**。

## 6. 封存內容與日後如何接回

保留 V36 parent／export／EMA、六組 QAT 的 checkpoint 與逐回合結果、40 PTQ、592 probe、12 累積 PTQ 原始報告、generated 路由與所有負結果。本次沒有刪檔、換主線 checkpoint、回復 FP32、重切資料或發布 GitHub。

暫停項目：額外累積 QAT、逐層 mAP 擴展、更多位元／activation 配置、finalist 選定、final export 重驗、formal、純整數部署與硬體測量。既有 QAT 名額不代表仍有 live queue。

日後使用者明確要求恢復時，先確認最終架構、activation 與 checkpoint：

1. 若 parent 不變，可沿用本輪篩選證據，先比較保留的兩個取捨點，再決定是否做少量短 QAT；不是全部重跑。
2. 若最終模型已變，需鎖定新 parent 並重驗量化覆蓋及基準。舊格式排序只作先驗，不能把目前 mAP 或最優路由直接移植成新模型的已驗證結論。
3. 最終再確認精度／成本偏好與 GPU 授權，才安排短 QAT、export、formal 和真實部署邊界。

## 7. 收尾驗證

- 舊六組：43 個來源 pin（含 parent／checkpoint）hash 通過；30 回合、480 個 mAP 值有限且範圍有效，六組 completion 的 `epochs_completed` 都是 5。
- 新累積：11 份完成 stage report、plan／accepted reference 的 13 個 hash 通過；另核對 parent manifest 及 12 份不同原始 metrics report 的 hash。
- 十二個候選的 192 項總差值由 accepted 原始值獨立重算一致，所有 build 的 parent 相同、148 個部署權重量化、124 個 activation quantizers。
- 本次 CPU 回歸：`tests/test_cumulative_ptq_queue.py`，10 passed；先前本版完整 CPU suite 為 351 passed，不宣稱本次重新跑過全 suite。
- 沒有新的 GPU 訓練、formal 或硬體 benchmark。收尾文件及機讀交接的最終檢查見[工作紀錄](../worklogs/2026-09-08-quantization-phase-handoff.md)。
