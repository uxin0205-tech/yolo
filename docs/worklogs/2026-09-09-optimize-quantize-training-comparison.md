# 2026-09-09：Optimize 與 Quantize 訓練方法及耗時比較

## 任務與範圍

使用者希望了解 `yolo_optimize/` 與 `yolo_quantize/` 接近 5 倍的訓練時間差，並判斷哪種方法適合目前研究。只讀取既有程式、設定及訓練紀錄；新增本分析與計時附件，未修改訓練程式、權重、資料、split，未啟動或停止 GPU 工作，也未使用子代理。

主比較為融合後 Detect＋Pose：Optimize 的 `native-parent-ema-control-adopted`，以及 Quantize 的 `continuous-0907-special-v1` 六組 QAT。另列 9 月 9 日融合前純 Detect 研究，避免把整個目錄誤認成只有一套方法。

本次採 diagnosing-bugs 技能的「先量測、再核對原因」流程；任務是既有方法比較，不是重現並修復功能錯誤，因此使用歷史日誌重算建立基準，不新增 GPU 重現或回歸測試。

## 結論與實測

時間差確實存在，而且主要發生在訓練步驟。融合模型穩態更新單位約相差 **5.11 倍**；加上驗證與邊界作業，每 epoch 約相差 **4.37 倍**。這是特定實作與訓練範圍的比較，不能推廣為所有 QAT 一定慢 5 倍。

以下均以 E2–E5 計算，避開首回合初始化及 scale-only 階段；QAT 共 24 個回合，Optimize 共 4 個回合。

| 項目 | Optimize 原生恢復 | Quantize 六組 QAT |
| --- | ---: | ---: |
| 每回合 COCO 訓練影像 | 118,287 | 118,287 |
| 每回合 optimizer 更新單位（macro） | 463 | 463 |
| 正常完整 macro 的 Detect／Pose 影像 | 256／16 | 256／16 |
| Detect 實體 microbatch | 32 | 16 |
| 完整 macro 的 Detect 前向／反向次數 | 8 | 16 |
| 完整 macro 的 Pose 前向／反向次數 | 1 | 1 |
| 每回合 Detect 實體批次數 | 3,697 | 7,393 |
| 每回合 Pose 影像呈現次數，含循環 | 7,404 | 7,396 |
| 每回合 macro 時間中位數的平均 | 1.162 秒 | 5.943 秒 |
| 每回合耗時平均，含驗證與部分邊界作業 | 651.45 秒＝10.86 分鐘 | 2,845.09 秒＝47.42 分鐘 |
| 每回合驗證時間平均 | 104.77 秒 | 88.60 秒 |
| 輸入尺寸／AMP | 640／FP16 AMP | 640／FP16 AMP |
| Optimizer／betas／weight decay | AdamW／0.948, 0.999／0.00027 | 相同 |

QAT 六組 E2–E5 的回合耗時範圍為 46.95–47.84 分鐘，並非只挑到一組偶然較慢的紀錄。Optimize 同期為 10.82–10.91 分鐘。

時間口徑：macro 統一使用同 epoch 相鄰 macro 日誌時間差的中位數。Optimize 另有 GPU synchronize 後的 engine 計時，四回合總訓練 engine 平均約 540 秒，但 QAT 沒有相同欄位，不假造對稱數字。Optimize epoch 為 `epoch_start` 至 `epoch_complete`；QAT 為相鄰 validation 完成時間，包含前一回合後續存檔等邊界成本，屬近似完整回合比較。QAT E1 起點缺失，CSV 留空。Optimize E1 採用了已完成的診斷前綴，延續程序約 44 分鐘只實際執行 E2–E5，不能當成完整五回合總時間。

## 時間差來源：已核對的機制與尚未量化的貢獻

### 1. QAT 多了整張運算圖的量化模擬

QAT 的 `qat-graph.json` 記錄 124 個 activation quantizer、148 個部署權重路徑。這些是全圖配置數，兩個任務各次 forward 實際經過的 head 路徑不同，不把所有配置數當成每次 forward 呼叫數。

[權重量化器](../../yolo_quantize/src/yolo_quantize/qat_weights.py) 的 `QATConv2d.forward()` 每次卷積先對浮點 master weight 做 fake quantization，再交给浮點卷積。均勻量化包含除尺度、round、clamp 與還原尺度；SD4 目前透過 15 個 codebook 值的逐權重距離與 argmin 尋找最近值。這些臨時張量與額外運算不會因部署格式名稱是 W8／W4 就自動消失。

[LSQ+ activation](../../yolo_quantize/src/yolo_quantize/quantizers.py) 同樣逐層執行尺度／offset、round、clamp 與反向梯度。因此這套 QAT 是用浮點運算學習量化後的誤差，不是已經用 INT8 卷積加速訓練。PyTorch 的 [QAT 官方說明](https://pytorch.org/blog/quantization-aware-training/) 也明確區分 fake quantization 與訓練後的實際量化轉換。官方文章的 LLM 測速不拿來估算本專案 YOLO 的倍率。

### 2. Optimize 更新的網路範圍較小

[Optimize 訓練器](../../yolo_optimize/src/yolo_optimize/training.py) 的本次 native scope 固定 backbone、MASF 與 attention；主要更新 neck、Detect head、Pose head。實际 `apply_stage()` 對 LR＝0 的範圍執行 `requires_grad_(False)`，不是只把 optimizer 的更新量設為零。

QAT 設定 `backbone_start_layer=0`、`tune_attention=True`，E2 起開放 backbone、neck、MASF、attention 與雙 head 的合法參數，另學習量化參數；硬體契約保護參數仍保持固定。代表 `pose-ls-sd4` 的控制紀錄為 499 個模型參數張量及 396 個量化參數張量可訓練，這是張量數，不是 895 個純量參數，也不等於所有張量每步都收到非零梯度。

QAT E1 雖為 scale-only，量化尺度位於網路不同深度，仍需通過相關運算反傳；權重 blend ratio＝0 時，目前 forward 也先計算量化結果再乘 ratio。因此 E1 並非幾乎免費。關於凍結與計算圖，參考 [PyTorch Autograd mechanics](https://docs.pytorch.org/docs/main/notes/autograd.html)：凍結某層參數不代表任何情境下都能略過該層的輸入梯度。

### 3. 相同資料被切成更多小批次

兩邊都標示 Detect logical batch128，但融合訓練的完整 macro 是 **兩份 logical Detect batch＋一份 Pose batch**，即 256＋16 張影像後更新一次。

Optimize 以 32 張為一批，做 8 次 Detect forward/backward；QAT 以 16 張為一批，做 16 次。總影像數相同，但 Python／kernel 啟動、逐次權重量化及其他每批固定成本會重複更多次。這不等於單靠 batch 就能解釋 2 倍，更不能乘上臆測的另一個倍率湊成 5 倍。

既有 Optimize batch benchmark 的 physical32 與 physical64 速度接近，並不支持「batch 翻倍一定快一倍」。QAT physical16→32 沒有相同方法、相同 scope 的配對測速證據；是否能承受顯存與實際收益都未驗證。

### 4. 量化實作還有額外同步與重複計算

[Activation adapter](../../yolo_quantize/src/yolo_quantize/activation_adapter.py) 的 `mode` 每次透過 `_mode_code.item()` 取得 Python 值；fake-quant 路徑會讀取兩次 mode。在 CUDA 圖上，這會引入 CPU 取得 GPU scalar 的同步點。權重 `_channel_scales()` 也多次存取會重新執行 softplus 的 `scale` property；SD4 使用展開的 codebook 距離。

這些是明確的效能檢查候選，不是 QAT 數學定義要求的全部必要成本。[PyTorch 效能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) 建議避免不必要的 CPU／GPU 同步，並說明 pointwise 運算的記憶體與 kernel 啟動成本。本次沒有 profiler 或單變因 GPU 實驗，不能宣稱上述任一項占了幾成，也沒有直接修改量化程式。

## 已排除或需避免的誤解

- 不是 COCO 只用五分之一：兩邊每回合均為完整 118,287 張，宏更新次數也都是 463。
- 不是未開 AMP，也不是 AdamW 對 MuSGD：這些代表 run 都是 FP16 AMP＋AdamW，betas 與 decay 相同；學習率不同，但較小學習率本身不會讓同一運算突然慢 5 倍。
- 不是驗證次數造成主要差距：Optimize 每回合驗證 EMA Float、EMA BitTrue、live BitTrue；QAT 驗證 EMA BitTrue。Optimize 驗證總時間反而約多 16 秒，與約 2,194 秒的完整回合差距方向相反。
- AMP 重試有首回合與個別後續回合事件，但六組 E2 全部零重試，仍穩定約 5.9 秒／macro，所以重試不足以解釋持續時間差。
- 不能把全部 PTQ 搜尋＋六組 QAT 的 queue 總時間，當作單一模型的訓練時間。

## 今天新增的融合前 Detect 分流

`studies/pre-fusion-full35-b100/` 是另一套任務：完整 COCO Detect、physical32×4＝128 張更新一次，沒有 Pose 訓練；主要開放 Q/K projection 與 Detect head，對照 arm 的 Q/K 梯度還可能為零。

已完成 E1 的原始 `elapsed_seconds` 在 EMA 與 live 驗證後寫入，包含兩次驗證：

| Run | E1 時間 | COCO 影像／更新次數 |
| --- | ---: | ---: |
| `a0-control-recovery-v1` | 495.70 秒＝8.26 分鐘 | 118,287／925 |
| `a0-qk-recovery-v1` | 577.01 秒＝9.62 分鐘 | 118,287／925 |

若使用者看到的是這個新分流與約 47.42 分鐘 QAT 的差距，倍率分別約 5.74／4.93。除了量化、訓練範圍與 microbatch，這裡還少了 Pose 任務；不能直接用 925 對 463 的更新數推論前者工作量更多。兩組目前可引用的是完成 E1，不能由單一回合預測長訓收斂。

## 哪種方法比較適合

目前若主要目標是架構、Detect／Pose 精度與訓練策略，建議先沿 Optimize 的用途完成模型與正式評估；模型固定、確定硬體與量化格式後，先看 PTQ 是否足夠，必要時再做 QAT 恢復量化損失。這與既有「量化延後至模型最後階段」的專案決策一致。

QAT 花更多時間是為了在低位元限制下保留精度，不表示量化前精度必然更好。反過來，Optimize 較快也不表示其候選一定更準：融合後第一輪沒有新精度候選通過驗收，仍保留原 J3 BEST。Quantize 的 Detect predictor LS-SD4 案例，相對自己的 PTQ 在 E5 恢復 COCO／Person mAP50–95 約 4.192／4.461 個百分點，說明其量化恢復用途有實證，但不是對 Optimize 的勝利。

兩邊 parent 與 BBAT5 評估範圍不同：Optimize 用正式 Pose train5,964／val683；QAT 用既有 search-train5,364／search-val600，屬同一 canonical bbat5-v1 的既有搜尋配置，本次沒有重切資料。兩者訓練迴圈會循環 Pose，故每 epoch 實際呈現約 7,400 張，而非剛好一次 Pose dataset。QAT 棒球 AP 數字不能直接與正式 val683 的 AP 比高低。

可靠的方法比較需要同 parent、同資料與驗證範圍、同訓練 scope／精度政策、同更新預算及對齊的硬體環境；部署好壞還需要相同目標硬體上的延遲、記憶體、模型大小及精度。這輪量化仍是模擬與搜尋證據，未完成純整數部署，沒有實測速度保證。

## 驗證方式與結果

執行：

```bash
python3 docs/worklogs/assets/2026-09-09-training-method-comparison/recalculate.py
```

通過既有紀錄的資料量與 macro 數一致性檢查：34 個融合訓練 epoch，7 組 run，另整理 2 組新 Detect 的完成 E1。主比較使用 28 個 E2–E5 回合。保存 9 份來源日誌的 SHA-256、[逐回合 CSV](assets/2026-09-09-training-method-comparison/epoch-timings.csv) 與[統計 JSON](assets/2026-09-09-training-method-comparison/timing-summary.json)。原始檔完整路徑見 JSON。執行只用 Python 標準函式庫，不 import Torch，不產生新模型評分。

同步更新本層工作紀錄索引；根 README 已持續提供資料集規範與工作紀錄入口，因此不重複修改。

## 困難與解法

沙箱啟動時出現 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`，改用通過審核的命令讀取本機檔案及新增工作紀錄；無自動審核拒絕。系統沒有 `python` 別名，改用 `python3`。兩套日誌的時間欄位不同，明確區分 engine 計時、相鄰 macro 間隔與 epoch 邊界，不把缺失時間補成精確數字。

## 未解事項或風險

沒有逐算子 profiler、匹配的 QAT 關閉量化對照、physical16／32 配對測速或完整歷史 GPU 時脈／共享負載遙測，因此可以確認差距與程式機制，但無法分配各因素的精確因果貢獻。未啟動新實驗、未變更訓練方法；後續若改善 QAT 效能，仍需保持量化數值與梯度契約並驗證結果。
