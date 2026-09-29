# 全模型量化：方法、選擇依據與下一步

本日收尾：六組 QAT 與十二個[累積 PTQ](2026-09-08-cumulative-ptq-plan.md)候選均完成，依使用者與老師決定延後後續量化。下文保留方法與原規劃依據，不是新 GPU 啟動授權；最新結論與邊界見[階段收尾報告](2026-09-08-quantization-phase-handoff.md)。

本報告集中說明現行決策，不重新定義舊實驗。數字以[本輪結果](2026-09-07-continuous-qat-recovery-results.md)為準，排程以[現行計畫](../CURRENT_PLAN.md)為準。整理本身沒有新增 GPU 實驗。

## 1. 目標與比較契約

目標是 148 個部署 Conv/Linear 權重路徑都有量化配置，各層可不同；不是強制所有層用同一低 bit。124 個 activation 輸出量化器與受保護算子另列覆蓋。fake quant 使用浮點計算，不等於完成純整數部署。

本輪鎖定 V36 `v36-qsilu-a8-full-coverage-epoch1`，PTQ 使用對應 export，QAT 使用對應 EMA full-resume。非替換路徑保留 parent 的混合量化，不能誤寫為其餘層全 FP32 或全 W8。CPU 投影與 GPU rounding 的邊界差異及歷史跨 parent 比較限制，見[證據稽核](2026-09-07-evidence-consistency-boundaries.md)。

每個候選要同時列出相對 parent 的增量，以及相對 accepted 原始基準的總損失。總損失包含 activation 替換：8 項 mAP50 各下降不超過 1.5 百分點，8 項 mAP50–95 各不超過 4 百分點，包含 COCO Person。門檻只是可接受上限，不是追求掉到上限。

## 2. Activation：函數與輸出量化是兩件事

目前主線是 **qSiLU 函數＋LSQ+ A8 輸出量化**。qSiLU 處理非線性近似；LSQ+ 決定近似後數值如何映射有限碼階，兩者不可互相替代。

現有實作在 `src/yolo_quantize/activation_adapter.py` 包裝 activation，`quantizers.py` 使用下式 fake quant：

`x_hat = clamp(round((x - offset) / scale), qmin, qmax) * scale + offset`

scale 以 softplus 保持正值，offset 可學習，round 使用 STE，scale 梯度有縮放。這允許量化範圍配合偏斜、含負值的 activation 分布；freeze observer 是停止蒐集範圍，不等於 QAT 中 scale／offset 永久凍結。這是本專案實作描述，不宣稱已逐項重現論文所有訓練細節。

保留 A8 是為本輪權重比較固定條件，不代表 A8 永遠最佳。現有 bit 介面支援 2–16 的整數值，故 A7/A6/A5 不是格式上被禁止；但支援設定不等於已有該精度結果或硬體 kernel。後續應在鎖定的權重配置上，少量逐區降低 activation bit，重驗完整指標，必要時短 QAT。不能將獨立最好的 activation 與 weight 直接拼成 winner。

Hardswish 可作有證據的局部候選；Q3 與早期結果提供探索方向，不是同 V36 起點的優劣結論。poly_quality 已排除，poly_shift 舊路線封存，不混入本輪主線。A-SD4 是 activation 使用非均勻碼本的另一个問題，不能因 W-SD4 有效就推定有效；目前沒有本輪 A-SD4 完整驗證。

## 3. 權重：為何先 SD4／三元

它們提供不同的非均勻重建方式，有機會比統一整數格式取得更低儲存成本，但適用性必須由權重與任務結果共同判斷。

| 候選 | 假設及判斷依據 | 不能推定的事 |
| --- | --- | --- |
| Fixed-SD4 | 固定所選尺度，檢查 SD4 碼本本身是否適合該層幅值 | 小權重多就必然最好 |
| LS-SD4 | 同路徑／碼本，允許尺度學習，測試 QAT 能否改善分配 | 等同 activation 的 LSQ+、或保證勝 Fixed |
| exact ternary | 檢查投影到零及正負尺度的重建損失 | 僅靠零比例即可判斷 mAP |
| Paper-TWN／filterwise TWN | 比較閾值、尺度與粒度策略，揭露每通道差異 | 不同粒度的 metadata 成本相同 |
| W8/W7/W6/W5/W4 | 作為非均勻格式不適合時的精度／成本替代選項 | 非 2 冪次 bit 沒有價值或必然加速 |

「一個接近零、一個分散」可以當初始直覺，不能當選擇規則。三元若有許多可置零的值，且保留幅值接近所選尺度，通常較容易重建；若非零幅值跨距很大，單一尺度也可能失真。SD4 是否適合要看正規化幅值是否落在碼本附近；極端離群值、各通道尺度差異與實際碼本佔用都重要。至少聯合觀察 NRMSE／SQNR、零比例、幅值分位數、投影殘差及 scale metadata，再用輸出與任務精度驗證。

原始方法依據保留在[Paper-TWN 文獻紀錄](../research/2026-09-04-paper-twn-primary-literature.md)及[分布分析](2026-09-07-weight-distribution-detailed-analysis.md)。後者的歷史 parent 數字不能直接加入本輪同起點排名。

## 4. 證據階梯與有限 QAT

| 層級 | 已有工作 | 可以回答／限制 |
| --- | --- | --- |
| 權重重建 | 分布與格式投影 | 數值適配；不是 mAP |
| 單層輸出 probe | 148 層 × 4 格式＝592 組 | 固定小樣本的輸出敏感度；不是 592 次完整驗證 |
| 區域 PTQ | 十區 × 4 格式＝40 組 | 同 parent 下整區替換的任務精度；不是逐層 mAP |
| 短 QAT | 首批六組，各最多 5 epochs | 測試可恢復性；不保證 PTQ 失敗都能救回 |
| 累積配置／formal | 尚待本批分析後執行 | 驗證各區交互作用及最終泛化 |

六組是 Pose Fixed/LS-SD4 對照、同 MASF 區域的三種三元策略，以及 Detect predictor LS-SD4 的有限恢復試驗。不是只挑 PTQ 最佳，也不是所有差結果都投入 QAT。MASF 體積小，這組回答三元方法可行性，不能冒稱已取得主要模型壓縮收益。

沿用 AdamW 和既有分模組小 LR，因為它是此 parent 已運行的配置，可減少同時更動因素；不宣称普遍勝過 MuSGD。5 epochs、patience 5、warmup 1、scale-only 1，weight blend 由 epoch index 0 到 1 過渡到全強度。patience 5 在此上限內通常不會有效縮短時間，成本控制主要靠少量候選。Detect logical batch 128／micro 16，Pose 16，不另加影像雜訊，不另訓 baseline。

Fixed/LS-SD4 是同 24 條 Pose 路徑的尺度策略比較；其他可訓參數與 activation 仍可適應，不可稱「只有一層權重有訓練」。已完成兩組的第 5 回合 BBAT box mAP50–95 約 83.35%／83.44%，完整指標互有優劣；一個 seed 的微小差異不支持全面宣布 LS 必勝。epoch 5 也不保證是 best checkpoint，後续需按預先固定的多指標規則挑選與重驗。

## 5. 接續方式與收斂

先完成本批並檢查有效結果；用單層 probe 找出整區失敗中的低敏感子層，挑少數做完整 search mAP，而非再跑巨大笛卡兒積。接著 backbone→neck→head 逐步累積：每次從上一個已接受配置分叉，完整重驗 16 指標，不把獨立增量直接相加。失敗層保留 parent 已有量化，優先 SD4／三元，之後 W6/W5，必要時 W7/W4。

滿足精度後，以含尺度／metadata 的儲存估算與實測耗時選非支配候選；無 kernel 不填虛構硬體加速。四天為軟目標，後續先限制最多兩組新增短 QAT，收斂最多兩個 finalists，export 重載、覆蓋與 formal 驗證後才交付最終结論。formal 不用來反覆挑超參數。

600 秒 shell monitor 只讀 queue 狀態，事件才回到分析。已有 queue 自動串接本批工作；新的累積方案仍需結果分析及產生計畫，不能把「規劃有寫」稱為「已接上自動 queue」。
