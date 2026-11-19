# 現行實驗計畫

日後恢復入口：[交接表與重新啟動條件](RESUME.md)。本頁保留已執行契約，不是新訓練授權。

2026-09-08 最新決定：**本批收尾，量化延後至模型最後階段。** 六組 QAT 與十二個累積 PTQ 均已完成，supervisor／600 秒 monitor 已自然結束；不啟動原先考慮的額外兩組 QAT。最新[交接決策](../artifacts/queues/full-model-cumulative-0908/phase-hold.json)優先於以下歷史執行契約，恢復需使用者明確要求。完整數字見[階段收尾報告](reports/2026-09-08-quantization-phase-handoff.md)。

此頁是人讀入口；原 [full-model-continuous-0907.json](../configs/experiments/full-model-continuous-0907.json)、[六組清單](../artifacts/queues/full-model-continuous-0907/selected-qat-jobs-v2.json)與[累積 PTQ 計畫](../artifacts/queues/full-model-cumulative-0908/plan.json)只保留為已執行契約，不是新的執行授權。舊 v5、3epoch 四天草案及 poly_shift queue 同樣不得自動重啟。

## 起點與目標

使用 V36 `v36-qsilu-a8-full-coverage-epoch1` 的部署 export 與對應 EMA full-resume；不是重新訓練 baseline。148 部署 Conv/Linear 權重路徑都有量化配置，124 個 qSiLU／LSQ+ A8 activation quantizers。保護算子與浮點執行邊界另行列帳，不冒稱純整數部署。

## 目前順序

| 階段 | 內容 | 狀態 |
| --- | --- | --- |
| P0 | parent/export 重載、搜尋精度重驗、分布與 precision inventory | 已有驗證產物 |
| P1 | 148 × 4 單層輸出探測、10 × 4 區域 PTQ | 完成；前者不是 mAP |
| P2a | 六組短 QAT，測尺度學習、三元格式及恢復可能 | 六組各 5 epochs 完成 |
| P2b | backbone→neck→head 累積 PTQ，再 W6/W5 | 12 候選完成；保留兩個取捨點，額外 QAT 延後 |
| P3 | 最多兩個 finalists、export 重驗、formal 與部署限制報告 | 未啟動，延後 |

## 本批六組

1. Pose tower Fixed-SD4：固定所選權重尺度。
2. Pose tower LS-SD4：同 parent、同 24 條替換路徑，學習尺度。
3. MASF exact ternary。
4. MASF Paper-TWN。
5. MASF filterwise TWN：與前兩組使用相同 MASF 區域，揭露格式／粒度差異。
6. Detect predictor LS-SD4：PTQ 未達部署門檻的有限 QAT 恢復試驗。

各最多 5 epochs，patience 5；AdamW 沿用已驗證配置，warmup 1、scale-only 1、weight blend 由 epoch index 0 到 1 進入全強度。Detect logical128／micro16、Pose16，不加新雜訊，不另跑 sham。外部 V35 sham 僅歷史參考，不能叫同 parent sham 因果對照。

## 如何選擇與接續

### 同一 benchmark／公平比較契約

2026-09-08 使用者補充：比較必須同條件。此為日後恢復的必要檢查，不代表重新啟動實驗；目前只補充契約，未新增自動 benchmark gate。

1. **評估一致**：同資料版本、精確樣本／split、類別映射、輸入解析度與前處理、評估程式版本、confidence／IoU／max-det 與實際後處理設定；任務固定分列 COCO80、COCO Person、BBAT box／pose／ball／bat 的同一組 16 指標。search 與 formal 不混在同一排行榜；不為對齊 benchmark 另切資料或抽樣。
2. **方法比較只改指定變因**：權重量化比較固定同一 parent／EMA／export 血緣、activation 函數與輸出量化、替換路徑、其餘層方案及 calibration 資料／順序／預算。改 activation 或 bit 本身可以是研究變因，但須明列，其餘條件對齊；不能因為叫同一 benchmark 就視為已控制變因。
3. **分清比較單位**：單層隔離比較使用同一個未追加的 parent；同一累積步驟的格式比較使用同一個已接受前綴。不同累積步驟可以展示精度／成本變化，但不能冒稱同一配置的隔離敏感度排名。592 probes、40 分區 PTQ、六組 QAT、12 累積 PTQ 分組呈現。
4. **訓練預算一致**：QAT 格式比較固定初始化、train assignment、seed／資料順序、augmentation、logical／micro batch、epochs／更新步數預算、optimizer／scheduler、驗證頻率及 checkpoint 選取規則。已有同組實驗沿用原契約，不為文件補充回溯重跑。尺度是否可學習等方法本身需要的差異明列；PTQ→QAT 顯示額外訓練成本，不當成相同成本的無訓練比較。不同 QAT 輸出 checkpoint 不必相同，必須相同的是起點與選取規則。
5. **性能量測一致**：延遲／吞吐／記憶體比較固定裝置、軟體／backend、輸入與 batch、warmup／重複次數、同步方式及計時範圍（是否含前後處理）。bit／格式可作指定變因；若因支援限制改 backend，另列為整體部署系統比較，不把差異全部歸因於量化。fake quant 時間、解析儲存估算與真實整數部署量測分開。

每組可比較結果日後應附 benchmark 身分與差異清單，包括 dataset／split、evaluator／設定、parent、activation、路由／前綴、校正與訓練預算及必要的硬體環境。條件缺失或不一致者標為「探索／跨設定」，保留但不混入同條件方法排名。最終模型若換 parent，重建匹配對照後再引用新精度；不改寫舊結果來假裝一致。

相對 accepted 的 8 項 mAP50 下降各 ≤0.015、8 項 mAP50–95 下降各 ≤0.04，含 COCO Person 與 activation 替換。達標後再比较量化成本、單項精度與耗時；不把代理誤差當作 mAP 或硬體速度。PTQ 差不直接永久淘汰格式，只分配少量有理由的 QAT 恢復名額。

同 parent 的各區結果不能直接相加。本輪已由 backbone→neck→head 實測累積配置，每層可不同格式，其餘路徑保留 parent 量化。保留較保守（最差總下降 1.403／1.553 pp）及較壓縮（1.403／2.954 pp）兩個點；未決定 final winner。原考慮的最多兩組新增 QAT 現在均不執行。模型最後版本若改變，須先重鎖 parent、覆蓋與基準，不能直接移植目前精度結論。

## 執行邊界

六組 QAT 與十二個累積 PTQ 已結束，沒有自動下一批訓練。原狀態 `decision_required` 保留不回寫；另以 `phase-hold.json` 記錄延後，該檔是交接決策而非 OS／CLI 阻擋器。600 秒 monitor 已收到终點事件而退出，不宣稱它會在對話結束後自動喚醒模型。

資料集與血緣不改；本次未中斷 GPU child、不換主線 checkpoint、不刪檔。日後明確恢復時先讀[收尾報告](reports/2026-09-08-quantization-phase-handoff.md)，再確認模型起點與授權；[證據分層](reports/2026-09-07-evidence-consistency-boundaries.md)的比較限制仍有效。
