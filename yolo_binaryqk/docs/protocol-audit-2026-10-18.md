# 2026-10-18：0.51～0.52 精度與恢復訓練配方核對

> 後續更新：使用者要求錯誤產物直接刪除、不封存；舊 v1 權重／resume／排程已移除，原產物連結改指向刪除清單與小型精度證據。現行方法以第 1 階段 v2 protocol 為準。

使用者記得的數字有依據。本次以目前 evaluator 對同一份原始 YOLO26m 跑完整 COCO val2017 5,000 張，得到 internal AP50–95 **0.518002**、官方 COCO bbox AP50–95 **0.524686**。只做評估，沒有加訓 baseline，也沒有修改權重或資料。

本輪 B1 的低精度確實存在。前一輪把已訓 A0 的低學習率微調設定套到初次二值替換，且進一步限制為 Q/K-only；它與舊流程的恢復方式不同，不適合作為分支能力上限的結論。已停止延續此配方，保留 B1 作為受限恢復的診斷。**第 1 階段尚未完成，尚無 branch winner。**

## 精度證據

以下均為 COCO80 bbox AP50–95，數值範圍 0～1。歷史 internal 與本輪 internal 屬同類指標，但歷史模型未全部重新評估，不能宣稱各次執行細節完全相同。官方與 internal 指標分欄，缺值表示未取得該次官方評估。

| 模型／狀態 | Internal AP50–95 | 官方 COCO AP50–95 | 證據 |
| --- | ---: | ---: | --- |
| 原生 B26-FP，歷史 | 0.517998 | — | [Attention 比較表](../../yolo_attention/reports/comparison.csv) |
| H-SCR，Attention 適應 10 epochs | 0.495243 | — | 同上 |
| W-DIR，再接全模型恢復 | 0.507457 | — | 同上；26/40 epochs early stop，最佳 epoch 21 |
| optimize 已訓 BinaryQK A0 | 0.506739 | — | [Optimize 精確比較表](../../yolo_optimize/reports/performance/comparison.csv) |
| 原生 parent，本次重驗 | **0.518002** | **0.524686** | [完整評估](../artifacts/stage1/protocol-audit-20261018/parent-evaluation.json) |
| B1 替換後，epoch 0 | 0.388735 | 0.401422 | [B1 原始結果](../artifacts/stage1/cleanup-invalid-v1.json) |
| B1 Q/K-only，epoch 10 | 0.428949 | 0.440541 | 同上；本次 run 最佳 epoch 10 |

同一 evaluator 下，原始 parent 與 B1 最佳模型的 internal 差距約 **8.91 AP**，官方差距約 **8.41 AP**；不能用 evaluator 差異解釋全部落差。B1 由 epoch 0 到 10 的 internal 改善約 4.02 AP，代表有恢復，但沒有證明已充分適應。

Parent SHA-256：`401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`。本次 E2E、FP32、640、batch 8、conf 0.001、iou 0.7、max_det 300；完整影像集合與 annotation SHA 保留於評估／資料契約。Ultralytics 8.4.90 實際來自 `yolo_p2/ultralytics`。

## 為何不能照舊配方繼續

| 項目 | 初次二值化舊流程 | optimize 後期 A0 微調 | 本輪已執行 B1 |
| --- | --- | --- | --- |
| 起點 | 原生 parent → H-SCR → W-DIR | 已約 0.5067 的 A0 | 原生 parent 直接替換 B1 |
| 可更新範圍 | Screening：Attention；Recovery：全模型 | Q/K＋Detect head | 僅 Q/K Conv、BN affine |
| 學習率 | `5e-5` | Q/K `5e-7`、head `2.5e-5` | Q/K `5e-7` |
| 預算 | Screening 10；Recovery 最多 40、patience 5 | 短期恢復 | 10 epochs |
| Score scale | H-SCR/W-DIR 為 dynamic，含可學 gamma | 已訓硬體化設定 | 初始校準後固定 FP32 |
| Normalization | Exact Softmax | Float PWL 訓練／BitTrue 評估 | Native Softmax |
| Batch／dtype | 16／AMP | 32、logical 128／AMP | 8、logical 128／FP32 |

來源：[H-SCR 設定](../../yolo_attention/artifacts/runs/h-scr/training.yaml)、[W-DIR 設定](../../yolo_attention/artifacts/runs/w-dir/training.yaml)、[舊 scope 實作](../../yolo_attention/src/yolo_attention/integration.py)、[A0 恢復實作](../../yolo_optimize/experiments/studies/pre-fusion-full35-b100/scripts/recover_a0.py)、[本輪實際 recipe](../artifacts/stage1/cleanup-invalid-v1.json)。歷史 checkpoint 的完整性及限制見 [TRAINING_AUDIT](../../yolo_attention/reports/TRAINING_AUDIT.md)。

上述是已確認的差異；尚未用配對實驗拆出 LR、scope、固定 scale 的個別貢獻。舊實作的梯度問題也不能重新引入：新實作已驗證精確二值前向、signed-dot 反向、Q/K 非零梯度及實際更新，這些修正仍要保留。

## 接續處理

應沿用預訓練 parent 做恢復微調，毋須隨機初始化重訓。新一輪須先固定共同配方，以舊流程的 Attention 適應及必要全模型恢復作參考，核對 loss scaling、BN／EMA、LR、資料與訓練成本，再做短測及正式比較。`5e-5` 是有來源的起始參考，不能只把數字放大 100 倍便宣稱問題已解決。

固定 scale 的影響須先做 score／probability／P·V 診斷；不得為追求舊精度，默默改成動態部署係數或提前展開第 2 階段的 Q10／二次冪矩陣。本輪分支仍只有 B1／B2／B3，保持共同 normalization、資料、parent 與公平預算。

修正版三候選需由同一份原始 parent 分別開始，使用新 run ID。不能讓 B1 多得舊 10 epochs 的權重適應，再與新起步 B2／B3 混成共同預算比較。舊 B1 結果、最佳權重及 resume 均保留；尚未執行新版恢復，也不保證能回到 0.51。

## 排程狀態與驗證

B1 於 2026-10-18 05:07:39（Asia/Taipei）完成。B2 於 05:07:59 在 GPU preflight 發現另一個 compute process，尚未開始訓練即退出；B3 未啟動。原 supervisor 的 failed 狀態與 log 保留。原先 13:22～16:22 的完成估計已失效，現在沒有長訓工作或新的 ETA。

已修正 supervisor：GPU 占用改為每 600 秒等待重查；處理 preflight 與 child 啟動間的占用競態；續接時嚴格核對完整結果及最佳 checkpoint，略過已完成候選；用 file lock 防止雙重排程。真正訓練錯誤仍停止，未完成 run 的 resume 由 trainer 核對 SHA、recipe 與 manifest。舊隊列已有 `STOP` 與 [停止延續原因](../artifacts/stage1/cleanup-invalid-v1.json)，不因排程修好便自動恢復不合適的配方。

驗證：38 項 CPU 測試通過，Ruff 通過；B1 最佳權重實際 strict reload 通過。新增測試涵蓋 GPU 忙碌後恢復、STOP 中斷等待、結果不完整／非法 checkpoint 路徑拒絕、完成候選略過、資源競態重試及排程鎖。背景續跑尚未做新的 GPU 長訓驗證。

可重用小型資料：[精確比較 CSV](../artifacts/stage1/protocol-audit-20261018/comparison.csv)、[B1 全曲線 CSV](../artifacts/stage1/protocol-audit-20261018/b1-curve.csv)、[來源 SHA 與稽核 JSON](../artifacts/stage1/protocol-audit-20261018/audit.json)。只讀這些資料即可重畫，不需要 GPU 或 checkpoint。
