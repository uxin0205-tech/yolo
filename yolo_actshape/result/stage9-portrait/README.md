# Stage9 補充：768×512 與 512×768

**已於 2026-11-13 00:07（Asia/Taipei）完成 20／20 輪、最佳重載與獨立 CPU 可攜驗證。** 兩段各 10 輪均正常達到預算，沒有中斷、資源暫停或 SIGSEGV 重試。本候選已停止，沒有啟動下一階段；連同前輪共完成 120E。

最佳是第一段 **E5**，後續半 LR 10E 沒有超過此最佳。**768×512 的十項 AP 全部低於 512×768 最佳模型**；相較原主線，九項低於採用容許範圍，僅 BBAT ball box 有提升，因此正式模型維持原 qSiLU Rep17＋20／rect640。

共同 parent 為 qSiLU Rep17＋20 E1，配方與先前相同，僅改 H×W。影像等比例縮放補邊，不旋轉。

| 候選／段落 | COCO AP | person AP | COCO 球 AP | COCO 棒 AP | BBAT box AP | Pose AP |
|---|---:|---:|---:|---:|---:|---:|
| reference_rect640/e0 | 51.4347% | 64.2796% | 51.5782% | 46.9576% | 61.4371% | 90.1389% |
| rect512_768/adapted_best | 51.9632% | 64.5865% | 52.1912% | 45.3351% | 62.4501% | 89.9026% |
| rect768_512/e0 | 49.8811% | 62.2309% | 49.6644% | 45.2146% | 59.8781% | 89.3000% |
| rect768_512/adapted_best | 50.3517% | 62.9266% | 49.8507% | 44.6974% | 60.5912% | 89.5456% |

原主線的全部採用門檻：未通過；完整十項 AP 與逐項差異見 [result.json](result.json)。
現行正式交接保持；本次候選可從 [獨立交接包](handoff/README.md) 使用，不覆蓋原主線。

雙任務 Conv＋P.V 分析成本 45.508166 GMAC；與反向長寬的像素面積相同。
GPU replay／同步時間見 result.json；它們不是 RTL／HLS 或板端量測。qSiLU／D1／B2／S1／folded17+20 保持。

直向畫布的 COCO／BBAT5 平均補邊比例為 41.43%／52.63%，橫向則為 19.23%／21.57%，有效影像縮放也較小。這是精度下降的可能因素，尚未透過額外實驗分離因果。相同畫布面積沒有帶來分析 MAC 節省；GPU 同步中位數為 29.4187 ms，橫向既有量測為 29.6809 ms，單次微小差異不足以宣稱穩定加速。

資料為完整 COCO／BBAT5 v1，沒有重切或換標註；單 seed、validation 選型，未增加相同原尺寸 control 加訓。

來源見 [protocol.json](protocol.json)、[reference-review.json](reference-review.json)、[source.json](source.json)。

清理只處理本候選已無用途的 optimizer resume 與被最佳取代的權重，清單見 [cleanup.json](cleanup.json)。

本次實際移除兩份 optimizer resume，共 **681,214,190 bytes（0.63 GiB）**，E5 最佳權重與獨立交接包仍保留。2026-11-17 已核對 466 個來源 SHA、372 個交接檔案與四列指標；重載十項 AP 最大誤差為 0，42 項 GPU replay 及既有 CPU 可攜驗證通過。此次查核沒有新增 GPU 工作或重跑已通過的模型測試，詳見 [最終核驗](final-audit.json) 與 [中文完成紀錄](../../../docs/worklogs/2026-11-17-actshape-portrait-results.md)。
