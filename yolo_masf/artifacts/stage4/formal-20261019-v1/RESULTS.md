# 第四階段 MASF 結果

| 任務 | Parent AP | Alpha-on AP | Alpha-off AP | 建議 |
|---|---:|---:|---:|---|
| detect | 0.514620791 | 0.514620791 | 0.514620791 | 未同時證實 parent 改善與 MASF 正貢獻，建議維持 parent |
| pose | 0.920709389 | 0.920709389 | 0.920709389 | 未同時證實 parent 改善與 MASF 正貢獻，建議維持 parent |

Detect 為 COCO val5000 官方 bbox AP；Pose 為 BBAT5 val683 原生 keypoint AP，不能混比。各輪完整指標見 task-metrics.csv，圖可由 CLI plot 在無 GPU、無 checkpoint 下重畫。單 seed、無 control 加訓；alpha-on/off 為同 checkpoint 診斷。CPU 等價與折疊證據在 cpu-diagnostics.json；沒有硬體實測加速結論。第四階段完成，停止，未啟動第五階段。


## 已訓五輪與後續補查

上表 alpha-on/off 對應被選定的 epoch 0，只能確認初始等價。真正第五輪 Detect AP 0.514576492、Pose keypoint AP 0.919994127；Pose bbox AP 0.630224466。已從各自 resume 匯出 `result/stage4/{task}/trained-epoch05.pt` 並嚴格重載，延長 run 將補已訓第五輪的正式開關診斷。不能把零 alpha 的 E0 表當作已訓分支沒有作用的證據。

使用者已授權延長與方法比較；原五輪預算已完成，第四階段整體仍在研究中，未啟動第五階段。
