# RTX 5060 Ti 最終結果入口（2026-08-24）

完整資料 Phase C 已完成。Full35 C-100%／Partial75 C-100% 的 Bit-True COCO mAP50-95 分別是 `0.501395`／`0.500428`，都低於各自 A2，因此 accepted checkpoint 仍是 A2。

Full35 A2 與 Partial75 A2 的 COCO 差距小於 `0.001`；依既定同機 tie-break，Partial75 的 FP16 p50 latency、GFLOPs 與 Params 較低，因此是兩個 P3-MASF 架構中的工程 winner。但 Partial75 A2 與 A0 的整體 COCO accuracy 幾乎相同，A0 latency 也更快，故不宣稱 P3-MASF 已優於 baseline。

此處宣告的是 Full35／Partial75 的「架構 winner」。`EXPERIMENT_SPEC.md` 規劃的 winner LR tuning T1／T2／T3 尚未執行，因此沒有虛構 tuning 結果或最終超參數 winner。

最重要的使用者 `detect_dataset` 結果：

- BBT5 overall AP50-95 最高：Full35 C-30%，`0.412511`。
- BBT5 sports-ball AP50-95 最高：A0，`0.318082`。
- BBT5 baseball-bat AP50-95 最高：Full35 C-30%，`0.515345`。
- 新的 Full35 C-100%：overall／ball／bat 為 `0.404843`／`0.309175`／`0.500511`。
- 新的 Partial75 C-100%：overall／ball／bat 為 `0.402681`／`0.306285`／`0.499077`。

相對各自 A2，Full35 C-100% 的 BBT5 overall／ball／bat 變化為 `+0.005395`／`-0.004962`／`+0.015753`；Partial75 C-100% 為 `+0.001127`／`-0.007387`／`+0.009640`。也就是 bat 有提升，但 ball 沒有；而兩者完整 COCO overall 都下降，所以不能取代 A2。

完整報告與證據：

- [最終結論、訓練、資源與 latency](../final/reports/RTX5060TI_FINAL_0824.md)
- [11 個候選的完整 COCO／BBT5 AP、P、R、F1](../final/reports/FULL35_PARTIAL75_AP.md)
- [訓練 mAP 比較圖](../final/reports/figures/training-map-comparison.png)
- [BBT5 AP 比較圖](../final/reports/figures/bbt5-ap-comparison.png)
- [RAM／VRAM 圖](../final/reports/figures/training-resources.png)
- [集中式權重索引](../final/weights/README.md)

Machine-readable 摘要是同名 `.json`；一列一 run 的訓練摘要是同名 `.csv`。2026-08-22 的 `half` 報告保留為歷史快照，不回填後續結果。
