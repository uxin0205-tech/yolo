# 訓練完成結果

固定 bbat5-v1 formal val（683 張），統一以 960 評估。沒有獨立 test split；結果為驗證集選模表現。

| 階段 | Box mAP50–95 | Pose mAP50–95 | 兩者平均 |
| --- | --- | --- | --- |
| stage1_adamw_640 | 0.58332 | 0.90225 | 0.74279 |
| stage2_adamw_640 | 0.59519 | 0.89700 | 0.74609 |
| stage3_adamw_960 | 0.61322 | 0.90847 | 0.76085 |
| stage4_musgd_960 | 0.62433 | 0.90507 | 0.76470 |

最佳候選：stage4_musgd_960。以 Box 與 Pose mAP50–95 平均選擇；各類別結果見 comparison.json。

best.pt 為上述最佳候選；last.pt 與 last-resumable.pt 為最後 MuSGD 階段的最後權重與完整續訓 checkpoint。

困難與解法：啟動期間見工作紀錄；執行警告與錯誤見 logs/。未解風險：沒有獨立測試集，且只比較此條接續訓練路徑，非全面超參數搜尋。
