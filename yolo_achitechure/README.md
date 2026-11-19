# RepConv 與 ACB 訓練研究

**2026-11-08 使用者已指定 `rep17_20` E1（qSiLU 訓練版）為主線。** [handoff](handoff/README.md) 的預設載入已切換；詳見 [selection.json](result/stage8-retrain/selection.json)。下方 `rep17_20_silu` 是先前自動推薦，不覆蓋使用者選擇。

**Stage8 新輪於 2026-11-08 14:33 完成；原自動建議為 `rep17_20_silu` E1，使用者已改選頁首的 qSiLU 訓練版。** 五組各完成 5E，SiLU 另做 3E 硬體適應但未採用較差後段；全部最終部署驗證通過，GPU 已釋放。共同 parent 是 Stage7 qSiLU E8，這是預訓權重適應。

| 候選 | 位置 | 結構 | 訓練 activation／normalization |
|---|---|---|---|
| rep17 | 17 | 原生 3×3＋1×1 RepConv | qSiLU／Float PWL |
| rep20 | 20 | 原生 3×3＋1×1 RepConv | qSiLU／Float PWL |
| rep17_20 | 17、20 | 原生 3×3＋1×1 RepConv | qSiLU／Float PWL |
| rep17_20_silu | 17、20 | 原生 RepConv | 僅兩個 Rep.act 用 SiLU，其餘 qSiLU；Float PWL |
| acb17_20 | 17、20 | 3×3＋1×3＋3×1 ACB | qSiLU／Float PWL |

Float PWL 固定 [-10,0]／20；部署全部轉為單 3×3 Conv＋qSiLU，完整 D1 整數 PWL＋LUT256 reciprocal。B2 二值 QK、S1 Q10／signed16 bias 繼承。只訓指定分支與兩 heads、BN running 固定；每輪完整 COCO5000／BBAT683 的訓練算子及硬體部署 AP 分開記錄，按部署 AP 選型。

每組5E；持續改善才降 LR 延長5E。SiLU 候選若有明顯部署切換落差且未嚴重退化，可接一次3E qSiLU/D1適應。總上限53E，依條件減少，較差後段不覆蓋最佳。原Stage7與舊最佳也參與最終採用判斷。

[執行手冊](RUNBOOK.md) · [protocol](stage8/protocol.json) · [optimize／原始論文核對](stage8/reference-review.json) · [CPU 驗證](stage8/cpu-contracts.json) · [工作紀錄](../docs/worklogs/2026-11-07-repconv-native-training.md)

[無 Rep／Rep／ACB 完整比較與推薦](../docs/research/stage8-comparison-2026-11-08.md)：包含十項 AP、43 個完整 epoch、PolyQuality 比較、訓練／部署差距與成本。推薦的 Rep 主要改善球棒，整體 COCO 接近無 Rep qSiLU；ACB 未通過球 Pose 門檻。

新輪小型證據位於 `artifacts/stage8/repconv-variants-20261107/`，正式交付位於 [result/stage8-retrain/](result/stage8-retrain/README.md)。目前 [handoff/](handoff/README.md) 指向新輪可攜包，預設為訓練與部署皆 qSiLU 的 `rep17_20` E1（部署 D1）。先前已清理約 1.93 GiB；本次另清理約 4.08 GiB，包含所有未採用權重、舊 optimizer resume、舊可攜包與 `achitechure_2`。僅保留選定 folded 模型、多分支最佳 EMA、小型證據及 optimize 仍依賴的 `achitechure_1`。詳見 [清理清單](result/stage8-retrain/cleanup-final-20261108.json)。

ACB 參考 [作者實作](https://github.com/DingXiaoH/ACNet/blob/master/acnet/acb.py)，使用零初始化新增BN與固定running來繼承權重，是本專案適應方式。沒有新加 identity 分支，因目標層stride2不能直接相加；不把不同通道或解析度的identity硬塞進去。單seed，沒有未改方法control加訓，未量測硬體加速。

Stage8 已完成；後續 [yolo_actshape](../yolo_actshape/README.md) 已依擴充授權完成 Stage9 累計 120E，正式仍採原 qSiLU Rep17＋20。全輪整理與交接見 [result/](../result/README.md)。所有正式資料與 `yolo_optimize` 維持唯讀。
