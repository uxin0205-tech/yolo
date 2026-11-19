# 整合研究索引

**11 月 19 日現行交接** 採用固定 512×768 硬體版與原 rect640 精度版。下列 11 月 17 日及更早的段落按當時選型記錄。[雙版本模型](../../result/README.md) · [yolo_reshape 特徵圖導讀](../../yolo_reshape/README.md)

2026-11-17：第 1～9 階段已完成本輪交付；特徵圖研究合計 **120E**。正式選定 **qSiLU 訓練版 Rep17＋20 E1／D1／原 rect640**，不是原自動推薦的 SiLU 訓練候選。所有本研究 GPU queue 已結束。

- **[研究總報告](yolo-research-summary-2026-11-17.md)**：十項 AP、每階段採用／未採用理由、真實融合血緣、硬體簡化、形狀研究與限制。
- **[統一交接 result/](../../result/README.md)**：正式模型、載入程式、多分支 EMA、manifest、清理及驗證。
- [小型數據與重畫程式](data/integrated-20261117/README.md)：不需 GPU／checkpoint。
- [整合研究計畫及完成狀態](yolo-research-plan-2026-09-21.md)：保留原規劃與授權修訂，不代表額外實驗授權。
- [無 Rep／Rep／ACB 歷史比較](stage8-comparison-2026-11-08.md) 與 [最終使用者選擇](../../yolo_achitechure/result/stage8-retrain/selection.json)。
- [特徵圖工作區](../../yolo_actshape/README.md)、[前 100E 結果](../../yolo_actshape/result/stage9/README.md)、[768×512 補充 20E](../../yolo_actshape/result/stage9-portrait/README.md)。
- [融合研究與訓練入口](../../yolo_combine/README.md)、[optimize 核對與重建](../worklogs/2026-10-29-stage6-clean-rebuild.md)、[融合研究完成紀錄](../worklogs/2026-10-30-stage6-results.md)。
- [MASF 研究入口](../../yolo_masf/README.md)、[KD 第五輪覆核與 Stage4 接續](../worklogs/2026-10-19-masf-stage4.md)。
- [本次整理紀錄](../worklogs/2026-11-17-integrated-research-handoff.md)、[完整工作紀錄索引](../worklogs/README.md)。

後續文件中的「待指定」「當時選定」應按其日期理解；現行狀態以本索引、總報告及 `result/manifest.json` 為準。未啟動新的研究或整網量化。
