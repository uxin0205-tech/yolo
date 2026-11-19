# 特徵圖形狀研究

2026-11-13 00:07：<!-- portrait-status -->**768×512 已完成 20／20E；Stage9 合計完成 120E。** 最佳為第一段 E5，十項 AP 均低於 512×768 最佳，原 qSiLU Rep17＋20／rect640 保持。<!-- /portrait-status --> 已完成交接與清理，本研究已停止。[直向結果與比較](result/stage9-portrait/README.md)。

前五組於 **2026-11-11 06:25（Asia/Taipei）** 完成，各 10E＋半 LR 10E，共 **100 輪**；追加 768×512 於 11 月 13 日完成 20 輪，累計 **120 輪**。完整 AP、重載與交接驗證完成，GPU 已釋放，沒有啟動下一階段。

正式採用 **qSiLU 訓練版 Rep17＋20 E1 與原 rect640 流程（`reference_rect640`）**。此流程按 batch 做矩形 letterbox，並非強制固定 640×640。固定尺寸候選沒有通過全部採用門檻；512×768 的 COCO AP 較高，但球棒與 Pose 有退化，詳見 [完整結果](result/stage9/README.md)。

[handoff](handoff/README.md) 是可直接使用的載入程式、權重及 manifest，指向 `result/stage9/handoff/`。部署保留 qSiLU／D1 整數 PWL exp＋LUT256 reciprocal／B2／S1／folded17+20。三個新增尺寸的最佳權重另外保留作成本取捨；未採用權重與 optimizer resume 已依清單移除約 3.57 GiB，沒有另建封存。

報告、CSV 與 [繪圖程式](result/stage9/redraw.py) 可獨立查閱；執行 `python yolo_actshape/result/stage9/redraw.py` 不需要 GPU 或 checkpoint。原七組 294 項加上直向 42 項，共 336 項 GPU replay 已通過，但不代表已完成 RTL／HLS、板端量測或整網整數量化。尚未更改固定輸入下的內部 stride 架構。

[硬體尺寸分析](analysis/HARDWARE_SHAPES.md) · [初輪配方](protocol.json) · [擴充配方](extension/protocol.json) · [最終核驗](result/stage9/final-audit.json) · [完成紀錄](../docs/worklogs/2026-11-11-actshape-stage9-results.md)

2026-11-17 已整理全輪成果：[統一交接](../result/README.md)、[總報告](../docs/research/yolo-research-summary-2026-11-17.md)。前 100E 與補充 20E 的原始來源鎖／結果各自保留，正式 handoff 不變。
