# 768×512 同配置補充實驗

**已於 2026-11-13 00:07 完成 20／20 輪並停止。** 最佳為第一段 E5，後續半 LR 10E 未超過；十項 AP 均低於 512×768 最佳，原正式主線保持。見 [完成結果](../result/stage9-portrait/README.md)、[候選交接](../result/stage9-portrait/handoff/README.md) 與 [完成核驗](../../docs/worklogs/2026-11-17-actshape-portrait-results.md)。

依 2026-11-12 授權，從與 512×768 相同的 qSiLU Rep17＋20 E1 parent，獨立跑 10E＋半 LR 10E。尺寸順序是 H×W；影像等比例縮放與補邊，不旋轉。

[事前參考](reference-review.json) · [固定配方](protocol.json) · [中文工作紀錄](../../docs/worklogs/2026-11-12-actshape-portrait-training.md) · [狀態](../queue-status.json)

在工作區根目錄執行 `yolo_actshape/run_portrait.sh status` 查詢。CPU 查核為 `CUDA_VISIBLE_DEVICES='' yolo_actshape/run_portrait.sh preflight`；GPU queue 為 `yolo_actshape/run_portrait.sh run`。執行中不可修改來源鎖或配方。

背景程序每 600 秒監測、30 秒檢查資源；完成本候選後停止。新增來源及結果與原 100E 分開，原主線保持。權重、續訓檔及資料不作 Git 發布。
