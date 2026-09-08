# Full35 全模型混合量化

目標：讓 Detect＋Pose 模型的部署權重全部有量化配置，並允許各層採不同格式，在精度與量化成本間取捨。**目前是量化模擬／搜尋驗證，尚未完成純整數部署。**

**2026-09-08 階段收尾：依使用者與老師的新決定，量化延後至模型最後階段。六組 QAT、十二個累積 PTQ 已完成，queue／monitor 自然結束；不接新增訓練。** 日後恢復需重新確認，現有資料及權重保留。

日後回來請從[恢復研究交接表](docs/RESUME.md)開始；本機 checkpoint／原始結果保留，GitHub 提供程式、報告及凍結證據快照，不是完整訓練環境備份。

## 先看這三份

1. [階段收尾報告](docs/reports/2026-09-08-quantization-phase-handoff.md)：完整進度、六組 QAT、十二個累積 PTQ、兩個取捨點及延後事項。
2. [目前執行計畫](docs/CURRENT_PLAN.md)：這輪做什麼、接下來做什麼。
3. [比較條件與限制](docs/reports/2026-09-07-evidence-consistency-boundaries.md)：哪些是權重誤差、輸出探測、mAP 或 QAT，避免混用歷史結果。

## 已做到哪裡

想了解選擇原因，先看[量化方法與決策說明](docs/reports/2026-09-08-quantization-method-and-decisions.md)：qSiLU／LSQ+、SD4／三元、逐層驗證與短 QAT 的關係。

以下為收尾快照；原始完成證據與[延後決策](artifacts/queues/full-model-cumulative-0908/phase-hold.json)分開保存，舊 `decision_required` 不代表要自動接 QAT。

| 項目 | 狀態 |
| --- | --- |
| 共同起點 | V36 鎖定 parent，qSiLU＋LSQ+ A8，148 個部署權重路徑 |
| 單層輸出 sensitivity | 148 層 × 4 格式＝592 組完成；不是逐層 mAP |
| 十區 PTQ | 10 區 × 4 格式＝40 組完整搜尋驗證完成 |
| 五回合 QAT | 六組各 5 epochs 全部完成；Detect predictor LS-SD4 第 2–5 回合恢復過門檻 |
| 累積配置 | 11 階段、12 候選 PTQ 完成；保留較保守／較壓縮兩個取捨點 |
| 延後、未完成 | 額外累積 QAT、逐層 mAP 擴展、finalists、最終 export／formal、整數部署 |

已完成[六組 QAT](docs/reports/2026-09-07-continuous-qat-recovery-results.md)中，Detect predictor LS-SD4 的 COCO／Person mAP50–95 相對 PTQ 恢復約 **4.19／4.46 百分點**。累積較壓縮配置的最差 mAP50／mAP50–95 總下降為 **1.403／2.954 百分點**，codes＋scales 估算較 V36 減少 **9.40%**；不是實測檔案大小／加速，也不是 final winner。

## 固定規則

- 同 parent 比較；先 SD4／三元，再 W6／W5，必要時 W7／W4。各層可不同，但不能直接拼接獨立最優結果。
- 8 項 mAP50 總下降各不超過 **1.5 pp**，8 項 mAP50–95 各不超過 **4 pp**；包含 activation 替換與 COCO Person。
- 首批最多 6 組 QAT，各最多 5 epochs、patience 5、AdamW；Detect logical batch 128／microbatch 16、Pose 16，不加額外影像雜訊。
- 四天原為軟目標，目前依新決定延後量化；沒有待啟動的新 QAT。原 600 秒 shell monitor 已在終點退出，不重啟已完成 queue。

## 資料夾怎麼看

| 入口 | 用途 |
| --- | --- |
| [docs](docs/README.md) | 現行計畫、報告、工作紀錄與歷史區 |
| [configs/experiments](configs/experiments/README.md) | 現行機讀計畫及不可任意移動的版本化契約 |
| [artifacts](artifacts/README.md) | 原始結果、parent、checkpoint、queue；不是日常閱讀入口 |
| [scripts](scripts/README.md) | CPU 報告／稽核、GPU 執行入口與維護命令 |
| [deliverables](deliverables/README.md) | 可分享圖表；舊圖有明確歷史日期 |
| [歷史區](docs/archive/README.md) | 舊 activation、poly_shift、V35/V36 血緣與過期規劃 |
| [整理紀錄](docs/organization/README.md) | 10 份原位報告、12 份歷史封存、3 份重複報告刪除及恢復位置 |

## 資料集與工作紀錄

模型唯讀來源：`/home/uxin/yolo/yolo_combine/final/full35/`。COCO80／Person 使用 `/home/uxin/yolo/coco2017.yaml`。

BBAT5 只用不可變 `/home/uxin/yolo/original/pose/derived/bbat5-v1/`：Pose 為 `configs/pose.yaml`，ball/bat Detect 為 `configs/detect.yaml`；registry 是 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。沿用既有 search/formal assignment，不重切、不改 labels；runtime View 僅重現同一資料版本。

完整規範見[共用資料集規範](../docs/agents/bbat5-datasets.md)。所有修改與驗證見[中文工作紀錄](docs/worklogs/README.md)。本次 `5090 Stop 0908` 發行範圍見 [PUBLICATION_0908.yaml](PUBLICATION_0908.yaml)；舊發行契約保留於 `PUBLICATION_0907.yaml`、`PUBLICATION_MANIFEST.yaml`，不回寫歷史 hash。
