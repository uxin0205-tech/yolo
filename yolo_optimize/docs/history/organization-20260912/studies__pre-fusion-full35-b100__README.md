# 歷史入口快照（非目前狀態）

原路徑：`studies/pre-fusion-full35-b100/README.md`。原始位元組另存同名 `.txt`；整理日期 2026-09-12。

# 融合前 Full35-B100：方向 1

後續狀態覆蓋（2026-09-11）：本研究已完成方向 1，使用者選 P3 bridge 接續的新 combine／activation／KD／推論也已有結果。所有原始產物已獨立封存並比對 SHA256，見全階段總報告（本機／歷史參照：`../../../reports/consolidated-20260911/README.md`；未隨本次報告發布）。下方保留當時分段決策，不代表目前有 GPU queue。

建立：2026-09-09。使用者指定獨立 Detect 的 Full35-B100 為新起點；本研究不接 Pose、不更換融合後權重。舊研究完整保留在 融合後結果（本機／歷史參照：`../../../optimizations/integrated-roadmap/round1-audit.md`；未隨本次報告發布），舊 artifacts 不搬移、不覆寫。

## 最新分段決策

2026-09-10 結果稽核：最後 P2 MASF 的 control／candidate 均已完成 5 個完整 epochs，五回合皆未通過增準門檻。E5 相對同回合 control 的 overall／person 為 −0.014887／+0.009926 個百分點，平均回合時間增加約 39%。依既定門檻不採用此 MASF 方案；本次只查核結果，未啟動後續訓練。完整配對與範圍見[最新結果稽核](<../../../../docs/worklogs/2026-09-10-masf-results-audit.md>)，原設計見[P2 實驗與門檻](<../../worklogs/2026-09-10-p2-masf-last-trial.md>)。以下為歷史階段。

2026-09-10：MASF 梯度橋接 E6–E10 已完成，收益仍未達方法驗收門檻。無 MASF control E8／MASF bridge E8 均通過獨立匯出與完整 COCO5000 重驗（四項 AP 差值0），各完成8張推論。預定實驗與數值驗收已結束，精度未完全回到FP，不自動升格；融合來源是否保留MASF待確認，後續階段未啟動。見 方向1集中報告（本機／歷史參照：`../../../reports/direction1-20260910/README.md`；未隨本次報告發布）。以下皆為歷史階段。

2026-09-10：真實 loss 的 256 張 train-only 校準通過，one-to-one 訓練梯度係數固定為 0.012076444778011642；128 張真實更新、首 loss 配對及推論／匯出等價性通過。已啟動 `masf-task-bridge-v1` E6–E10，直接比較完成的同起點 native fork，不重跑對照；推論沒有新增 MASF pass。方向 1 尚未完成，詳見 [梯度橋接訓練紀錄](<../../worklogs/2026-09-10-masf-task-bridge-training.md>)。以下為歷史階段。

2026-09-10：control／fork E6–E10 已全部完成，提高 P3 head LR 後仍沒有穩定增準，不升格候選、不盲目加訓。已完成 CPU 梯度診斷：原生 one-to-one 的 detach 阻止其直接監督 MASF；先 detach raw P3 再執行 MASF 可保持前向並接通該監督，但尚未證明 AP 收益。下一步是真實 loss 的 train-only 梯度校準，尚未啟動新 GPU 工作。詳見 [E10 結果、梯度與等待限制](<../../worklogs/2026-09-10-masf-e10-gradient-and-wait.md>)。以下續訓排程為歷史記錄。

MASF control／shared／fork 各 5 epochs 已全部完成，尚無足夠的增準收益；fork 比 shared 少部分 overall 退化，但不能宣布移位已成功。CPU 診斷確認 context 確有更新、P3 head 權重變動僅約 0.042%；下一步配對續訓 control／fork 至共同 E10，只將 P3 head LR 提高 5 倍，其他 scope、MASF LR、BN 與架構不變，保留 optimizer／EMA、不重新 warmup。見 [MASF E5 與 head 適應](<../../worklogs/2026-09-09-masf-e5-head-adaptation.md>)。舊三組全部保留，新流程先驗證續訓狀態與 smoke，再啟動正式訓練。

MASF 暫時關閉診斷已完成：B100 overall／person 差值 +0.000116168／−0.000057415，沒有大幅主要指標依賴。接續同 late E8 parent 的 control／shared／P3 Detect-only 三組增準實驗，各 5 epochs；MASF context 重用 B100 初始化但 gate 歸零，先驗證整圖等價、P4／P5 隔離、真實梯度與匯出，再正式訓練。既有無 MASF parent 已就緒，故不額外跑 B100 關閉 bridge。細節、超參數與風險見 [MASF 增準計畫](<../../worklogs/2026-09-09-prefusion-masf-p3-start.md>)。AST 已通過，實際前置結果與 AP 增準仍待驗證。

Rep17 的 control／rep 各 5 epochs 均已完成；Rep 最佳 E4 overall／person=0.508097536／0.627500653，未超過起點，也未達相對同回合 control 的 +0.001 改善門檻。不採用、不擴 layer20。接續 B100 原位 MASF alpha=0 的免訓練診斷，目的是區分既有依賴與收益，**不是以刪除 MASF 為優化終點**；使用者再次確認 MASF 的目的為提高精度。正式位置／訓練實驗仍以 MASF 增準及 overall／person 驗收為目標。見 [融合前 Rep17 結果與 MASF 診斷](<../../worklogs/2026-09-09-prefusion-rep17-result-masf-off.md>)。下方「下一步 RepConv」屬歷史階段。

HOG 已於 E4 依 patience 4 停止，同回合四項 AP 都低於原生 E4；本版不採用。固定 1024 張 train 檢查發現 ball 自身框有 13／59 個無有效 P3 cell，bat 是 1／38；恢復舊 head BN 統計也未改善。見 [HOG 結果](<../../worklogs/2026-09-09-hog-result-ball-bat.md>)。下一步是兩組都只訓練 layer 17、其餘參數與全部 BN 統計固定的 [RepConv 對照](<../../worklogs/2026-09-09-rep17-prefusion-start.md>)，先做整圖與部署前置驗證；不疊 HOG，不擴 layer 20。MASF 後續仍觀察 ball／bat。

最新範圍補充：HOG／MASF 另列 COCO sports ball／baseball bat 作輔助觀察；overall／person 仍為主要 gate，不恢復 ball／bat 硬性停止條件，也不擴回融合後資料。現有驗證已保存這兩類，不需中斷正在執行的 HOG。見 [ball／bat 觀察紀錄](<../../worklogs/2026-09-09-hog-masf-ball-bat-observation.md>)。

E10 未超過 E8，保留 late E8 探索權重但不正式升格；接續原方向1的原生5／HOG最多10（patience4、warmup1）同起點比較。已有 HOG 12項測試與256張train-only梯度校準通過，μ約0.875對應P3梯度比例約5%，正式訓練前仍須真實更新／strip驗證。最新結果、訓練範圍及原構想差異見 [E10與HOG紀錄](<../../worklogs/2026-09-09-e10-hog-prefusion.md>)。不處理融合後、ball／bat gate 或 person-only；以下歷史階段均保留。

E8 完成：late overall／person=0.508153635／0.627520517，narrow=0.507432701／0.626632484。late 的獨立匯出權重已通過完整 COCO5000 重驗（兩項 AP 差值 0）及固定8張含 person 圖片推論；尚未目視或驗證使用者實際影片。接續兩组至 E10，不改拓撲。詳見 [E8 結果與推論紀錄](<../../worklogs/2026-09-09-e8-backbone-inference.md>)。尚無同預算比較通過預設驗收門檻的 winner；以下為各歷史階段。

E5 兩臂已完成：QK overall／person=0.506623691／0.626800401，control=0.507063915／0.626509338，仍無通過驗收的候選。下一步從同一 QK E5 分成 narrow 與 layer8–10／22 周邊解凍的 late 組，先真實更新校準，再至共同 E8；推論拓撲不變、非 head BN 統計及 score／bias 固定。詳見 [E5 與 Backbone 適應紀錄](<../../worklogs/2026-09-09-e5-backbone-adaptation.md>)。下方 E3／E5 排程文字為歷史階段。

E3 兩臂已完成：QK overall／person=0.505981519／0.626092575，control=0.506635073／0.626101753，尚無增準。先延續至共同 E5 再決定是否解凍 attention 周邊／Backbone 後段；使用者已允許必要時修改 Backbone 並重訓，但尚未換架構。最新範圍、超參數與驗證見 [E3 分析紀錄](<../../worklogs/2026-09-09-e3-review-backbone-scope.md>)。以下 E3 排程、v1 patience 與四項 gate 文字屬歷史；目前只用 overall／person、patience 6，延續 optimizer／EMA，不重新 warmup。

使用者最新指定：本分支只以 COCO overall／person 決策，ball／bat 不再作停止或驗收 gate，也不追加其診斷。兩臂 E1 的小幅 overall／person 下降未達安全停止幅度，會保留 optimizer／EMA 接續至共同 E3，再依趨勢決定延長。詳見[續訓工作紀錄](<../../worklogs/2026-09-09-coco-person-continuation.md>)。下方較早的四項 gate 敘述保留為歷史，不作目前 queue 規則。

[分段診斷與最新基準](<../../worklogs/2026-09-09-attention-gradient-recovery.md>)。目前 FP overall/person=0.518019276/0.630794912，A0=0.506738574/0.626805274，B100=0.503589001/0.624111237。共同列兩項，不將 overall 提升、person 退化稱為成功；patience 僅在保護項皆過門檻且 overall 或 person 創新高時重設。

使用者補充歷史順序是先 BinaryQK 再 MASF，並授權必要時從較早起點分段重訓。已核對 A0 SHA256 正是正式 BinaryQK parent。B100 仍保留原始比較錨點；先做 FP／A0／B100 同口徑基準，再做不加 MASF 的 A0 原生 loss QK 梯度成對恢復，通過後才加入下述優化。不是從隨機權重訓練，也不是默默用 A0 取代 B100 成果。

兩個 A0 恢復 arm 均 fresh optimizer／EMA、10-epoch horizon、warmup1、patience4；只開 Q/K projection（LR5e-7）與 Detect head（LR2.5e-5），其餘參數固定，非 head BN statistics 固定。AdamW 與完整 COCO／physical32×4 沿用；唯一方法變因是精確二值前向的 surrogate backward。baseline 不修改；candidate 是本目錄獨立類別。每項 AP 相對 A0 退超過 0.005 即安全停止；run-local best 不自動接受。先 GPU smoke，再正式排程。即使梯度修復成立，仍須完整 AP 實測才能聲稱補回精度。

`recover_a0.py` 使用固定四個 microbatches 累積，非 upstream warmup 期間變動 accumulation；loss 是 native batch-sum。訓練狀態快照保存 optimizer／scaler／EMA／RNG，但尚未實作或宣稱完整中斷 resume；若中斷，先稽核快照與資料順序，不自動當作 exact resume。

## 來源與比較邊界

- 唯讀來源：`/home/uxin/yolo/yolo_achitechure/achitechure_1/final/`。
- B100 精確 ID：`full35-b-f10`，不是 retained A2。Float 與 Bit-True 檔案 SHA256 必須符合來源 `models.json`。
- 舊 B100 COCO internal AP50–95 為 0.503503；A0 為 0.506754。B100 舊 gate 是 rollback，使用它是本次明確指定，並非宣稱它已勝過 A0。
- 約 0.011641 的 BinaryQK 缺口來自更早 A-FINAL／B26-FP；不能直接套成 B100 的實測差值。Full35 Float 仍含 BinaryQK，Float／Bit-True 差異不等於 FP-QK／binary 差異。
- 新輸出一律放本目錄 `artifacts/`，程式放 `scripts/`。所有分支同起點且不覆蓋來源。

## 執行順序與門檻

| 順序 | 工作 | 預算／決策 |
| --- | --- | --- |
| P0 | B100 雙權重、圖、固定係數、資料與訓練狀態稽核；完整 COCO 基準 | CPU preflight → Bit-True／Float 驗證；先確認完整載入，不做 partial load |
| W | 原生 Detect loss 對照與 raw P3 HOG9 | control 5 epochs；HOG 最多 10、patience 4；相同 10-epoch scheduler 前綴、warmup 1 |
| R | layer17 單點 RepConv | 與同 parent 原生 5 epochs 比較；初始及 fuse 等價，無收益不加 layer20 |
| M | 現有 shared P3 MASF → P3 Detect 分支 | 先原位 alpha0 bridge，5 epochs；bridge 不過就停止換位，不直接搬已訓練權重 |
| Q0 | 固定 PoT 下分別介入 site10／22 score | 完整驗證，保留 bias／normalization；只是因果診斷，不冒稱純 FP teacher |
| Q1 | 一個合法 BinaryQK recovery challenger | 根據 Q0 與訓練梯度決定 matched 10 epochs；先完成梯度、儲存恢復、推論一致性契約，不繞過 baseline guard |
| Q2 | 單一 teacher loss／STE-window 或 B4 fixed groups | 僅相應誤差證據支持才追加；B4 無每圖 selector，但增加 partial-popcount 成本；不預先堆疊 |
| O | 有必要才延長訓練或比較 MuSGD | 新 parent 重新校準；不得搬用融合後的成功／失敗結論；不在加模組時同時換 optimizer |
| V | 最終單模組／組合驗證 | 完整 COCO、去 HOG、RepConv fuse、原始與候選同圖比較；只組合通過候選 |

首輪擬定 AdamW，Neck LR 1e-5、Detect head LR 2.5e-5，betas=(0.948,0.999)、eps=1e-8、weight decay=0.00027、clip=10。Backbone／attention／MASF 固定，shared BN statistics 固定，head BN train。HOG head LR 3e-4；μ 以 train-only 梯度校準至主梯度約 5%，允許 2–10%，校準更新丟棄。

imgsz=640、完整 COCO80 train118287／val5000；physical batch 先以 32 驗證，logical128=32×4，smoke 成功才啟動正式訓練。AMP FP16。新的 EMA 初始化與 loss schedule 必須先稽核並成對固定，不假設 B100 inference 檔具有完整 resume 狀態。

Bit-True COCO overall AP 為主選模；person／sports-ball／baseball-bat AP 為保護指標。候選相對 parent 及同預算 control 各保護 AP 不退超過 0.001，overall 至少 +0.001 才暫時接受；任何保護 AP 下降超過 0.005 暫停分析。不同早停預算不宣稱公平勝出。完整 val 每 epoch 執行，EMA 與 live 分開記錄。尚未建立新基準前不啟動訓練。

## 架構

```text
原本 B100（獨立 Detect）：
layer16：raw P3 → MASF → p3_shared ─┬─> layer17 → P4 → layer20 → P5
                                   └─> Detect([p3_shared,P4,P5])

條件式目標（非立即全部加入）：
layer16：raw P3 ─┬─> layer17 Conv／RepConv → P4 → layer20 → P5
                ├─> MASF → p3_det ───────────────┐
                └─> HOG9 → loss（僅訓練）        │
layer19：p4_raw ────────────────────────────────┤
layer22：p5_raw ────────────────────────────────┤
                         Detect([p3_det,p4_raw,p5_raw])
```

單任務不執行 Detect／Pose 衝突投影，不使用 joint score。第二輪創新、person-only 與另一個量化專案保持排除。棒球新驗證只能用 canonical bbat5-v1；舊 567 張 split 不沿用、不與 683 張結果混比。

GPU 子工作由 shell supervisor 阻塞等待，每次最多 600 秒；正常不讀 log 或查 GPU，退出即處理。沒有心跳證據時不宣稱可偵測所有 STALLED。使用者插入問題時保留子程序與可恢復事件記錄。

目前狀態以 `artifacts/` 的 preflight、metrics 與後續工作紀錄為準；本計畫不是已完成訓練的宣告。
