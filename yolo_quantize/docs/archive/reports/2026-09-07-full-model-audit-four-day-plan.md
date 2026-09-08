# 全模型量化盤點與四天執行計畫

> 歷史盤點快照：本文的 V35 CPU、早期 recovery PTQ、3epoch 與 planned 狀態保留當時語意，不代表現行進度。最新以[比較條件釐清](../../reports/2026-09-07-evidence-consistency-boundaries.md)及[本輪五回合 QAT 結果](../../reports/2026-09-07-continuous-qat-recovery-results.md)為準。1,332 筆為權重投影量測，並非逐層 mAP。

詳細配套：[SD4／三元分布與既有實驗分析](../../reports/2026-09-07-weight-distribution-detailed-analysis.md)，涵蓋格式選擇原因、逐層證據及比較限制。

盤點日期：2026-09-07。範圍為 yolo_quantize 全資料夾功能與研究進度，深度核對 V35/V36 產物；未重跑所有歷史實驗。四天指後續 GPU 可用後約96小時工作窗口。

## 目前結論

V36 queue 已 complete，3 epochs QAT 真實完成。部署 Conv/Linear 權重已148/148覆蓋：133個W8、13個LS-SD4、1個W6、1個W4。124個activation quantizers使用qSiLU + LSQ+ A8。全權重fake quant已有可用結果，全模型純整數部署尚未完成。

| epoch（從0起算） | worst total mAP50 delta | worst total mAP50–95 delta | 搜尋deployment gate |
|---|---:|---:|---|
| 0 | -0.018468 | -0.024101 | 未通過 |
| 1 | -0.008347 | -0.011567 | 通過 |
| 2 | -0.009762 | -0.011451 | 通過 |

epoch 1相當於總下降0.835／1.157百分點；門檻分別1.5／4百分點，包含activation替換。所有16項指標由原始metrics減去plan釘住的accepted重算；不是平均值gate。epoch0為scale-only/blend=0訓練，epoch1起full weight fake quant。這是search結果，尚未做formal final評估與export重載accuracy parity。

數據、checkpoint hashes與圖：[交付目錄](../../../deliverables/full-model-audit-2026-09-07/README.md)。V36 run為`artifacts/runs/qat/v36-qsilu-full-coverage-short-v2/v36-qsilu-full-coverage-short-qat-v1-qat-seed1/`；plan在V36 queue的`short-qat-full-coverage-v3/generated/qat-plan.yaml`，兩者以SHA關聯。名稱best_joint不能取代實際export重載核對。

## 全資料夾在做什麼

| 路徑 | 用途與完成邊界 |
|---|---|
| src/yolo_quantize | Full35 adapter、量化格式、metric gate、CPU/PTQ/QAT/queue；有實作不代表每一分支已跑完 |
| tests | 數值、schema、runtime回歸；不能取代accuracy驗證 |
| configs | activation與版本化實驗規格；保留历史規格，執行以最新parent/hash為準 |
| artifacts/manifests | catalog、固定calibration/probe、parent血緣 |
| artifacts/reports | CPU/PTQ/re-gate機讀結果；主要研究證據 |
| artifacts/queues | 分階段計畫、選擇、status/events；V36已完成，四天新計畫尚未接成live queue |
| artifacts/runs | checkpoint、validation、inference；約45GiB量級，保留不可重建證據 |
| artifacts/datasets | BBAT5 canonical runtime view/cache；不是新資料版本 |
| artifacts/archives | 封存舊poly_shift/V30等線，不自動復跑 |
| docs/reports、docs/research | activation預選、Q3、文獻、PTQ、門檻與歷史研究報告 |
| docs/superpowers | V36設計與舊checklist；此次加上實際狀態註解 |
| docs/worklogs | 中文變更、困難、驗證紀錄 |
| scripts、deliverables | CPU準備與可重建老師版圖表；此次增加完整盤點附件 |
| quantize_spec.md、原始PDF | 早期方法/研究來源，不覆蓋實際Full35資料契約 |
| IMPLEMENTATION_PLAN.md、README.md | 規劃與入口；本次同步最新狀態 |
| PUBLICATION_MANIFEST.yaml | 歷史發行範圍；本次沒有新的commit/push |

精確檔案數/非symlink容量見folder-inventory.csv。未跟隨外部資料連結，亦未刪除任何檔案。上層Git有大量無關未追蹤子專案，本次不混入發行。

## 需求對照與重要更正

1. **逐層sensitivity**：148×9=1332格CPU weight NRMSE/cosine/bytes已做；GPU為backbone/neck/head各4種特殊格式及4種uniform，共24個cohort + 1 final。沒有十區全格式accuracy矩陣，更沒有148層逐一mAP矩陣。
2. **SD4與LSQ**：CPU fixed-sd4是靜態optimal scale，並沒有學習。QAT的13個FixedSD4 quantizers以LS-SD4方式訓練。固定尺度/學習尺度的公平配對仍缺。
3. **parent不一致**：CPU profile用V35 QAT epoch3，PTQ activation checkpoint卻仍是早期qSiLU recovery。原PTQ保留為跨parent排序遷移探索，不能解釋成V35 learned parent的隔離sensitivity。
4. **delta口徑**：先前special-neck-fixed-sd4的-0.0063745/-0.0297308是incremental；正確total為-0.0165743/-0.0405280，屬recover而非green。所有最終gate必須使用total。
5. **失敗歸因**：final混合PTQ同時增加特殊routes；uniform又建立於累積routes上。不能將reject全歸因W8，亦不能判定所有W6/W5或三元都不可用。
6. **QAT比較**：V36沿用外部V35 sham，是unpaired continuation，不是同parent paired增益。沒有新訓練baseline，符合使用者要求。
7. **監測**：曾有空run目錄/新增qparams錯誤；此次completion與3個epoch真實存在。NO_CHANGE不代表健康，monitor baseline若已error/complete必須立刻返回。
8. **activation**：qSiLU+A8為主線，Hardswish僅區域旁支；poly_quality排除、poly_shift封存。BinaryQK/MASF保留。A-SD4、額外activation bit、MuSGD/新KD不列四天必要路徑。

此次修正文檔中的誤述；parent loader、續跑與monitor實作修正列Day1，未藉整理報告重新開GPU。

## 十區可使用不同量化方式

| 區域 | V36目前權重路徑配置 |
|---|---|
| backbone_early | 21 W8 |
| backbone_deep | 21 W8 + 1 LS-SD4 |
| backbone_attention_safe | 7 W8 |
| neck | 31 W8 + 1 LS-SD4 + 1 W6 |
| masf | 1 W8 + 2 LS-SD4 |
| neck_attention_safe | 3 W8 + 2 LS-SD4 |
| detect_one2one_tower | 12 W8 + 6 LS-SD4 |
| detect_one2one_predictor | 5 W8 + 1 LS-SD4 |
| pose_one2one_tower | 23 W8 + 1 W4 |
| pose_one2one_predictor | 9 W8 |

這是已訓練配置，不是每區最優結論。先在同一parent上每次只改一區，然後鎖定backbone，在该結果上測neck，最後測head；不得直接拼接獨立winner。敏感層可回到原parent格式/經驗證W8，所有部署權重維持量化。

## 全模型完成定義

- 部署Conv/Linear為148路、22,571,840 elements，全量化覆蓋已達到；training-only 99 modules不屬推論目標。
- protected 4 modules、131,072 elements需逐一列出角色與precision，BinaryQK等不能從分母排除後宣稱一般INT全覆蓋。
- 124個activation quantizers不代表每個tensor都已量化。還需operator/tensor表：residual add、concat、attention、MASF、head decode、requantization、保留float運算。DFL/NMS依實際Full35圖核對。
- fake quant的float .pt不是packed整數引擎。儲存估算為sum(elements×bits/8)+scale/zero-point/packing metadata，另列bias、protected、alignment。延遲要backend實測。
- 精度為COCO80、COCO Person、BBAT box/pose及ball/bat分項，8項mAP50+8項mAP50–95；formal只给鎖定finalists。

## 四天計畫（96小時預算，尚未排進live queue）

| 天 | 工作 | 數量上限與交付 |
|---|---|---|
| Day1 | V36 export CPU reload/parity；確認既有候選accuracy，鎖定parent；統一CPU/PTQ/QAT來源；修monitor terminal/續跑；全圖precision coverage | 1個parent重載驗證、checkpoint manifest、十區/148層/activation/protected表、queue dry-run；工程4–8h，GPU預留2h |
| Day2 | 十區各自隔離測optimal Fixed-SD4、exact ternary、filterwise TWN、Paper-TWN；148路固定probe單層輸出sensitivity | 40格PTQ + 592格輕量probe；GPU預留8–12h。probe不稱mAP。最多2組×3 epochs QAT，優先LS-SD4及可恢復三元 |
| Day3 | backbone→neck→head逐區累積，每次在最新locked policy驗證；再W6/W5，W7/W4按需要fallback/擴展 | W6/W5 20格、W7/W4最多20格、累積最多20格；敏感cohort拆單層accuracy算在這些預算內；最多2組×3 epochs QAT |
| Day4 | 鎖定最多2個完整finalists、必要短QAT；formal、export parity、實際bytes、backend可支援部分的延遲與老師版報告 | 全四天新增QAT最多6組×3 epochs，當日最多2組；最後8h保留驗證/整理；交付完整manifest/區域格式/16項指標/未支援算子表 |

優先交付：全權重覆蓋且通過門檻的一個模型、十大區域sensitivity、精度與壓縮折衷、明確整數部署邊界。96小時不保證完成自製kernel或所有格式所有層QAT。

V36相鄰epoch時間約2845秒（47分鐘），以3 epochs含載入/validation估2.5–3.5h/組；6組預留15–21 GPU小時。PTQ先以Day1實測修正估算，工程/除錯另留24–36h，其餘是GPU共享緩衝。等待他人使用的GPU時間另計；逾期交付已完成矩陣，未跑的格明記pending。

### 訓練技巧與選擇規則

每組3 epochs，patience5，seed1，detect logical128/micro16，pose16，clip10，accepted augmentation，無新增雜訊。patience5在3epochs內不會提早停止，時間由epoch cap/預篩控制。AdamW沿用既有weight_decay0.00027、beta1=.948、beta2=.999、qparam_lr_ratio1；LR：backbone3.8e-7、neck1.9e-6、MASF3.8e-6、attention5e-8、兩head各5e-6。先固定超參數，只有數值/恢復問題才記錄有依據的調整。

只有同parent可恢復、有限數值且有壓縮價值的候選進QAT。recover不能鎖為最終winner，崩潰三元直接淘汰。固定SD4對照與LS-SD4要同parent/routes，才能研究學習scale效果。Hardswish最多2格區域旁支，替代既有擴展預算，不新增總量；MuSGD、A-SD4、額外KD/最終kernel優化留下一輪。

資料固定COCO `/home/uxin/yolo/coco2017.yaml`，BBAT5 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`；搜尋pose-search.yaml、formal pose.yaml。沿用joint loader整合任務，不新切30%資料、不改label/assignment、不混成新資料版本。

### 執行計畫與排程狀態

[four-day-plan.json](../../../deliverables/full-model-audit-2026-09-07/four-day-plan.json)是待實作DAG規劃，status=planned_not_enqueued。順序：parent audit → 統一來源preflight → 特殊格式獨立PTQ → 少數短QAT → uniform/逐區累積 → finalists → export/報告。每job需plan/parent/data/evaluator hashes、依賴、timeout及gate；失敗不自動跳過。

shell起始立即辨識terminal狀態，平時600秒讀小status，无變化由shell等待。error才喚回診斷；有效checkpoint續跑須檢查hash與配置，空失敗目錄以明確attempt ID記錄，不能靠不斷換vN名稱假裝修好resume。後續實作可直接依本計畫落地；本次沒有啟動新GPU或更改現有程序。

## 驗證、清理與限制

CPU報告腳本驗證plan/source/checkpoint hashes、148唯一path、1332有限CPU數據及3epochs×16 metrics，輸出CSV/JSON/PNG/PDF/SVG。詳細結果見[工作紀錄](../../worklogs/2026-09-07-full-model-audit-four-day-plan.md)。未重跑GPU accuracy與硬體測速。

finish-work規定先列清單再授權刪除；本次所有檔案保留，清理清單見交付README。沒有commit/push，也沒有聲稱新四天queue已啟動。
