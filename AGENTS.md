## 使用者全域工作偏好

遵循 `/home/uxin/.codex/AGENTS.md`。目前採單一主代理模式，不啟動或委派子代理；技能中的分工步驟由主代理直接完成。此偏好取代先前的主動協作要求，並保留清楚簡潔的中文表達、適量驗證與自主完成任務規則。

最新變更見 [2026-09-08 停用子代理工作紀錄](docs/worklogs/2026-09-08-disable-subagents.md)。

## Agent skills

### Issue tracker

Issues and specs are tracked in GitHub Issues for `uxin0205-tech/yolo`. See `docs/agents/issue-tracker.md`.

### Domain docs

This repository uses a single-context domain documentation layout. See `docs/agents/domain.md`.

## PTQ、QAT、Training 與 GPU 工作監測

依全域規則，PTQ、QAT、training 及其他 GPU 工作期間每 600 秒監測一次，異常在任務範圍內自主處理，完成後直接接續分析；具體細節由各專案決定。規則變更見 [2026-09-08 工作紀錄](docs/worklogs/2026-09-08-gpu-monitoring-policy.md)。

## YOLO 研究：動手前先參考 optimize

依使用者 2026-10-18 明確要求，所有研究階段，以及階段內每項實作、方法修改、候選實驗與訓練啟動前，都必須先唯讀核對 `yolo_optimize` 的對應方法，將核對結果用於本輪決策。不得先自行選配方、實作或跑完實驗後才補找參考。

- 查閱對應程式、實際 run 設定、parent／checkpoint 血緣與已完成結果；不能只看摘要或摘取單一超參數。若 optimize 沿用上游子專案，繼續追查該階段的原始來源。
- 區分初次替換、恢復訓練與已訓模型的後期微調；核對可訓練層、LR、BN／EMA、loss scaling、batch／累積、dtype、預算、scale／bias、normalization，以及資料與 evaluator 是否適用於當前 parent。
- 在該階段 protocol／實驗紀錄中先寫明參考檔案與 run／checkpoint、已知結果、採用項目、差異及理由、驗證方式。已有核對資料可沿用，但每個新候選或新配方仍須確認其適用性；重要來源以 SHA 或版本追溯。
- 未找到對應方法或仍有關鍵差異未釐清時，先完成查核與必要診斷，明記缺口，不以猜測直接啟動長訓。新延伸方法須明確標示，不能冒稱 optimize 已驗證。
- GPU 工作前的說明須包含參考來源、本輪差異、預算與資源需求。這是代理必須完成的事前工作，不新增逐項向使用者請求確認的流程。

`yolo_optimize` 始終唯讀。參考方法仍須遵守使用者指定的候選、硬體友善契約、資料邊界及階段停止規則。詳見 [本次規則紀錄](docs/worklogs/2026-10-18-optimize-reference-before-work.md)。

依使用者同日後續指示，已確認錯誤、被取代且不再被使用的實作、配方與實驗產物應直接刪除，不另建 archive／backup。先記錄必要的失敗原因、小型數據與刪除清單，並核對 active process／外部依賴；不把錯誤 run 的 checkpoint 留作另一套可執行方案。此授權不涵蓋唯讀 optimize、正式資料或其他仍需使用的模型。見 [恢復重建與直接清理紀錄](docs/worklogs/2026-10-18-binaryqk-recovery-v2.md)。

## 第六階段：以最佳融合結果為目標

依使用者 2026-10-29 最新要求，目標是找出實際表現最好的 Detect–Pose 融合結果；optimize 的方法、配方與暫時門檻作為參考，不機械照搬。動手前仍須唯讀核對對應來源，依本輪結果自主決定階段內的必要診斷、恢復、降 LR、延長、訓練範圍與任務權重調整，不逐項索取確認。精度波動先查原因，不能只因一個參考停止條件就宣告整個研究失敗。

新配方在安全的段落邊界固定 parent、來源、差異理由、預算與選型；執行中的來源鎖不任意改寫。錯誤修正需記錄前後 SHA 與續跑狀態。最終如實報告各項 AP、與目標的差距及成本；較佳結果不被退化的後段覆蓋。確實無法改善、數值／資料／硬體契約問題或資源衝突時，說明證據及已處理內容。整個第六階段完成後停止，第七階段仍待指定。詳見 [本次決策紀錄](docs/worklogs/2026-10-29-stage6-best-fusion-objective.md)。

## 第八階段：訓練與部署算子分開定義

2026-11-04 最新確認：**簡化算子是部署／推論要求，不是 RepConv 訓練階段的強制要求。** 訓練可依原本 RepConv 流程使用常規算子；部署時才完成 folding 與 qSiLU、完整 PWL／reciprocal 等優化，並以最終部署版本驗收。2026-11-07 最新使用者已重新授權訓練並允許比較變體，解除先前暫停。五組政策見 stage8/protocol.json：Rep17／20／17+20 使用 qSiLU+Float PWL；另外只改 Rep17+20 activation 的 SiLU 候選，以及 ACB17+20。全部最終部署仍為 qSiLU／D1。每組5E，有依據才降 LR 延長；符合條件的 SiLU 候選另可3E硬體適應，總上限53E。完成本階段即停止，不再要求逐組啟動確認。

先前「Rep 訓練前向也必須使用 qSiLU／D1」的規定，依本次明確指示解除；不能再把部署限制強加於訓練。Rep17、Rep20、Rep17＋20 的最終部署仍須符合 qSiLU、D1 整數 PWL exp＋LUT256 reciprocal、B2 二值 QK、S1 Q10 scale／signed16 bias 等已選硬體友善要求。具體訓練算子、parent、BN、可訓範圍與配方須在新方案中明列，不能把需求澄清擴大為所有設定已獲批准。

訓練與部署各自記錄 manifest／backend、activation、轉換步驟與 checkpoint；先驗證相同算子下的 Rep folding 等價，再獨立驗證算子優化後的完整 AP。最終以部署版本的精度與成本選型，原生訓練模型的 AP 不能代替部署結果。其他剩餘浮點、非線性、除法、BN 與 head／decode 仍如實稽核；不得宣稱未量測的硬體加速。

已完成 Stage8 的小型結果與來源鎖保留為實驗事實；11 月 7 日直接清理無用舊權重與 resume，暫留最佳 folded Rep17＋20 的獨立交接包。新輪五組 CPU 與可攜驗證通過，依最新授權啟動 queue，Stage9 未啟動。見 [最新查核](docs/research/repconv-optimize-operator-review-2026-11-07.md)。最新界線見 [訓練與部署要求紀錄](docs/worklogs/2026-11-04-repconv-deployment-contract.md)；先前實際做法見 [Rep 訓練方式核對](docs/research/repconv-training-clarification-2026-11-04.md)。

## 第九階段：特徵圖研究的工作區

依使用者 2026-11-07 指定，第 9B 階段的特徵圖形狀研究使用新的 `yolo_actshape/`，包含長方形／二次冪尺寸、padding／分塊、相容驗證、精度與成本比較，以及有依據的必要適應。從第 8 階段最終選定成果繼承權重與部署契約；共用 Attention 仍集中於 `yolo_attention`。本次只指定資料夾，未授權啟動第 9 階段實驗；現有第 8 階段 queue 照原安排執行並於該階段完成後停止。見 [工作區入口](yolo_actshape/README.md) 與 [分工紀錄](docs/worklogs/2026-11-07-actshape-workspace.md)。

## 研究階段的 result 交接

依使用者 2026-10-28 後續要求，第六階段剩餘工作一次排成條件式 queue，授權在此階段內依既定規則接續 J1 延長、必要 J2／J3、驗證與交付，不逐段再次等待啟動。整個第六階段完成後仍停止，第七階段待指定。正在執行的配方與來源鎖定不得中途改寫。見 [本輪重建與 queue 紀錄](docs/worklogs/2026-10-29-stage6-clean-rebuild.md)。

依使用者 2026-10-28 要求，第六階段的 Pose／融合恢復須參照原 Pose 與 optimize 的完整分段訓練、warmup／cosine、跨段降低 LR 與 plateau 恢復流程。短預算結束是收斂檢查點；仍有改善時可在已授權階段內延長，停滯時先評估降 LR 恢復再停止，不直接把20輪當成整個融合的失敗結論。每段先固定parent、配方及預算；執行中的source lock與配置保持可追溯，新政策在安全的段落邊界啟用。詳見 [最新研究決策規則](docs/worklogs/2026-10-29-stage6-best-fusion-objective.md)。

依使用者 2026-10-19 指示，切換階段時將已選定的成果資料夾整理為所屬專案的 `result/stageN/`，下一階段從其中的權重、manifest、指標與 lineage 出發，在該階段的目標資料夾工作；不要求後續研究一直留在最初的資料夾。共用算子保留單一維護位置。

搬移前確認程序已停止及依賴，搬移後核對 checkpoint SHA、載入與設定繼承。必要歷史入口可保留相容符號連結，不複製整套成果或另建 archive／backup。進行中的 run 不搬移。每階段完成後仍先停止交付，等待使用者指定下一階段。見 [第 2 階段與交接紀錄](docs/worklogs/2026-10-19-binaryqk-stage2.md)。

## BBAT5 棒球資料集

- 所有新 BBAT5 detection、pose 與融合實驗一律使用不可變的 `/home/uxin/yolo/original/pose/derived/bbat5-v1/`。
- 正式 Pose YAML 是 `configs/pose.yaml`；正式 ball/bat Detect YAML 是 `configs/detect.yaml`；全域 machine-readable registry 是 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。
- `/home/uxin/yolo/original/pose/dataset/` 與 `detect_dataset/` 是唯讀原始／歷史來源，只用於稽核與重建，不得再作新訓練入口。
- 子專案 `artifacts/datasets/` 只可建立保留 bbat5-v1 assignment 與 labels 的可重建 runtime View，不得成為另一個資料版本。
- 後續凡是使用 BBAT5、BBT5 或棒球專用資料集的工作，不得自行重新切分、抽樣、替換來源或改動影像與標註，除非使用者明確授權建立新版本。
- 若路徑或設定有問題，先記錄問題並只修正必要的路徑或設定，不得用另建 split 規避。
- 完整規範與 GitHub snapshot 的發行邊界見 `docs/agents/bbat5-datasets.md`。
- 依使用者 2026-08-23 明確授權，GitHub 另保存 `original/` 的原始 Pose、歷史 Detect 與
  canonical lineage；這只增加可追溯來源，不把 raw/basic split 升格為正式訓練入口。
- `original/` 發布永久排除 weights、checkpoint、cache、run 與重複 archive；Git 中既有的
  `original/**/*.pt` 必須移除，本機權重不得因此刪除。

## 語言與工作紀錄

- 所有進度回報、研究報告、實驗結論、交付說明與困難說明一律使用中文；程式識別字、指令、檔案路徑及必要原文可保留原語言。
- 每次修改、實驗或驗證都必須留下中文工作紀錄，至少包含變更內容與原因、驗證方式與結果、遇到的困難及解法、未解事項或風險。沒有困難時也要明記「無」。
- 工作紀錄放在 `docs/worklogs/`，並同步更新 `docs/worklogs/README.md` 的索引；根層 `README.md` 必須持續提供資料集規範與工作紀錄入口。

## 第九階段：特徵圖研究與已授權 queue

依使用者 2026-11-08 明確要求，Stage8 最終採用 **qSiLU 訓練版 Rep17＋20 E1（`rep17_20`）**，不是 SiLU 訓練候選。交接包只保留此 folded 模型，另留最佳多分支 EMA；未採用權重與舊 optimizer resume 已直接清理。`achitechure_1` 因 optimize 外部依賴保留。

使用者已授權在 `yolo_actshape` 自主研究並排 GPU queue，後續再要求增加輪數。Stage9 已固定六組完整 E0，按事前規則最多選兩組，各 **10E＋半 LR 10E，總上限40E**；條件式校準、smoke、訓練、降 LR、完整驗證與交付不用逐段再次詢問。共同 parent 的 SHA 為 `2454548104ed64768eb9ecdb99d44804ee9d8e1e371e2a508cf052c0911fdd35`，部署繼承 qSiLU／D1／B2／S1／folded17+20。

執行中的來源鎖與配方不得任意改寫。異常先查核，在安全邊界記錄原因與新舊 SHA 後處理；不以重啟無限增加預算。GPU 每600秒監測、30秒檢查資源衝突；使用者希望長時間訓練穩定後退出 working，回報預估完成時間，由背景 supervisor 繼續監測。完成整個Stage9後停止，未授權啟動其他研究。詳見 [Stage9 工作紀錄](docs/worklogs/2026-11-08-actshape-stage9.md) 與 [固定 protocol](yolo_actshape/protocol.json)。

2026-11-08 後續使用者明確授權「都可以一起做做看驗證看看，在 GPU 上試」。Stage9 新增 256×512、384×512、512×768 各 10E＋半 LR 10E，及七組 canvas 的記憶體排列／分塊 GPU 等價性診斷。擴充上限 60E，連同初輪總上限 **100E**；三組均先完整 E0、梯度校準與 smoke，不僅因 E0 尺寸掉點就排除。初輪來源鎖保持，擴充位於 `stage9_extension/`、`extension/protocol.json` 與 `artifacts/shape-expanded-20261108-v1/`，必須等初輪完成並釋放同一 queue 鎖才能使用 GPU。整體入口為 `yolo_actshape/queue-status.json`，完成全部 Stage9 後停止。硬體目標尚未指定；GPU replay／時間、MAC／容量分析與 RTL／板端量測分開，不宣稱未做的加速。見 [擴充紀錄](docs/worklogs/2026-11-08-actshape-expanded-gpu.md)。


## Stage9 補充：768×512 同配置訓練（2026-11-12）

依使用者指定「768×512、使用一樣的訓練配置」與「train」，已授權新增 H×W=768×512 單一候選，從相同 qSiLU Rep17＋20 E1 parent 獨立開始，10E＋半 LR 10E，共新增 20E；原 100E 完成結果保持，累計上限 120E。實作與來源鎖位於 `stage9_portrait/`、`portrait/protocol.json` 及 `artifacts/shape-portrait-20261112-v1/`，沿用原訓練與 evaluator。任務權重使用同一 train-only 校準規則；只有原校準明確失敗時可按已固定政策補充一次 16+8 校準，不加 epoch。

此候選已於 2026-11-13 00:07 完成 20／20E、完整重載與可攜交接，Stage9 累計 120E 已停止。最佳為 adapt10 E5，十項 AP 均低於 512×768 最佳，原主線保持。2026-11-17 已核對實體權重與清理，見 [完成紀錄](docs/worklogs/2026-11-17-actshape-portrait-results.md)。每 600 秒監測、30 秒檢查外部 GPU；長訓穩定後退出 working，完畢交付 `result/stage9-portrait/` 並停止。原正式 handoff 不自動覆蓋，不增其他候選或第三段。詳見 [工作紀錄](docs/worklogs/2026-11-12-actshape-portrait-training.md)。

## 全輪完成與統一交接（2026-11-17）

依使用者要求已整理第 1～9 階段，根層 [result/](result/README.md) 為統一使用入口；正式模型維持 qSiLU 訓練版 Rep17＋20 E1／D1／原 rect640，Stage9 累計 120E 結束，沒有額外研究授權。完整結論、依賴與剩餘硬體限制見 [研究總報告](docs/research/yolo-research-summary-2026-11-17.md)。

本次刪除完成階段的 6 份 Stage1／2 optimizer resume 與 92 個快取，保留最佳權重與來源鎖。原 handoff 中的歷史 resume 路徑不代表仍可用，請讀各 result 的 `availability.json`；Stage3 KD resume 因 HOG source guard 依賴保留，MASF 候選／必要 resume 按既有要求保留。不可重新把來源鎖改成新狀態或把清理解讀為重訓授權。見 [整理紀錄](docs/worklogs/2026-11-17-integrated-research-handoff.md)。

## 11 月 19 日雙版本交接與舊參考退役

依 2026-11-19 使用者決定，固定高 512、寬 768 的形狀適應 E2 為硬體選擇，原 rect640 的 qSiLU 訓練版 Rep17＋20 E1 為精度選擇。兩份權重與各自的輸入契約見 [result](result/README.md)。新增的 [yolo_reshape](yolo_reshape/README.md) 導覽原處的 yolo_actshape/ 特徵圖實驗。

使用者已指定刪除 yolo_optimize/，先前唯讀規則記錄其退役前狀態。現存研究以各階段的 protocol、reference review、小型數據與 [退役前來源指紋](docs/research/data/1119/reference-index.json) 追溯。歷史訓練的 source guard 若需要已移除的檔案，重現前須取得原 SHA 相符的內容。完整研究已停止。[本次工作紀錄](docs/worklogs/2026-11-19-selections-and-cleanup.md)
