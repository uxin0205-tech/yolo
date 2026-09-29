# 2026-09-05：V29低Token監測與qSiLU完整平行線

> 後續狀態：V29 matched-sham gate故障、修復與續跑證據見[同日修復紀錄](2026-09-05-v29-matched-sham-gate-repair.md)。使用者之後再次更改優先序；V29安全暫停與qSiLU接手見[封存與handoff紀錄](2026-09-05-v29-archive-and-qsilu-handoff.md)。本檔保留當時決策，不回寫歷史。

## 變更內容與原因

- 正在執行的V29 corrected queue保持不變、不發送signal、不重啟訓練。
- 將外部監測週期由300秒改為600秒，只讀`execution-status.json`與`execution-state.json`；正常事件不讀完整console。已啟動supervisor命令內的`--wait-seconds 300`在source queue已存在時不參與訓練監測，而V29 plan內的300秒只在arm間等待外部GPU時使用，因此不為改數值破壞現有provenance。
- 新增`configs/experiments/v30-qsilu-complete-quantization-lane-v1.yaml`與可執行第一格`v30-qsilu-all-ten-regions-w8-search-v1.yaml`，把qSiLU從原本「最後activation coupling候選」升為完整平行線：全十區W8 PTQ、paired all-W8 QAT與parent鎖定、148 paths×8 formats CPU profile（W7／W6／W5／W4、optimal Fixed-SD4、Paper-TWN v2、TWN-v3 filter-wise、exact ternary）、backbone→neck→head逐區PTQ、LS-SD4／三元matched短QAT。
- qSiLU線明定不得直接接用poly_shift V19的weight checkpoint。兩個activation必須各自形成locked weight parent，再在相同16項指標、資料與訓練預算下比較。
- 保留V4–V13 qSiLU結果作historical controls；新parent形成後的逐path結果仍須重驗，沒有覆寫舊artifact。

## 驗證方式與結果

- 唯讀確認現行supervisor仍為第一個`pose-tower-primary--exact_scaled_ternary`的matched sham，`execution-status.json`為`arm_started`；未中止或重啟。
- 唯讀確認qSiLU checkpoint存在且SHA-256為`7679186695317e431cd7deb17289f426f4b39b7a4993e4548e74f5ba2766190e`。
- 交叉核對V5：九區累積W8候選通過，但未包含`backbone_attention_safe`；該區isolated W8最差total mAP50 delta為`-0.017921761035523787`，屬recover而非green。因此V30先補全十區累積W8，不把九區結果誤稱為148層完整parent。
- YAML解析、唯一鍵、來源hash與連結已在本次CPU/config修改後統一驗證；不使用GPU。

## 困難與解法

- 困難：使用者要求降低Token，同時目前GPU訓練不可中止；直接修改正在執行的immutable V29 plan會造成來源雜湊漂移。
- 解法：不動現行runtime輸入，只把外部檢查改成600秒；所有新queue從一開始固定600秒。
- 困難：既有qSiLU資料看似已有all-W8，但實際累積candidate只有九區。
- 解法：明列十區coverage，先做單一全十區W8 search gate，通過green或recover才進paired QAT。
- 困難：受限sandbox的`apply_patch`因loopback namespace失敗；第一次替代寫入又因README標題sentinel不同而安全中止。
- 解法：先依規範嘗試`apply_patch`；確認為相同環境問題後，以精確sentinel、檔案存在性檢查及限定工作目錄的寫入完成。中止後先稽核已寫檔案再補齊，未觸及工作區外檔案。

## 未解事項或風險

- V29七個paired jobs尚未完成，現階段不能宣稱任何短QAT winner。
- V30已登錄但尚未啟動GPU；仍需完成可執行stage materialization與CPU preflight，且必須等V29 queue狀態為`complete`。
- qSiLU全十區W8可能無法由15 epoch QAT恢復到mAP50每項`-0.015`；若失敗，保留九區W8或FP/W8混合parent，不放寬總門檻。
- formal、20／60 epoch、multi-seed及硬體量測仍延後到finalists確認後；無資料集變更。
