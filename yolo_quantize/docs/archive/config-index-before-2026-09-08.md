- `v30-qsilu-all-ten-regions-w8-search-v1.yaml`：qSiLU+A8的148層／十區W8單格reviewed search；由hash-pinned V29 archive handoff後執行。
- `v30-qsilu-complete-quantization-lane-v1.yaml`：對qSiLU+A8另建全十區W8 parent、148×8 CPU profile（W7／W6／W5／W4、optimal Fixed-SD4與三種三元）、backbone→neck→head PTQ、LS-SD4／三元matched短QAT完整線；600秒event-only監測。
# 整理前實驗契約索引（歷史）

- `full35-integrated-quantization-roadmap-v1.yaml`：凍結總orchestration基線；V19已鎖定、V29 PTQ已完成，7-job short-QAT已在可續跑checkpoint暫停並轉交qSiLU。
- `v32-qsilu-regional-hardswish-conditional-v1.yaml`：Q3證據約束的條件旁支；qSiLU完成後先測neck-attention單區Hardswish，未promotion則停止。
- `v29-v19-progressive-ptq-queue-v1.yaml`：以V19 Epoch-5為parent，依backbone→neck→head執行十階段W4／SD4／TWN／exact ternary PTQ與條件式短QAT。
- `v28-paper-twn-progressive-region-plan-v1.yaml`：凍結的Paper-TWN逐區規劃基線；實際執行由V29取代，不原地改寫。
- `activation-smoke-v2.yaml`：模型、checkpoint、資料、校正、probe、矩陣、graph與產物契約。
- `activation-preselection-v1.yaml`：A8主線、A6／A7探索、停止項目與W-SD4／A-SD4公平比較設計。
- `full35-quantization-plan-v5.yaml`：現行總契約；八項mAP50、`-0.015`總下降且包含activation，記錄backbone early bit邊界與後續分階段執行。
- `full35-quantization-plan-v4.yaml`：凍結歷史總契約；保留原mAP50–95與`-0.04`門檻，不原地改寫。
- `v5-qsilu-backbone-early-bit-search-v1.yaml`：qSiLU＋A8／backbone early W7–W4候選validation，重用SHA驗證的accepted／matched；已完成。
- `weight-region-sensitivity-plan-v4.yaml`：上述W7–W4 diagnostic的窄範圍授權；已完成。
- `weight-region-sensitivity-plan-v5.yaml`：v5 stage 2窄範圍授權；只涵蓋qSiLU＋A8其餘九區的isolated W8 diagnostic，不含訓練或formal validation。
- `v5-qsilu-remaining-nine-regions-w8-search-v1.yaml`：v5 stage 2的九區candidate-only完整mAP50搜尋；重用SHA驗證的accepted／matched，不含訓練或formal validation。
- `v4-plus-prepared-plan-v4.yaml`：現行未執行矩陣；uniform V4為15格、regional V4H為6格＋最多2格條件式、V5為150格static universe。
- `weight-region-sensitivity-plan-v2.yaml`：現行weight runner契約；三個active parent、BN-folded View、固定diagnostic manifest，且`execution_authorized: false`。
- `weight-region-sensitivity-plan-v3.yaml`：2026-09-03窄範圍執行契約，只授權qSiLU＋A8／`backbone_early`／W8一格diagnostic；不授權其餘矩陣或訓練。
- `v4-qsilu-backbone-early-w8-search-v1.yaml`：上述單一cell的accepted／matched／candidate八指標search validation契約；已執行並依停止線停下。
- `full35-quantization-plan-v3.yaml`與`v4-plus-prepared-plan-v3.yaml`：2026-08-31凍結歷史。
- v1／v2更早版本保存當時含`poly_quality`的決策與hash，不再是執行入口。

所有歷史檔案保留血緣。使用者已於2026-09-03授權依v5分階段繼續，但每階段仍須有明確cell清單、hash、停止線與報告；不得因此直接展開formal validation或正式訓練。
# 最新持續規劃

`full-model-continuous-0907.json`定義每組5epochs、無四天硬截止、依精度／量化trade-off滾動追加及600秒shell monitor。它是planned_not_enqueued規劃，不是直接啟動GPU的CLI輸入。歷史配置不可原地改寫。
