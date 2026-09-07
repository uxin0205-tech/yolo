# 2026-09-03 V5 mAP50 門檻與 backbone early bit 搜尋

## 變更內容與原因

- 依使用者更正，把現行硬門檻改為八項`mAP50`最差total delta `>= -0.015`，且total明確包含activation替換；舊`mAP50–95/-0.04`產物凍結保留。
- 新增`FULL35_MAP50_KEYS`及可選metric family，讓舊契約可重播、新v5預設使用mAP50。
- 新增SHA fail-closed的CPU re-gate，從既有raw metrics重算W8，不重跑validation。
- 新增硬門檻先行的accuracy／balanced／hardware Pareto selector；balanced只在距accuracy最差mAP50不超過`0.002`內壓縮。
- runtime weight adapter新增W8–W4、Fixed SD4及Paper-TWN統一seam；Fixed SD4使用exact scaled-codebook，Paper-TWN使用論文靜態式。
- 以TDD新增candidate-only resumable search race，重用已驗證accepted／matched，只執行W7、W6、W5、W4 grid與W4 exact候選。
- 新增現行總契約[`full35-quantization-plan-v5.yaml`](../../configs/experiments/full35-quantization-plan-v5.yaml)與詳細報告；v4不覆寫。

## 驗證方式與結果

- 門檻預設的red測試先確認舊`recovery_floor=0.06`會失敗；改為v5的`0.04`後，metric gate與Pareto測試`11 passed`。
- GPU前diagnostic五格皆完成且tensor結構有效；weight NRMSE依序為W7 `0.010941`、W6 `0.021577`、W5 `0.044955`、W4 grid `0.089494`、W4 exact `0.088849`。
- W8 raw metrics re-gate：最差total為COCO box `-0.011408`、最差W8 incremental為COCO person `-0.000781`，decision=`green`。
- 完整search validation：W7 `recover`（`-0.020950`）；W6 `reject`（`-0.044686`）；W5 `reject`（`-0.286769`）；W4 grid `reject`（`-0.971333`）；W4 exact `reject`（`-0.854778`）。
- Pareto eligibility只有W8，因此accuracy／balanced／hardware三個角色目前都指向W8。
- 執行使用完整5,000張COCO val與固定600張BBAT5 search-val，包含COCO person及BBAT ball／bat box／pose；沒有formal validation或訓練。
- 本輪程式變更前的完整CPU回歸為`125 passed`，ruff check及format皆通過；門檻預設修正後另跑目標測試`11 passed`，完整回歸會在本階段文件與下一區域runner完成後再執行。
- 完成後GPU為440 MiB used、0% utilization，沒有留下訓練程序。

## 遇到的困難及解法

1. 一般sandbox命令因nested `bwrap`無法建立loopback而失敗。解法：只在核准的workspace範圍以escalated command執行，所有檔案修改仍透過`apply_patch`。
2. 舊W8報告以mAP50–95作headline，若直接改寫會破壞來源血緣。解法：保留舊檔，以raw metrics檔SHA-256驗證後另產生mAP50 re-gate。
3. W4 exact雖改善重建和mAP，仍嚴重失敗。解法：保留為強baseline與反例，不將static改善誤稱為可部署結果，後續改做layer/group routing。
4. candidate validation逐格耗時且GPU曾需讓其他工作使用。解法：runner支援atomic write與resume，accepted／matched只在SHA與calibration identity一致時重用；完成後立即釋放GPU。

## 未解事項或風險

- qSiLU＋A8 activation已消耗COCO box約`0.011576`的總budget，因此W8全模型累積空間很小；isolated通過不等於組合通過。
- 其餘九區尚未完成W8 task sensitivity；目前不能宣稱全模型W8。
- W7只有`recover`，fold-aware QAT、matched sham及訓練超參數實作尚未完成。
- Fixed SD4／LS-SD4／ternary runtime或QAT尚未取得task mAP，不能依weight distribution直接選winner。
- uniform Hardswish、poly_shift與Q3 regional Hardswish仍需在新mAP50總門檻下重新建立matched結果。
- packed integer export、native kernel與目標硬體latency／power仍未完成。
