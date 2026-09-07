# 2026-09-01：完成 CPU integer boundary 與 exact W4／SD4 routing v3

## 變更內容與原因

- 新增`IntegerBoundaryContract` Module與Full35 graph Adapter，統一RNE、saturation、requant、Add／Concat及INT32 accumulator規則；特別分開一般zero-point與LSQ+ arbitrary offset，避免把activation量化語意建模錯誤。
- 新增`ExactWeightRouting` Module，要求qSiLU、Hardswish、poly_shift三個hash-bound parent、master／deployment雙view及四個公平format全部存在，才可產生routing。
- 新增`ExactWeightPreparation`與CPU CLI，以不可覆寫、原子寫入方式產生三份1,184-cell exact profile、integer boundary manifest、routing v3及CPU delivery。
- 保留舊`mse_grid_v1`報告與34層routing v2；新結果另立v1/v3 artifact，不回寫歷史hash。
- 原因是文獻／架構稽核指出十點absmax grid不是強baseline，而且第一個W8 GPU bridge前必須先說清楚hybrid-integer邊界。

## 驗證方式與結果

- 全程使用`CUDA_VISIBLE_DEVICES=-1`；未執行training、calibration、mAP validation、QAT或export。
- red→green tests覆蓋standard zero-point、LSQ+ arbitrary offset、RNE tie、saturation、Add／Concat對齊、INT32 bound、真實Full35 graph signature、三parent／雙view routing完整性、exact不劣於grid及artifact防覆寫。
- 真實graph結果：148 deployment weight sites、124 activation quantizers、21個reviewed core Concats、20個reviewed core residual Adds；148/148 W8/A8 MAC bounds通過INT32，最大`150,405,120`。
- 每個parent完成1,184筆，共3,552筆新static measurements；routing v3有36個cross-parent、cross-view stable candidates，比v2多2個且未移除舊候選。
- 最終完整CPU回歸為`100 passed in 11.08s`；ruff check／format通過。22份YAML以duplicate-key rejecting loader解析、20份JSON解析、delivery內5個artifact SHA重算一致；31份Markdown共120個本地連結缺失為0。

## 遇到的困難及解法

- 困難：LSQ+使用`q × scale + offset`，不保證offset能表示成integer zero-point。解法：reference Interface同時支援兩種affine形式；MAC靜態上界以raw codes保守估計，bias／padding correction明確延到calibrated scale／offset可用時處理。
- 困難：完整exact event sweep比小區域benchmark慢且記憶體較高。解法：三個parent分開、順序CPU執行，每份完成後原子封存；單份約64–68秒、峰值約1.26–1.29 GB，沒有平行載入三個模型。
- 困難：舊routing與新exact結論不能混寫。解法：v2保持歷史，v3明列source artifact／checkpoint SHA及selection rule，delivery另列停止線。

## 未解事項或風險

- 第一個需要GPU的工作是V4 W8 bridge：calibrate/freeze LSQ+ scale／offset、量Add／Concat saturation、layer-output NRMSE／Top-300 overlap與含COCO person的八指標validation；本輪依指示停在其前。
- fold-aware QAT契約尚未決定，只阻擋QAT。
- target hardware、integer reciprocal與native kernel尚未固定，禁止速度／能耗winner宣稱。
- 36層routing只是static候選，不是mAP winner或可直接export的mixed policy。
