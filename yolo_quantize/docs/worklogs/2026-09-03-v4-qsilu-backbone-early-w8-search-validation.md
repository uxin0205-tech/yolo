# 2026-09-03 V4 qSiLU backbone early W8 搜尋驗證

## 變更內容與原因

- 新增 [`weight-region-sensitivity-plan-v3.yaml`](../../configs/experiments/weight-region-sensitivity-plan-v3.yaml)，只授權 `qsilu_pq--lsq-plus-a8--backbone_early--w8-mse_grid_v1` 一格 GPU diagnostic；原本 `execution_authorized=false` 的 v2 歷史計畫未原地修改。
- 新增 [`v4-qsilu-backbone-early-w8-search-v1.yaml`](../../configs/experiments/v4-qsilu-backbone-early-w8-search-v1.yaml)，固定 accepted／matched／candidate 三角色、同一 evaluator/data contract、八指標與停止線。
- `WeightSensitivityStudy` 新增 exact-cell authorization 驗證；CLI 必須同時提供 reviewed plan 與 `--execute-reviewed-plan`，超出 cell 清單或 CUDA 不存在會在載入模型前失敗。
- 新增 [`search_validation.py`](../../src/yolo_quantize/search_validation.py)：可續跑、原子寫入三角色搜尋驗證；把 qSiLU parent effect 與 W8 incremental effect 分開，完成後套用八指標 gate。
- 新增 [`validation_source.py`](../../src/yolo_quantize/validation_source.py)：將已校正的 activation-output wrapper 與 BN-folded shared graph 正確投影到 Full35 官方 Detect／Pose task model。
- 新增 [`search_data.py`](../../src/yolo_quantize/search_data.py)：建立只含 symlink 的 BBAT5 v1 search runtime View，讓 Ultralytics cache 寫在子專案，避免修改 canonical dataset。
- 新增 CLI entry point `yolo-quantize-search-validation`。
- 以 TDD 補上 plan authorization、CUDA fail-closed、runner dispatch、runtime View assignment/idempotence/canonical lineage，以及 candidate W8 task materialization／context restore 測試。
- 實際執行一格 W8 diagnostic與一組 accepted／matched／candidate search validation；沒有訓練。

原因是舊單張 probe 只能證明 tensor 結構與有限值，不能回答 mAP；而直接拿 activation 報告與 weight static reconstruction 拼接又會違反耦合實驗要求。本輪用 matched qSiLU+A8 control 隔離 `backbone_early` W8 的額外影響。

## 驗證方式與結果

### CPU／靜態驗證

- TDD CUDA guard：先得到 `unrecognized arguments: --device 99` 紅燈；補上 fail-closed 後 1 passed。
- TDD runner seam：先因 `run_search_validation` 不存在得到紅燈；實作後搜尋模組 6 passed。
- TDD canonical lineage：真實 BBAT5 canonical file 是二段 symlink，舊檢查因 `resolve()` 誤判；新增紅燈測試後改成先驗證字面 canonical parent、runtime View 直接指回 canonical entry，lineage與idempotence 2 passed。
- official task materialization：qSiLU wrapper 在 Detect/Pose model 都存在、BN=0；`backbone_early` W8 weight 傳入 task model且 context 結束後原 shared weight 完整回復，1 passed。
- 完整 pytest：runner初版為`109 passed`；加入canonical lineage回歸後，最終為`110 passed in 15.20s`。
- ruff：`ruff check src tests` 通過；`ruff format --check src tests` 通過。
- BBAT5 runtime View：train 5,364、val 600、source-group overlap 0、assignment unchanged；manifest SHA-256 `2a5a9593d3703d86cfebb315fa5d2e5954fac1860d777cb0222d153416196ac5`。

### GPU diagnostic

- 輸出：[`weight-ptq-qsilu-backbone-early-w8-diagnostic-v3.json`](../../artifacts/reports/weight-ptq-qsilu-backbone-early-w8-diagnostic-v3.json)。
- 結果：`diagnostic_pass`；21 modules／1,218,240 weights；NRMSE `0.0054508`、SQNR `45.2708 dB`、cosine `0.9999853`、clipping `0.00037349`。
- 此結果明記 `selection_claim=false`，沒有單靠 probe 選 winner。

### GPU 八指標 search validation

- 輸出：[`v4-qsilu-backbone-early-w8-search-v1.json`](../../artifacts/reports/v4-qsilu-backbone-early-w8-search-v1.json)，SHA-256 `738f5103656deebcb248ad6e7d0770caa662fa191c90882f45f8f78e47bc6ad2`。
- accepted、matched、candidate 三角色均完整產出八項 mAP50–95。
- 最差 total delta：COCO box `-0.013504`，通過 `-0.04`。
- 最差 W8 incremental delta：BBAT ball box `-0.007147`，通過 `-0.01`。
- gate decision：`green`；只代表 search-eligible。
- 完成後 `nvidia-smi`：440 MiB used、0% utilization，GPU 已釋放。
- 詳細逐項結果與限制見 [`V4 W8報告`](../reports/2026-09-03-v4-qsilu-backbone-early-w8-search-validation.md)。

## 遇到的困難及解法

1. 執行環境的巢狀 `bwrap` 無法建立 loopback，所有一般 shell／`apply_patch` 一度回 `Failed RTM_NEWADDR`。解法：在核准範圍內以 escalated command 執行，檔案修改仍一律透過 `apply_patch`，沒有繞過工作區範圍或執行破壞性命令。
2. 子專案尚未 editable-install，直接 `python -m yolo_quantize...` 找不到 package。解法：沿用既有 runner 契約，明確設 `PYTHONPATH=src`；沒有安裝或升級套件。
3. 新的 package export 讓 `python -m yolo_quantize.weight_sensitivity` 出現 duplicate-import RuntimeWarning。解法：把 `Full35SearchValidationPlan` 與 weight sensitivity exports 都改為 lazy import，原 CLI 回歸測試轉綠。
4. BBAT5 canonical entries 本身是指向唯讀歷史來源的 symlink，對 file 呼叫 `resolve()` 會失去 canonical lineage。解法：用 lexical absolute path 驗證它位於 canonical `images/train`，runtime symlink 直接指向 canonical entry；新增二段 symlink 測試防回歸。
5. official `JointValidator` 會重新 materialize Detect/Pose model，若直接使用原 SourceBundle 會丟失 A8 wrapper。解法：新增 deployment validation source adapter，在 task template fuse 前投影 wrapper，再由官方 name mapping 複製 calibrated state；另測 W8 weight 同步與還原。
6. 雖設定 `save_coco_json=false`，Ultralytics 的 COCO final-val 邏輯會自動啟用 predictions JSON。這是官方 evaluator 既有行為，三角色一致；報告已分開記錄 requested setting 與 effective auto-export，未拿 console COCOeval AP 混入 Full35 八指標 gate。

## 未解事項或風險

- 本輪只驗證 qSiLU+A8／`backbone_early`／W8，不能外推到其他 region、W7–W4、SD4、LS-SD4、ternary 或其他 activation parent。
- W8 最敏感的 BBAT ball box 距 `-0.01` 門檻只剩 `0.002853`，低位元必須逐類別判定。
- search-val 不是 formal val；正式 finalist 尚未確認。
- accepted 與 qSiLU parent 使用不同 checkpoint，只有 candidate-minus-matched 能歸因給 W8。
- packed integer export、activation saturation、bias／padding lowering、native integer boundary與實際 FPGA／ASIC latency／power仍未完成。
- 沒有啟動 QAT、matched QAT sham、formal training、多 seed 或最終優化。
