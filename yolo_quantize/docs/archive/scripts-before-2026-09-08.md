# 整理前腳本說明快照（歷史）

`trace_continuous_precision.py`在CPU記錄兩任務ATen實際dtype與浮點部署边界；`run_continuous_special_queue.py`會啟動GPU工作，須既有使用者授權與preflight成功，先執行`validate_continuous_parent.py`搜尋精度重驗，再接40格PTQ，600秒更新status、完成後轉decision_required，尚不自動選QAT。

持續實驗P0：`preflight_continuous_parent.py`嚴格重載V36，預設CPU；僅在GPU空閒且明確授權下使用 `--projection-device cuda:0` 核對GPU匯出EMA投影。`profile_continuous_distribution.py`只讀checkpoint做148路兩view分布；`prepare_continuous_matrix.py`在parity通過後生成40格獨立PTQ並驗證schema，不自行開GPU。執行需 `PYTHONPATH=src` 與既有本機artifacts。監測入口為 `python -m yolo_quantize.blocking_monitor STATUS_JSON`，600秒靜默等待，只輸出狀態事件。

`report_weight_evidence.py`承接盤點表與V36 CPU profile，生成1332筆逐層CSV、十區誤差圖、詳細數據附錄及來源hash。執行：`/home/uxin/yolo/.venv/bin/python scripts/report_weight_evidence.py`；先執行下述盤點腳本更新來源表，不載入模型或使用GPU。

`report_full_model_audit.py`讀取既有V36完成/QAT metrics、plan、CPU profile與PTQ re-gate，驗證來源hash，輸出`deliverables/full-model-audit-2026-09-07/`的CSV/JSON/PNG/PDF/SVG。執行：`/home/uxin/yolo/.venv/bin/python scripts/report_full_model_audit.py`。不執行GPU或訓練；script與圖表可提交Git，checkpoint不隨報告發布。

`render_activation_preselection.py`只讀取既有`activation-smoke-v2.json`並輸出PNG、SVG、PDF，不使用GPU、不載入模型，也不執行validation或訓練。

`prepare_v1_v3.py`建立固定32/64 diagnostic manifest；指定active activation parent時，另建立unfused／BN-folded dual-view manifest與CPU靜態格式分析。`--activation-region REGION=ACTIVATION --view-only`可只重建regional policy parity，不重複相同checkpoint的weight-only分析。`poly_quality`只可透過`--historical-parent`重建凍結證據。腳本不呼叫CUDA、validation、QAT或訓練。

完整main profile每個parent約需10–12分鐘CPU；日常回歸使用tiny tensor pytest。`--profile full`只準備granularity／scale ablation介面，本輪沒有執行全模型full profile。

`yolo-quantize-search-validation`／`python -m yolo_quantize.search_validation`只執行reviewed search plan明列的accepted／matched／candidate；必須提供`--execute-reviewed-plan`，支援`--resume`與原子JSON。2026-09-03的v1 plan只授權qSiLU＋A8／`backbone_early`／W8，不會自動展開其他bit、region、QAT或formal validation。

圖表需求為Python 3.12、`matplotlib==3.11.1`與Noto Sans CJK字型。完整命令、字型路徑與限制見[子專案README](../../scripts/../README.md)。
