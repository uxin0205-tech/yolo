# 2026-09-07 全模型量化盤點與四天計畫

- 變更與原因：依使用者finish-work要求，整理全資料夾用途、V36完成狀態、十區異質量化覆蓋與96小時預算；新增可重建CSV/JSON/PNG/PDF/SVG、清理建議及入口更新。未更改訓練程式或啟動GPU。
- 實測：V36完成3 epochs，148路為133 W8、13 LS-SD4、1 W6、1 W4；epoch1/2相對accepted通過兩族deployment gate。全部16 metrics、source/plan/checkpoint hashes與coverage由CPU重新驗證。
- 更正：先前incremental誤稱total；CPU V35 parent與PTQ舊qSiLU parent不同；cohort不是完整逐層accuracy sensitivity；final reject不可歸因只有W8；NO_CHANGE不能當健康證明。
- 困難：sandbox bwrap loopback啟動失敗，改以經批准的唯讀命令及絕對路徑apply_patch完成；報告腳本初次lint有C408/格式問題，以formatter與等價字典轉換處理。
- 驗證：CPU報告生成成功；相關pytest與ruff最終結果見本檔後續驗證段。此次未跑全suite、GPU重評或延遲量測，因僅報告/計畫變更。
- 風險/未解：四天DAG是planned_not_enqueued，parent/monitor修正仍待Day1；protected四模組、activation完整邊界與整數kernel未證實全量化；正式驗證未做。輸出只是搜尋驗證證據。
- 清理與發行：清單列C01/C02；未授權刪除，所以所有檔案保留。無commit/push；上層Git無關修改不動。

## 最終驗證

- `/home/uxin/yolo/.venv/bin/python -m pytest -q tests/test_full_coverage_successor.py tests/test_qat_plan.py tests/test_qat_runtime.py`：32 passed，23.56秒。
- `/home/uxin/yolo/.venv/bin/ruff check scripts/report_full_model_audit.py`：All checks passed；format套用後重建報告。
- `report_full_model_audit.py`：3 epochs、每epoch16指標、148唯一paths、1332筆有限CPU數據驗證通過；四個full-resume checkpoint與plan sources hashes符合。
- GPU、全suite、formal與硬體測速此次未執行。沒有宣稱整個研究或四天新queue完成。

- 最終文件連結、DAG六組QAT預算、圖表存在性與148行sites CSV檢查通過；ruff check/format皆通過。git diff --check通過，但整個子專案未追蹤，不能將此當作新檔完整diff證明。
- 圖表生成成功，view_image視覺檢查因sandbox helper啟動失敗而未完成；未聲稱已逐像素檢查。
