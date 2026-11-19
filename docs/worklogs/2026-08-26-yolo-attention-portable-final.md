# 2026-08-26：YOLO26m PWL Attention 可攜訓練包

## 任務與範圍

將 `yolo_attention_final/final/` 從只含最小推論 runtime 的交付目錄，改為可單獨複製、
安裝、設定 COCO2017、檢查、推論與啟動既有 LR sweep／staged recovery queue 的自包含訓練包；
另撰寫 Binary Q/K 與 Bit-True PWL 對外演算法文件。

本次處理的是 COCO80 Detect 專案。沒有使用或改動 BBAT5 v1、Pose／Detect Task View、
split、影像或標註，也沒有啟動新的 GPU 訓練。

## 變更內容與原因

- 將 `src/yolo_attention/` 的完整 production modules 同步至
  `final/yolo_attention/`，補齊 training、evaluation、queue、workflow、scope 與
  provenance 功能；原因是原本 `final/` 只能推論，訓練仍依賴上一層 repository。
- 將 15 份 training／evaluation／variant configs、`weights/v1-br-best.pt` 與
  `s0-phase-b-bittrue` immutable parent checkpoint/provenance 納入 `final/`。
  只保留重跑需要的 parent，沒有複製 2.6 GB 歷史 runs。
- 修改 `final/training_recipe.py`，把 project root 固定為 `final/` 本身，並新增：
  - `configure`：設定 COCO2017 root、device、batch、workers；
  - `doctor`：檢查必要檔案、資料目錄與三個 checkpoint SHA-256；
  - queue 初始化後禁止改寫設定，以維持 provenance。
- 新增 `final/pyproject.toml`、交付包 `.gitignore` 與
  `scripts/build_portable_final.sh`，讓完整 source 與必要資產可重建同步。
- 新增 `final/BINARY_QK_PWL_ALGORITHM.md`，說明 deterministic sign、clipped STE、
  XNOR＋popcount、identity／Walsh–Hadamard 雙 basis、power-of-two scale、
  decomposed 2D bias、Q8.8/UQ1.15 PWL interpolation、訓練／Bit-True 分流及已知限制。
- 同步更新子專案與 `final/` 的 README、TRAINING 文件與操作命令。

## 驗證方式與結果

- `PYTHONDONTWRITEBYTECODE=1 /home/uxin/yolo/.venv/bin/pytest -p no:cacheprovider tests -q`
  - 結果：`25 passed`。
- `/home/uxin/yolo/.venv/bin/ruff check .`
  - 結果：通過。
- `ruff format --check final/run.py final/training_recipe.py tests/test_portable_final.py`
  - 結果：3 個檔案格式正確。
- `bash -n scripts/build_portable_final.sh`
  - 結果：通過。
- 在 `final/` 執行 `python run.py train`
  - 結果：dry-run 成功、`missing_inputs=[]`，沒有啟動 GPU。
- 在 `final/` 建立唯讀 LR sweep state
  - 結果：19 jobs，`project_root` 正確指向 `final/`，graph validation 通過。
- 在 `final/` 執行 `python run.py check`
  - 結果：正式權重 SHA-256、YOLO26m scale m、80 classes、兩個 Attention sites、
    Bit-True PWL `[-10,0]`／20 segments／21 endpoints 全部通過。
- `doctor_report()`
  - 結果：三個 checkpoint hash 全部通過；portable 預設不包含 COCO2017，
    所以未執行 `configure` 前 `dataset_ok=false`，符合預期。
- isolated `configure_runtime` regression test
  - 結果：成功寫入 data root、device、batch、workers，且不改動正式資料集。

## 困難與解法

- 內建 `apply_patch` sandbox helper 因 loopback 權限失敗；改用相同 unified diff 的
  標準 `patch` 工具完成局部修改。過程產生的 `.orig`／`.rej` 已列入待授權清理清單，
  尚未擅自刪除。
- 系統沒有全域 `python`／`ruff` 指令；改用既有
  `/home/uxin/yolo/.venv/bin/python` 與 `ruff` 完成驗證。

## 未解事項或風險

- COCO2017 圖片與標註不包含在 154 MB 交付包；新機器仍須自行提供並執行
  `python run.py configure --data-root ...`。
- 本次只驗證訓練 queue 的結構與 dry-run，沒有重新啟動長時間 GPU 訓練。
- PWL numerator 是 Bit-True，但 denominator 仍是 exact floating-point reference；
  尚未完成 HLS、上板、功耗或實體 latency 測量。
- 可重建 cache 與 patch 暫存檔的刪除仍等待使用者依清理清單授權。
