# 2026-08-27：YOLO26m final 完整 workspace、架構與創新文件

## 任務與範圍

依使用者要求，把 `yolo_attention_final/final/` 整理成可直接交接的完整 workspace，
在單一 README 說明整體檔案架構、模組邊界、資料對接方式、安裝、推論與訓練入口，
並把 Binary Q/K、PWL 與相關設計創新整合成一份可供他人閱讀的正式文件。

本次只整理 COCO80 YOLO26m Binary Q/K＋PWL 專案，沒有啟動 GPU 訓練，也沒有改動
COCO2017 或 BBAT5 v1 的影像、標註、split 與資料集設定。BBAT5 v1 不是此 workspace
既有模型的資料契約，未混用或重新切分。

## 變更內容與原因

- 重寫 `final/README.md`：
  - 列出交付檔案樹、37 個 production Python 模組、12 份 training recipes、
    evaluation／variant 設定、9 個功能測試、CPU smoke 與三種 checkpoint 角色。
  - 用 Module、Interface、Implementation、Seam、Adapter 與 Depth 描述架構，
    說明 CLI、設定、模型整合、workflow、queue、artifact 與 evaluation 的責任邊界。
  - 定義傳入 COCO2017 的目錄契約、`configure` 行為、推論 source 契約、
    runtime 產物結構與 checkpoint hash／用途。
  - 補齊安裝、workspace 驗證、CPU smoke、推論、訓練預覽及正式執行指令。
- 重寫 `final/BINARY_QK_PWL_ALGORITHM.md`，作為單一對外創新與演算法文件：
  - 說明 deterministic Binary Q/K、identity＋Hadamard dual basis、
    per-head power-of-two scale、2D relative bias、Bit-True PWL、
    Float／Bit-True 分工、fail-closed 雙站點整合，以及 gate／rollback／provenance。
  - 寫出張量資料流、XNOR＋popcount 等價式、clipped STE、
    Q8.8／UQ1.15 逐段線性插值、儲存成本與實作對應檔案。
  - 將可負責任的 project-level 創新敘述與不可過度宣稱的邊界分開，
    不把尚未完成的文獻新穎性調查或硬體量測寫成既成事實。
- 將 `tests/` 與 `scripts/smoke_cpu.py` 納入 `final/`，並更新
  `scripts/build_portable_final.sh`，使重建後仍保有自包含測試與 CPU smoke。
- 更新 `final/pyproject.toml`，加入 `dev` 測試／格式依賴及 pytest、Ruff 設定。
- 修正 CPU smoke 的可攜匯入：以腳本位置推導 `final/` 根目錄並使用 variant 絕對路徑，
  避免誤載同一工作區其他專案的 `yolo_attention`。
- 格式化四個測試來源，確保建置腳本重建後的 final 仍可通過 Ruff。

## 驗證方式與結果

- `scripts/build_portable_final.sh`
  - 結果：成功重建 `final/`；shell 語法檢查通過。
- 在 `final/` 執行
  `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q`
  - 結果：`21 passed`。
- `ruff check --no-cache .`
  - 結果：通過。
- `ruff format --check --no-cache .`
  - 結果：55 個 Python 檔案格式正確。
- `python run.py check`
  - 結果：正式 checkpoint SHA-256、YOLO26m scale m、80 classes、兩個
    Attention sites 與 Bit-True PWL contract 全部通過。
- `python run.py train`
  - 結果：只進行 dry-run，完整 recovery recipe 可解析，沒有啟動訓練。
- `python run.py doctor`
  - 結果：三個必要 checkpoint hash 全部正確；因未放入外部 COCO2017，
    正確回報 `dataset_ok=false` 與缺少的 train／val images，exit code 1 符合設計。
- 從 `/tmp` 執行絕對路徑 `final/scripts/smoke_cpu.py`
  - 結果：CPU forward 成功，輸出兩個固定 Attention sites 與 tuple 結果。
- `python -m json.tool results.json`
  - 結果：machine-readable 結果格式正確。
- final 交付統計：37 個 production `.py`、9 個 `test_*.py`、12 份 training YAML，
  目錄大小約 154 MB；沒有 `.orig`、`.rej`、`.pyc` 或 cache 目錄。

## 困難與解法

- 初次 CPU smoke 誤載外層另一份 `yolo_attention/src`，該版本未執行 fixed-scale
  calibration，因此 fail-closed 機制拒絕 forward。改以 `scripts/smoke_cpu.py`
  所在位置建立明確的 workspace import root，並以絕對路徑載入 variant；從
  `/tmp` 重跑後成功。
- 內建 patch helper 在目前 sandbox 因 loopback namespace 權限失敗；改用
  `patch --no-backup-if-mismatch` 套用精確 unified diff，沒有產生新的
  `.orig` 或 `.rej`。

## 未解事項或風險

- 為避免複製大型資料，final 不包含 COCO2017 images／labels。新環境必須先安裝依賴，
  再執行 `python run.py configure --data-root /絕對路徑/coco2017`。
- 本次沒有重跑 GPU 訓練或完整 COCO2017 5,000 張驗證；正式數值沿用既有、
  已留存 provenance 的結果。
- Bit-True 目前精確模擬 PWL numerator，但 denominator 仍是 float reference；
  尚未完成 RTL／HLS、實測 latency／energy 與端到端 fully integer datapath。
- 創新文件描述的是本專案可驗證的設計整合與工程貢獻；若要做論文的全球首創或專利
  novelty claim，仍需另做正式文獻與專利檢索。
