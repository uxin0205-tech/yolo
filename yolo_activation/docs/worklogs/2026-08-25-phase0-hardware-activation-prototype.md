# 2026-08-25：Phase 0 硬體友善 Activation 原型

## 任務與範圍

依使用者要求，在尚未收到 baseline 資料夾前，先縮減昂貴 baseline、加入允許乘法但常數路徑
硬體友善的 activation，並完成可執行的純函數、固定點與 ONNX 驗證。本輪沒有讀取或修改
COCO2017／BBAT5 影像與 labels，沒有建立 dataset view、啟動模型訓練、下載套件或宣稱硬體效能。

資料邊界維持：

- Canonical BBAT5 v1 ball/bat Detect：
  `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml`；未重切、抽樣或修改。
- COCO2017 COCO80 Detect：`/home/uxin/yolo/coco2017.yaml`；未與 BBAT5 混用 head、label space 或
  metrics。

## 變更內容與原因

- 新增 `pyproject.toml` 與 `src/activation_lab/`：提供 `build_activation()`、
  `default_validation_plan()`、`validate_activation()` 三個小型公開入口，將數學 profile、硬體
  proxy、固定點 rounding 與 ONNX 稽核封裝於內部。
- registry 限定 SiLU、Hardswish、ReLU diagnostic、`poly_shift`、`poly_quality`。既有函數 baseline
  只保留 SiLU 與 Hardswish；ReLU 預設不進 recovery，避免昂貴候選膨脹。
- 實作無參數 integral-polynomial float reference，先 clip `|x|` 再算 powers，避免 tail 上未採用
  的多項式溢位；所有 module 的 `state_dict()` 為空，便於未來先 strict-load checkpoint 再替換。
- 新增硬體主候選 `poly_shift`：固定 `a=9/2,T=8`，內區間係數為
  `9/32,-15/256,11/2048,-3/16384`。資料 powers 需四次乘法；常數乘法可用 dyadic/APoT
  shift/add 實作，沒有假稱所有乘法都消失。
- 新增 Q16.10 signed integer emulator：定義 round-to-nearest、算術右移、tail complementary
  rounding、int64 中間值與最終 saturation；這是 bit contract，不是最終 word length。
- 新增 opset 18 ONNX export／node audit，禁止 proposed graph 出現 `Exp/Sigmoid/Div`。
- 新增 `scripts/validate_phase0.py`，可輸出 machine-readable JSON；新增 12 個 pytest cases，覆蓋
  registry、無參數 checkpoint 介面、解析對稱／tail、固定點、ONNX 與 boundary autograd。
- 新增 `docs/research/phase0-hardware-activation-prototype.md`，記錄公式、精簡 baseline、實測數值、
  後續模型漏斗、重訓決策與證據邊界；同步根 README、研究索引、原始規格 decision overlay、
  數學設計與文獻 frontier。
- 新增 `.gitignore` 排除 Python／pytest cache 與生成的 ONNX binary。

## 驗證方式與結果

1. 使用 `/home/uxin/yolo/.venv/bin/python -m pytest`，結果 `12 passed`，耗時約 0.8 秒。
2. 使用 `scripts/validate_phase0.py` 在 `float64 [-12,12]`、24,001 點執行完整 registry；輸出與 ONNX
   只放在 `/tmp/yolo-activation-phase0.jauizc/`，沒有寫入資料集或 model artifacts。
3. `poly_shift` 相對 exact SiLU 的均勻曲線 MSE `0.0001837969`、MAE `0.00953378`、max error
   `0.0269959`；`poly_quality` MSE `0.0000857188`。這些只作函數 gate，不是 detection AP。
4. Q16.10 `poly_shift` 相對 float reference 的 MSE 為 `0.295534 LSB²`、max error
   `2.22721 LSB`；離散 `A(x)-A(-x)=x` 與兩側 exact-ReLU tail 均為 `0 LSB` 最大誤差，網格內
   saturation count 為 0。
5. ONNX graph：SiLU 為 `Sigmoid + Mul`；Hardswish 保留單一 `HardSwish`；ReLU 為單一 `Relu`；
   兩個 proposed profiles 為 `Abs/Clip/Mul/Add/Sub/Relu`，沒有 `Exp/Sigmoid/Div`。`poly_shift`
   的十個 ONNX `Mul` 包含四個資料乘法與六個常數乘法；標準 graph 本身不證明後者已降成
   shift/add。
6. Proposed float reference 的解析 symmetry 與 exact-ReLU tail 誤差均在 float64 rounding
   `5.4e-15` 以下；boundary 周邊輸出及 autograd 全為 finite。
7. 本地環境確認 PyTorch `2.11.0+cu128`、ONNX 可用、ONNX Runtime 與 `onnxscript` 未安裝；沒有
   為本輪下載新相依套件。

## 困難與解法

- 工作區 sandbox 持續出現 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`。必要 shell
  驗證改用經核准、限定於本工作區或 `/tmp` 的命令。
- `apply_patch` 對既有檔案的讀取連續受到同一 loopback helper 故障阻擋；新文件仍使用
  `apply_patch` 建立。既有文件更新在多次失敗後，改用經核准、逐段完全匹配的 `perl -0pi`
  原文替換，隨後以 `sed/rg` 唯讀核對，沒有廣泛重寫其他內容。
- PyTorch 2.11 對 `dynamo=False` legacy ONNX exporter 顯示兩則 deprecation warning；目前環境沒有
  新 exporter 所需的 `onnxscript`。本輪保留不需下載的 legacy path，成功產出並載入 opset 18
  graph；後續 baseline 環境到位時再遷移 exporter。
- 本地沒有 `ruff`，因此沒有虛構 lint 結果；以 pytest、`git diff --check`、來源複核與後續連結
  validator 代替。

## 未解事項或風險

- Baseline 資料夾、實際模型、checkpoint、activation manifest、訓練 recipe 與 baseline AP 尚未
  交付，故沒有模型整合、zero-shot、recovery 或逐區域結果。
- 目標板卡、runtime/compiler、位寬、DSP/LUT/BRAM、latency/power budget 未定；現有 operator
  proxy 不可稱為硬體加速或資源節省。
- Float reference 是未融合十個 `Mul` 的標準 ONNX graph；必須由 custom kernel、compiler IR 或
  RTL/HLS synthesis 證明 dyadic constant lowering，並量完整 detector。
- `int64` emulator 不等於最終 datapath；仍需 intermediate range、word length、overflow、PTQ/QAT
  與 calibration 驗證。
- 均勻網格曲線接近 SiLU 不保證 COCO2017 或 BBAT5 AP；兩個資料集仍須分開報告各自 baseline
  delta。seed 1、最多 seed 2 只能形成探索性證據。
