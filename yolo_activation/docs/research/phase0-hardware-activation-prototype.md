# Phase 0：硬體友善 Activation 原型與驗證計畫

日期：2026-08-25
狀態：純函數／固定點／ONNX 原型已通過；尚未收到 baseline 資料夾，未做 detection 訓練或板上實測
證據邊界：本文的 operator count、曲線誤差與整數模擬不是 AP、latency、area、power 或新穎性證明

後續 manifest、安全替換與受限 Pareto placement 已整理為 [SIPA-BCSP 訓練架構](../../training/README.md)。

## 1. 本輪收斂結果

第一批不再把文獻版圖中的所有候選都升級成昂貴 baseline。正式模型到位後只保留下列五個明確
角色，其中既有函數 baseline 只有 SiLU 與 Hardswish：

| 角色 | 名稱 | 第一輪投入 |
| --- | --- | --- |
| Accuracy reference | `silu` | 重現交付 checkpoint 與原 recipe；所有 delta 的基準 |
| Hardware-neighbor baseline | `hardswish` | 唯一進入等成本 recovery 的既有硬體鄰近 baseline |
| Primitive floor | `relu` | 只做 zero-shot、export 與極簡原語下界；預設不做 recovery |
| Primary proposed | `poly_shift` | 先測；採 power-of-two threshold 與 dyadic/APoT 常數 |
| Accuracy ablation | `poly_quality` | 測量硬體限制相對較貼近 SiLU profile 的代價 |

PReLU、PWLU-lite、FReLU、Mish、GELU、ReLU6 與動態 activation 都留在研究版圖，不進第一輪完整
驗證。只有上述短名單被證據淘汰、或交付 backend 明確不支援時，才從 frontier 補位。

## 2. 硬體版的數學定義

延續受約束 integral-polynomial family：

\[
A_{T,a}(x)=\frac{x}{2}+H_{T,a}(|x|),
\qquad
H_{T,a}(u)=T R_a(\min(u/T,1))+\frac{(u-T)_+}{2}.
\]

硬體版固定為 \(a=9/2,T=8\)：

\[
R(z)=\frac94z^2-\frac{15}{4}z^3+\frac{11}{4}z^4-\frac34z^5.
\]

把常數 threshold 直接摺入係數後，在 \(0\le u\le8\) 可寫成：

\[
H(u)=\frac9{32}u^2-\frac{15}{256}u^3
     +\frac{11}{2048}u^4-\frac3{16384}u^5,
\]

而 \(u\ge8\) 時 \(H(u)=u/2\)。因此：

- `T=8` 的 clipping／range comparison 可直接使用二進位友善的固定閾值；
- 四個 polynomial 常數都是 dyadic 或 APoT，例如 `15/256` 可分解成四個 shift term；
- forward 需要四次資料相乘來建立 `u²,u³,u⁴,u⁵`；使用者已允許乘法，這些乘法不能假裝成
  shift/add；
- 六個固定常數乘法（四個係數及兩個 `1/2`）可在自訂整數 datapath 降成 shift/add；
- 不需要 `Exp`、`Sigmoid` 或 runtime `Div`，並保有解析式的
  \(A(x)-A(-x)=x\) 與有限範圍外 exact-ReLU tails。

`poly_quality` 則使用 \(a=4,T=109/16\)，保留一般 constant multiplier，作為 accuracy-oriented
對照。兩個 proposed profile 都沒有可訓練參數；profile selection 才是後續模型層級的靜態決策。

## 3. 實作介面與邊界

原型採純 in-process 模組，外部只需三個入口：

- `build_activation(name)`：建立新的無參數 PyTorch module；
- `default_validation_plan()`：取得固定且精簡的 baseline／candidate 分工；
- `validate_activation(name, ..., onnx_path=...)`：一次完成曲線、導數、解析性質、固定點與 ONNX
  graph 稽核。

整數 rounding、係數展開、operator proxy 與 ONNX 解析都藏在套件內。這讓後續模型 adapter 只需
依 manifest 把 eligible SiLU 替換成 registry module，不需要知道固定點細節。目前沒有 baseline
model graph，因此刻意不先實作 Ultralytics／YOLO 版本特定 adapter，也沒有猜測 layer mapping。

所有 activation 都是 parameter-free，`state_dict()` 為空。未來應先對原模型 strict-load 交付
checkpoint，再依 manifest 替換 activation；「可載入權重」不代表替換後精度不變。

## 4. 已完成的純函數驗證

環境為 Python 3.12.3、PyTorch 2.11.0+cu128；曲線使用 `float64`、`[-12,12]`、24,001 點均勻
網格。以下只比較函數形狀，不是模型資料分布：

| Activation | MSE vs SiLU | MAE vs SiLU | Max abs error | 最小輸出 | 導數範圍 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SiLU | 0 | 0 | 0 | -0.278465 | [-0.09984, 1.09984] |
| Hardswish | 0.00244951 | 0.0310944 | 0.142278 | -0.375000 | [-0.49967, 1.49967] |
| ReLU | 0.0131787 | 0.0685294 | 0.278465 | 0 | [0, 1] |
| `poly_shift` | 0.000183797 | 0.00953378 | 0.0269959 | -0.289889 | [-0.11229, 1.11229] |
| `poly_quality` | 0.0000857188 | 0.00653458 | 0.0210018 | -0.279040 | [-0.108, 1.108] |

`poly_shift` 的曲線 MSE 約為 `poly_quality` 的 2.14 倍，但仍只有 Hardswish 的約 7.5%；這只表示
它值得進入模型 pilot，不能推導 detection AP。

### 4.1 Q16.10 整數模擬

整數 emulator 使用 signed Q16.10 input/output、`int64` 中間值、round-to-nearest 與最後一步
saturation。結果如下：

| 項目 | 結果 |
| --- | ---: |
| 相對 float `poly_shift` 的 MSE | 0.295534 LSB² |
| 最大絕對誤差 | 2.22721 LSB |
| 離散 `A(x)-A(-x)=x` 最大誤差 | 0 LSB |
| 兩側 exact-ReLU tail 最大誤差 | 0 LSB |
| `[-12,12]` 網格輸出飽和數 | 0 |

這證明 rounding contract 在指定 Q-format 可重現；不代表 16-bit 是最終位寬。`int64` 中間值也不是
硬體資源結論，後續仍要做逐級 range analysis、word-length optimization、QAT 與 RTL/HLS synthesis。

### 4.2 ONNX graph 稽核

使用 opset 18 匯出：

| Activation | ONNX node 摘要 | `Exp/Sigmoid/Div` |
| --- | --- | --- |
| SiLU | `Sigmoid ×1, Mul ×1` | `Sigmoid` |
| Hardswish | `HardSwish ×1` | 無 |
| ReLU | `Relu ×1` | 無 |
| `poly_shift` | `Abs ×1, Clip ×1, Mul ×10, Add ×5, Sub ×1, Relu ×1` | 無 |
| `poly_quality` | 同 `poly_shift` | 無 |

ONNX 的十個 `Mul` 包含四個資料相乘與六個常數相乘；標準 ONNX graph 不會自行證明常數已降成
shift/add。真正宣稱 dyadic datapath 前，必須以 custom kernel、HLS/RTL 或目標 compiler IR 證明
constant lowering，並量完整 detector latency／memory／power。

## 5. Baseline 資料夾到位後的最小驗證漏斗

COCO2017 與 Canonical BBAT5 v1 是兩個獨立 Detect 任務，以下步驟各跑一份 checkpoint 與報表，
不得混合 raw AP：

1. **Contract gate**：核對 model source/config、checkpoint hash、dataset YAML/fingerprint、recipe、
   activation manifest 與 baseline metrics；缺件就停止，不猜設定。
2. **Baseline reproduction**：只重現 all-SiLU；先凍結可接受的重現誤差，再看 activation 結果。
3. **Cheap zero-shot**：同一乾淨 checkpoint 分別替換 Hardswish、ReLU、`poly_shift`、
   `poly_quality`，固定 BN 處理、imgsz、batch、subset 與 seed 1。
4. **等成本 short recovery**：預設只跑 Hardswish、`poly_shift`、`poly_quality`；ReLU 不進昂貴
   queue。先比較各資料集自己的 AP delta、收斂、NaN/Inf、activation distribution 與 export。
5. **有限 placement**：先測全網單一 activation；只有全網替換未過 gate、且 region sensitivity
   顯示互補時，才做 early/deep/neck/head 的 one-at-a-time。最終最多兩個 proposed profiles，連同
   必要保留的 SiLU/Hardswish，部署 graph 最多三種 activation kernel。
6. **Finalist**：seed 1 先決策；只有 Pareto finalist 或臨界結果在資源允許時補 seed 2。完整重訓
   或 QAT 只給最後 1–2 個 policy。

昂貴實驗的優先順序是 `SiLU reproduction → poly_shift → Hardswish → poly_quality`。若
`poly_shift` 已在預先凍結的 accuracy gate 內且部署 proxy 支配 `poly_quality`，就不再擴充更多
activation；若未通過，先看 region placement，而不是立即把整個文獻候選表加入 sweep。

## 6. 是否需要重訓

- 本輪純函數、整數與 export 驗證不需要訓練。
- 固定函數可直接載入原 convolution 權重做 zero-shot；但 activation 改變了特徵分布，正式選型
  至少要有相同 budget 的 short recovery。
- `poly_shift` 與 `poly_quality` 本身無參數，因此不要求為 activation 新增 optimizer state；模型
  其他權重仍需 recovery。
- 最後若採 integer/PTQ，至少要 calibration；若 AP 無法恢復，才啟用 QAT。只有最後 1–2 個候選
  才考慮完整 recipe／from-scratch，不為每個 smoke control 重訓。

## 7. 重現方式

```bash
/home/uxin/yolo/.venv/bin/python -m pytest

tmp_dir=$(mktemp -d -p /tmp yolo-activation-phase0.XXXXXX)
/home/uxin/yolo/.venv/bin/python scripts/validate_phase0.py \
  --onnx-dir "$tmp_dir/onnx" \
  --output "$tmp_dir/report.json"
```

2026-08-25 本輪結果為 `12 passed`；ONNX 使用 legacy TorchScript exporter，因環境未安裝新版
exporter 所需的 `onnxscript`，會出現兩則 deprecation warning，但 opset 18 graph 已成功載入與
逐 node 稽核。這是已知維護風險，不影響本輪 graph 內容。

## 8. 仍未解的風險

- 尚未收到 baseline 資料夾，沒有模型 AP、實際 activation 分布、layer sensitivity 或 recovery
  結果。
- 尚未指定板卡、runtime、compiler、precision、DSP/LUT/BRAM 與 latency/power budget，不能排序
  最終硬體效益。
- 現行 PyTorch／ONNX 是未融合 reference graph；四次資料相乘可能在已有 fused SiLU/HardSwish
  的 GPU、CPU 或 NPU 上更慢。
- `poly_shift` 的良好均勻曲線誤差可能不轉移到 COCO2017 或 BBAT5 的 empirical distribution。
- 1 seed、最多 2 seeds 只能支持探索性結論；新穎性仍需更完整引用鏈與專利檢索。
