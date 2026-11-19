# 2026-10-18：B2 運算量與整網占比核對

## 任務與事前依據

使用者詢問第 1 階段減少多少運算量。本次只分析已完成模型，用 CPU 核對，沒有新訓練、GPU 工作或下一階段實驗。沿用已讀取的 YOLO 技能，並查閱 `yolo-export` 的成本與實測區分要求。

實作分析工具前，先唯讀核對 optimize 的 `experiments/benchmark/accounting.py`、`test_accounting.py`、`artifacts/accounting-v2/fp.json` 及 `reports/performance/comparison.csv`；原始 parent SHA 與本輪相同。採用其 Conv forward hook 與 `TorchDispatchMode` 矩陣 MAC 計數，使用 eval／no_grad／inference_mode=False，避免 dispatcher 漏算。相同口徑不包含 BN、activation、Softmax、decode、sort、memory traffic 等。

再讀取本輪 `basis.py`、`model.py`、已訓 B2 handoff、校準 site shapes 與本機原生 Attention。區分移除的 Q·K 浮點 MAC、新增 Hadamard／scale／加法，以及二值 XNOR／popcount；不把二值 bit-products 當等價 FLOPs，也不以減少 MAC 推論實測加速。

## 變更與驗證

新增 [獨立成本腳本](../../yolo_binaryqk/scripts/stage1_cost.py)、[小型數據](../../yolo_binaryqk/artifacts/stage1/formal-20261018-v2/operation-cost.json) 與 [成本報告](../../yolo_binaryqk/artifacts/stage1/formal-20261018-v2/COST.md)，原正式訓練程式與 checkpoint 不變。同步結果報告、README 及索引，沒有改變研究候選或訓練配方。

使用 `CUDA_VISIBLE_DEVICES=''` 執行腳本的 `--verify-models`，完成兩模型的完整 CPU 前向、有限輸出、相同參數、實際 site 形狀及 Conv／矩陣 MAC 核對；原始模型計數與 optimize 完全一致。兩處均為 20 × 20、4 heads、key_dim 32。整網浮點 MAC 由 37,505,638,400 降至 37,464,678,400，減少 **40,960,000（0.109210%）**。

B2 增加 81,920,000 bit-products、1,024,000 次 Hadamard 加減、2,560,000 次尺度乘法、1,280,000 次分支加法，省掉原生 102,400 次 Q scaling。扣除新增浮點算術後，浮點小計淨減少 77,158,400 FLOPs（0.102862%），仍未扣二值／整數成本；不能當總成本或速度降幅。

收尾跨至 2026-10-19：Ruff 通過；以 `.venv/bin/python -S yolo_binaryqk/scripts/stage1_cost.py` 在不載入第三方套件或 checkpoint 下重算成功。7 份文件共 115 個本機連結全部有效。沒有新增 GPU 使用或改動研究階段狀態。

## 困難與未解事項

現有 B2 為未打包的布林 reference，沒有 packed kernel 或硬體量測。總量口徑採本機未 fuse 前向，含現行 Detect head 執行路徑，不代表裁剪 one2many head 後的匯出模型。實測速度、功耗與面積尚未量測。

初次 Ruff 檢查指出 import 排序問題，僅調整排序後重查；運算公式與模型不變。其餘執行困難：無。本輪是推論成本分析，訓練反向仍有浮點代理，不能直接套用相同降幅。
