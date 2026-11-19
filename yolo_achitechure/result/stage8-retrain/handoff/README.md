# 最終 qSiLU Rep17＋20 交接

最終模型為 rep17_20 E1，訓練與部署 activation 都是 qSiLU。部署為 folded Conv3×3＋qSiLU，保留 B2 二值 QK、S1 Q10 scale／signed16 bias、D1 整數 PWL exp＋LUT256 reciprocal。

本包只保留選定模型。在新的 Python 程序使用 `from load_model import load_model; model = load_model()`，輸入 RGB BCHW FP32、0–1，呼叫 `model(tensor, task="both")`；也可分別使用 detect／pose routing。前處理與外部評估資料不在本包內。

執行 `python -B validate.py` 可做 CPU 160／640 輸出一致性與錯誤 activation 拒絕檢查。載入器核對 runtime、權重 SHA、qSiLU、D1 與 Rep 位置，請連同 runtime／manifest 使用。

原比較指標在上層保留，未採用權重與舊 optimizer resume 已清理。多分支最佳 EMA 保存在上層 training/，與此 folded 部署檔分開。Conv／QKV／P·V／projection 等仍有浮點路徑，未做 RTL／HLS 或板端量測。
