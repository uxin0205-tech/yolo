# 最小 checkpoint runtime

本套件只包含載入與執行最終 Bit-True checkpoint 所需的 source modules，刻意與完整 training／queue implementation 分離。

- `attention.py`：完整 Attention dataflow。
- `binary_basis.py`：Hadamard Binary Q/K score block。
- `normalization.py`：Bit-True Q8.8/UQ1.15 PWL normalization block。
- `projection.py`、`relative_bias.py`：Q/K/V projection 與 decomposed 2D bias。
- `config.py`：serialized model contract。
- `quantization.py`、`schedule.py`、`bdcn.py`：checkpoint import 相容所需的支援模組。

每個模組開頭都有簡短的中文區塊說明。這些檔案與 `src/yolo_attention/` 對應的 production modules 維持相同演算法；公開操作入口統一使用上一層的 `run.py`。
