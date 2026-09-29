# PWL configurations

- `exact.yaml`：固定 H/PoT/decomposed-bias 架構的 Exact Softmax reference。
- `float-pwl-8.yaml`：`[-8,0]`、16 段、$Δ=0.5$ 的 floating PWL。
- `bittrue-pwl-8.yaml`：同範圍的 Q8.8/UQ1.15 project bit-true PWL。
- `float-pwl-10.yaml`、`bittrue-pwl-10.yaml`：tail mass gate 失敗時才使用的
  `[-10,0]`、20 段版本；$Δ$ 仍為 0.5。

這些設定提交 Git，不含 checkpoint 或資料路徑。
