# BCND variant configs

本資料夾保存不覆寫舊結果的 BDCN 缺陷修正與穩定學習設定。

- `bdcn-v2.yaml`：從 A0（V1-BR）開始，使用 unconstrained global codebook
  訓練 10 epochs，作為已完成的負面對照。
- `bdcn-v2-r1.yaml`：沿用 v2 checkpoint，只把 denominator 換成 reciprocal LUT。
- `bdcn-v3-fixed.yaml`：64-level fixed exponential codebook 的 epoch-0 COCO
  control；不訓練。
- `bdcn-v3-learn.yaml`：從相同 A0 和 exponential initializer 開始，只學
  bounded monotonic codebook。
- `bdcn-v3-r1.yaml`：沿用 v3 learned checkpoint，評估 reciprocal LUT。

所有 v2/v3 設定共同使用 distance range $[0,8]$ 與 $L=64$。實際 step 為

$$
\Delta=\frac{8}{L-1}=\frac{8}{63}\approx0.126984.
$$

因此 $d=1.875$ 與 $d=8$ 不會再落入同一格。$d>8$ 仍會飽和到最後一格，
但 fixed initializer 的最後權重是 $e^{-8}\approx0.000335$，不是舊版
$e^{-1.875}\approx0.153$；evaluation 必須另外回報 overflow rate。

v3 learned table 使用每一步 log-ratio bound $\epsilon=0.01$：

$$
\log\frac{C_{l+1}}{C_l}=-\Delta+\epsilon\tanh(r_l).
$$

它在 $r_l=0$ 時精確等於 exponential table，同時限制訓練不能重新形成舊版
過度平坦的 attention tail。這些檔案提交 Git；執行產物不寫入此資料夾。
