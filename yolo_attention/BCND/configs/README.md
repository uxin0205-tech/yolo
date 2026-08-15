# BCND v2 variant configs

本資料夾只保留一條缺陷修正路徑，不做額外 levels sweep：

- `bdcn-v2.yaml`：從已完成 scale/bias 補償的 A0（V1-BR）開始，使用
  learned global codebook 訓練 10 epochs。
- `bdcn-v2-r1.yaml`：沿用前一個 checkpoint，只把 denominator 換成
  reciprocal LUT 作驗證。

兩者共同使用 distance range $[0,8]$ 與 $L=64$。
實際 step 由

$$
\Delta=\frac{8}{L-1}
$$

推導，因此 $\Delta=8/63\approx0.126984$，接近舊解析度 0.125。設定中的舊 `bdcn_step` 欄位為
向後相容保留；只要 `bdcn_distance_max` 存在，執行時以推導 step 為準。

這些檔案提交 Git；執行產物不寫入此資料夾。
