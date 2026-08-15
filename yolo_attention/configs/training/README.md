# training recipes

這裡只定義訓練資源與 schedule，例如 epochs、batch、device、optimizer、seed、freeze scope 與 parent checkpoint；不放 basis、bias 或 normalization 演算法。

- `screening.yaml`：I/H/T5 各 10 epochs。
- `recovery-*.yaml`：winner 的 Direct／Progressive recovery。
- `normalization-recovery.yaml`：N1 attention-only PMP recovery。
- `bdcn-codebook.yaml`：D1 三候選各自一次完整 10-epoch codebook-only training。
- `bdcn-codebook-seed1.yaml`：只有排名接近時才執行的 10-epoch seed confirmation。
- `quantization-qat.yaml`：optional phase，未核准不得執行。

以 `python -m yolo_attention.cli train --variant ... --training ... --run-id ...` 預覽；只有明確加 `--execute` 才訓練。本目錄提交 Git。
