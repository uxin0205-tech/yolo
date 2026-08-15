# tests

現有 CPU tests 涵蓋：

- I/H/T5、FWHT、STE 與 XNOR-popcount。
- Dense/Decomposed bias。
- Exact/LUT/PWL/PoT/HardSigmoid/ReLU/Multimax row normalization 與 L3-A/B/C。
- BDCN fixed/learned codebook、D1→D2/R trained-bank state preservation、1/2-PoT、R0/R1/R2 數值範圍、fused bucket-PV 等價與雙 Attention 共享。
- Queue 的一般 catastrophic-mAP fail-closed gate，以及 R2 accuracy-bound diagnostic 只記錄負結果、不阻塞 R1 selection 的明列例外。
- per-head QKV gather/interleave、BN fold、Attention/C2PSA P0。
- YOLO26m 兩個 Attention path、fail-closed conversion 與 CPU forward。
- trainable scope（含 codebook-only）、score/probability Progressive、workflow、fake quant、registry、profiling 與 CLI。
- YAML recipe、artifact provenance 與 training runner wiring。
- Queue schema/store/lock、純 selection gate、動態 graph、標準 evaluation contract、analytical profiles、live backend injection、failure/retry 與 CLI execute gate。

~~~bash
python -m pytest -q
~~~

tests 驗證程式與數值 contract，不取代 COCO2017 正式 mAP。

Queue tests 全部使用 fake backend/checkpoint/metrics；測試命令不得加入真正的 `queue --execute`，也不得下載資料、權重或初始化 CUDA。
