# BinaryQK：精度與部署成本

[完整量測口徑](<README.md>)／[精確 CSV](<comparison.csv>)。

## BinaryQK

所有 AP 為既有完整驗證，範圍 0–1；— 表示此模型沒有該任務或此配對未驗證，不表示 0。Detect-only 的 COCO ball／bat 與 BBAT 二類 AP 不可混用。

| 案例／工作量 | COCO | person | COCO ball | COCO bat | BBAT box | BBAT Pose |
| --- | --- | --- | --- | --- | --- | --- |
| 原生 FP Detect / detect | 0.518019 | 0.630795 | 0.526041 | 0.483490 | — | — |
| BinaryQK A0 / detect | 0.506739 | 0.626805 | 0.513110 | 0.495726 | — | — |
| B100 shared MASF / detect | 0.503589 | 0.624111 | 0.516234 | 0.469482 | — | — |

| 案例 | BBAT ball box | ball Pose | bat box | bat Pose |
| --- | --- | --- | --- | --- |
| 原生 FP Detect | — | — | — | — |
| BinaryQK A0 | — | — | — | — |
| B100 shared MASF | — | — | — | — |

### Params、Model size、MAC／FLOPs

Tensor MB 是本次 FP32 重建後參數＋buffers 的 payload，非已打包的低位元模型。來源檔 MB 是實際 checkpoint 大小；標記「snapshot」者含訓練狀態，不宜與推論檔直接比較。MAC／FLOPs 只包含下述可精確計數的 subtotal，二值位元乘積另列。

| 案例 | Params M | Tensor MB | 來源檔 MB／類型 | MAC G subtotal | FP FLOPs G subtotal | Binary bit-products M |
| --- | --- | --- | --- | --- | --- | --- |
| 原生 FP Detect | 21.896 | 87.811 | 44.256 / inference/source | 37.506 | 75.011 | 0.000 |
| BinaryQK A0 | 21.897 | 87.815 | 44.306 / inference/source | 37.465 | 74.929 | 40.960 |
| B100 shared MASF | 21.973 | 88.125 | 44.487 / inference/source | 37.940 | 75.880 | 40.960 |

### Peak memory、CPU／GPU latency、target latency、energy/frame

B1／640／FP32、CPU 4 threads。延遲為 median；GPU 是 CUDA events 的模型核心時間。含模型內 decode／top-k，但不含外部 NMS、影像讀取與 H2D。GPU peak 為暖機後的 PyTorch allocated tensor 高水位，並非整張卡總使用量。

| 案例 | GPU peak MiB | CPU ms | GPU ms | target ms | GPU J/frame | target J/frame |
| --- | --- | --- | --- | --- | --- | --- |
| 原生 FP Detect | 415.3 | 193.197 | 9.100 | 未量測 | 3.5326 | 未量測 |
| BinaryQK A0 | 447.9 | 217.529 | 25.387 | 未量測 | 4.4037 | 未量測 |
| B100 shared MASF | 450.0 | 223.887 | 25.607 | 未量測 | 4.4403 | 未量測 |

GPU energy/frame 是 NVML 整卡能量差，含桌面／idle 活動；不能視為純模型、整機或目標板能耗。target 欄位因未指定／連接目標設備而保留缺值。
