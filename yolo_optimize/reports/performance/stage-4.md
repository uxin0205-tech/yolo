# MASF：精度與部署成本

[完整量測口徑](<README.md>)／[精確 CSV](<comparison.csv>)。

## MASF

所有 AP 為既有完整驗證，範圍 0–1；— 表示此模型沒有該任務或此配對未驗證，不表示 0。Detect-only 的 COCO ball／bat 與 BBAT 二類 AP 不可混用。

| 案例／工作量 | COCO | person | COCO ball | COCO bat | BBAT box | BBAT Pose |
| --- | --- | --- | --- | --- | --- | --- |
| P3 control E5 / detect | 0.507974 | 0.627688 | 0.513173 | 0.480453 | 0.429281 | — |
| P3 shared E5 / detect | 0.507614 | 0.627748 | 0.512773 | 0.480411 | 0.428614 | — |
| P3 fork E5 / detect | 0.507963 | 0.627627 | 0.512289 | 0.480683 | 0.428782 | — |
| P3 control E8 / detect | 0.508267 | 0.627699 | 0.514006 | 0.480128 | 0.429300 | — |
| P3 bridge E8 / detect | 0.508212 | 0.627664 | 0.513148 | 0.480664 | 0.429390 | — |
| P2 control E5 / detect | 0.508420 | 0.627698 | 0.515551 | 0.480665 | 0.429989 | — |
| P2 MASF E5 / detect | 0.508271 | 0.627797 | 0.514300 | 0.480843 | 0.429814 | — |

| 案例 | BBAT ball box | ball Pose | bat box | bat Pose |
| --- | --- | --- | --- | --- |
| P3 control E5 | 0.298412 | — | 0.560151 | — |
| P3 shared E5 | 0.298586 | — | 0.558641 | — |
| P3 fork E5 | 0.297688 | — | 0.559877 | — |
| P3 control E8 | 0.298893 | — | 0.559707 | — |
| P3 bridge E8 | 0.298998 | — | 0.559782 | — |
| P2 control E5 | 0.298705 | — | 0.561273 | — |
| P2 MASF E5 | 0.297748 | — | 0.561879 | — |

### Params、Model size、MAC／FLOPs

Tensor MB 是本次 FP32 重建後參數＋buffers 的 payload，非已打包的低位元模型。來源檔 MB 是實際 checkpoint 大小；標記「snapshot」者含訓練狀態，不宜與推論檔直接比較。MAC／FLOPs 只包含下述可精確計數的 subtotal，二值位元乘積另列。

| 案例 | Params M | Tensor MB | 來源檔 MB／類型 | MAC G subtotal | FP FLOPs G subtotal | Binary bit-products M |
| --- | --- | --- | --- | --- | --- | --- |
| P3 control E5 | 21.897 | 87.815 | 341.761 / snapshot | 37.465 | 74.929 | 40.960 |
| P3 shared E5 | 21.973 | 88.125 | 343.013 / snapshot | 37.940 | 75.880 | 40.960 |
| P3 fork E5 | 21.973 | 88.125 | 343.013 / snapshot | 37.940 | 75.880 | 40.960 |
| P3 control E8 | 21.897 | 87.815 | 341.758 / snapshot | 37.465 | 74.929 | 40.960 |
| P3 bridge E8 | 21.973 | 88.125 | 343.011 / snapshot | 37.940 | 75.880 | 40.960 |
| P2 control E5 | 21.897 | 87.815 | 359.723 / snapshot | 37.465 | 74.929 | 40.960 |
| P2 MASF E5 | 21.973 | 88.125 | 360.975 / snapshot | 39.365 | 78.730 | 40.960 |

### Peak memory、CPU／GPU latency、target latency、energy/frame

B1／640／FP32、CPU 4 threads。延遲為 median；GPU 是 CUDA events 的模型核心時間。含模型內 decode／top-k，但不含外部 NMS、影像讀取與 H2D。GPU peak 為暖機後的 PyTorch allocated tensor 高水位，並非整張卡總使用量。

| 案例 | GPU peak MiB | CPU ms | GPU ms | target ms | GPU J/frame | target J/frame |
| --- | --- | --- | --- | --- | --- | --- |
| P3 control E5 | 447.9 | 222.156 | 25.506 | 未量測 | 4.4409 | 未量測 |
| P3 shared E5 | 448.8 | 224.457 | 25.323 | 未量測 | 4.4825 | 未量測 |
| P3 fork E5 | 455.0 | 222.531 | 25.480 | 未量測 | 4.4705 | 未量測 |
| P3 control E8 | 448.2 | 219.230 | 25.608 | 未量測 | 4.4236 | 未量測 |
| P3 bridge E8 | 455.5 | 222.794 | 25.549 | 未量測 | 4.4891 | 未量測 |
| P2 control E5 | 447.9 | 225.751 | 25.235 | 未量測 | 4.3895 | 未量測 |
| P2 MASF E5 | 450.6 | 256.962 | 25.606 | 未量測 | 4.5954 | 未量測 |

GPU energy/frame 是 NVML 整卡能量差，含桌面／idle 活動；不能視為純模型、整機或目標板能耗。target 欄位因未指定／連接目標設備而保留缺值。
