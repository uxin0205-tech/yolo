# 推論：精度與部署成本

[完整量測口徑](<README.md>)／[精確 CSV](<comparison.csv>)。

## 推論

所有 AP 為既有完整驗證，範圍 0–1；— 表示此模型沒有該任務或此配對未驗證，不表示 0。Detect-only 的 COCO ball／bat 與 BBAT 二類 AP 不可混用。

| 案例／工作量 | COCO | person | COCO ball | COCO bat | BBAT box | BBAT Pose |
| --- | --- | --- | --- | --- | --- | --- |
| 原框＋KD 關鍵點 / both | 0.503885 | 0.625887 | — | — | 0.618008 | 0.890906 |
| qSiLU Pose one2one / pose | — | — | — | — | 0.618008 | 0.891329 |
| qSiLU Pose one2many core / pose | — | — | — | — | 0.618969 | 0.899451 |

| 案例 | BBAT ball box | ball Pose | bat box | bat Pose |
| --- | --- | --- | --- | --- |
| 原框＋KD 關鍵點 | 0.505192 | 0.859649 | 0.730823 | 0.922163 |
| qSiLU Pose one2one | 0.505192 | 0.859649 | 0.730823 | 0.923008 |
| qSiLU Pose one2many core | 0.494348 | 0.843451 | 0.743590 | 0.955451 |

### Params、Model size、MAC／FLOPs

Tensor MB 是本次 FP32 重建後參數＋buffers 的 payload，非已打包的低位元模型。來源檔 MB 是實際 checkpoint 大小；標記「snapshot」者含訓練狀態，不宜與推論檔直接比較。MAC／FLOPs 只包含下述可精確計數的 subtotal，二值位元乘積另列。

| 案例 | Params M | Tensor MB | 來源檔 MB／類型 | MAC G subtotal | FP FLOPs G subtotal | Binary bit-products M |
| --- | --- | --- | --- | --- | --- | --- |
| 原框＋KD 關鍵點 | 26.530 | 106.421 | 106.814 / inference/source | 47.883 | 95.766 | 40.960 |
| qSiLU Pose one2one | 23.532 | 94.360 | 106.826 / inference/source | 40.822 | 81.644 | 40.960 |
| qSiLU Pose one2many core | 23.532 | 94.360 | 106.826 / inference/source | 35.851 | 71.701 | 40.960 |

### Peak memory、CPU／GPU latency、target latency、energy/frame

B1／640／FP32、CPU 4 threads。延遲為 median；GPU 是 CUDA events 的模型核心時間。含模型內 decode／top-k，但不含外部 NMS、影像讀取與 H2D。GPU peak 為暖機後的 PyTorch allocated tensor 高水位，並非整張卡總使用量。

| 案例 | GPU peak MiB | CPU ms | GPU ms | target ms | GPU J/frame | target J/frame |
| --- | --- | --- | --- | --- | --- | --- |
| 原框＋KD 關鍵點 | 485.3 | 648.194 | 39.333 | 未量測 | 8.2794 | 未量測 |
| qSiLU Pose one2one | 472.4 | 572.230 | 34.650 | 未量測 | 6.9371 | 未量測 |
| qSiLU Pose one2many core | 472.4 | 507.763 | 31.073 | 未量測 | 5.9130 | 未量測 |

GPU energy/frame 是 NVML 整卡能量差，含桌面／idle 活動；不能視為純模型、整機或目標板能耗。target 欄位因未指定／連接目標設備而保留缺值。
