# 融合：精度與部署成本

[完整量測口徑](<README.md>)／[精確 CSV](<comparison.csv>)。

## 融合

所有 AP 為既有完整驗證，範圍 0–1；— 表示此模型沒有該任務或此配對未驗證，不表示 0。Detect-only 的 COCO ball／bat 與 BBAT 二類 AP 不可混用。

| 案例／工作量 | COCO | person | COCO ball | COCO bat | BBAT box | BBAT Pose |
| --- | --- | --- | --- | --- | --- | --- |
| 舊 combine best_joint / both | 0.498022 | 0.620381 | — | — | 0.630036 | 0.903717 |
| 新 bridge J3 / both | 0.504242 | 0.626983 | — | — | 0.603892 | 0.886071 |
| Pose head 恢復 / both | 0.504242 | 0.626983 | — | — | 0.608773 | 0.886747 |

| 案例 | BBAT ball box | ball Pose | bat box | bat Pose |
| --- | --- | --- | --- | --- |
| 舊 combine best_joint | 0.507437 | 0.859909 | 0.752634 | 0.947526 |
| 新 bridge J3 | 0.486159 | 0.855779 | 0.721624 | 0.916362 |
| Pose head 恢復 | 0.492201 | 0.854418 | 0.725345 | 0.919075 |

### Params、Model size、MAC／FLOPs

Tensor MB 是本次 FP32 重建後參數＋buffers 的 payload，非已打包的低位元模型。來源檔 MB 是實際 checkpoint 大小；標記「snapshot」者含訓練狀態，不宜與推論檔直接比較。MAC／FLOPs 只包含下述可精確計數的 subtotal，二值位元乘積另列。

| 案例 | Params M | Tensor MB | 來源檔 MB／類型 | MAC G subtotal | FP FLOPs G subtotal | Binary bit-products M |
| --- | --- | --- | --- | --- | --- | --- |
| 舊 combine best_joint | 26.530 | 106.421 | 106.826 / inference/source | 47.883 | 95.766 | 40.960 |
| 新 bridge J3 | 26.530 | 106.421 | 106.825 / inference/source | 47.883 | 95.766 | 40.960 |
| Pose head 恢復 | 26.530 | 106.421 | 106.825 / inference/source | 47.883 | 95.766 | 40.960 |

### Peak memory、CPU／GPU latency、target latency、energy/frame

B1／640／FP32、CPU 4 threads。延遲為 median；GPU 是 CUDA events 的模型核心時間。含模型內 decode／top-k，但不含外部 NMS、影像讀取與 H2D。GPU peak 為暖機後的 PyTorch allocated tensor 高水位，並非整張卡總使用量。

| 案例 | GPU peak MiB | CPU ms | GPU ms | target ms | GPU J/frame | target J/frame |
| --- | --- | --- | --- | --- | --- | --- |
| 舊 combine best_joint | 472.0 | 254.588 | 30.158 | 未量測 | 5.8767 | 未量測 |
| 新 bridge J3 | 477.8 | 254.268 | 30.195 | 未量測 | 5.8376 | 未量測 |
| Pose head 恢復 | 477.8 | 256.804 | 30.115 | 未量測 | 5.8456 | 未量測 |

GPU energy/frame 是 NVML 整卡能量差，含桌面／idle 活動；不能視為純模型、整機或目標板能耗。target 欄位因未指定／連接目標設備而保留缺值。
