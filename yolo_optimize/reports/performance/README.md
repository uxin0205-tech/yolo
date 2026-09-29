# 全階段 AP50–95 與部署成本比較

更新：2026-09-12。26 組代表／配對，補入 AP50–95、Params、Model size、MAC／FLOPs、Peak memory、CPU latency、GPU latency、target hardware latency、energy/frame。這不是對全部 364 個本機模型逐一 benchmark，也沒有新增訓練或重測 AP。

## 量測口徑與證據強度

CPU 是 Intel Core Ultra 9 285K，GPU 是 RTX 5090；同一 PyTorch 環境、B1、640×640、FP32、固定合成輸入、CPU 4 threads，無 autocast／compile／額外 fuse，TF32 關閉。模型保持原始硬體友善 BinaryQK／PWL；qSiLU 是分段多項式軟體實作，不因此假稱 GPU 融合 kernel 或板端加速。

AP 重用各 checkpoint 的原完整 COCO val5000／BBAT5 v1 val683 結果。合成輸入只作算量／latency 測試，不更改資料 assignment，也不能取代真實影片 latency 或目視驗收。不同 task 的時間不能直接當演算法加速比：共享雙 head 每 frame 執行 Detect＋Pose，Detect-only／Pose-only 只做一個任務。

九項都列欄位，但 target latency／target energy 未量測：沒有指定／連接目標板、bitstream／runtime 與量測設備，不能拿 RTX 5090 遙測填 FPGA 欄位。也沒有整機插座功耗、FPGA rail power 或 CPU energy 的證據。

### 成本公式

Conv MAC = B×Hout×Wout×Cout×(Cin/groups)×Kh×Kw；矩陣乘法 MAC = batch×M×K×N。FLOPs subtotal = 2×浮點 MAC subtotal；整數 MAC／operations 與二值位元乘積另外記錄，不把 XNOR/popcount 算成普通浮點 FMA。

表內 MAC subtotal 為 Conv＋浮點／整數矩陣乘法，沒有計入 BN、activation、PWL／reciprocal、pooling、decode／top-k、整數 reductions、NMS、記憶體流量。它是明確範圍的運算估算，不是完整全算子 FLOPs，也不是硬體 cycle 數。相同 MAC 不代表相同 latency；qSiLU 的額外算子主要就在此 subtotal 之外。

Model size 同時提供實際來源檔 bytes 與 FP32 tensor payload；snapshot 包含 optimizer／EMA 等，不能用檔案大數倍推論部署模型也大數倍。BinaryQK 只二值化 attention 的 Q/K 運算，不是整個 YOLO 權重已壓成 1 bit。Params 計入仍註冊的雙分支權重，即使推論只執行其中一條。

### 時間、記憶體與能量

CPU 暖機 2 次＋10 次量測，GPU 暖機 10 次＋50 次量測；median／P90／全部樣本在 JSON，另存 GPU synchronized wall time。端到端 pipeline 的影像解碼、resize、H2D、外部 NMS 不在 core latency。one2many 比較尤其不能省略這項限制。

GPU peak allocated／reserved 在暖機後重置；主表使用 allocated，包括常駐模型與輸入，不等於 nvidia-smi 所見全卡顯存。CPU RSS 是 worker 全生命週期高水位，含載入與模型副本，只供記憶體稽核，不與 GPU activation peak 相加。

Energy/frame = (NVML end_mJ−start_mJ)/1000/frames。每案例 3 個至少 2 秒、至少 10 frames 區段取 median，不使用 TDP×latency。整卡 telemetry 包含 idle／桌面工作，沒有扣 idle、沒有整機功耗，也沒有獨占 GPU 或鎖定時脈；小幅差異應視為本次觀察，不作節能定論。CPU／GPU 都只有單機一次順序測試，非多輪隨機排序的統計保證。

### 量測修正

融合模型的 head decode cache 在 CPU→GPU 切換時未跟隨普通 module buffer，已於量測工具移動 cache 並清除 shape，短測通過，沒有改權重。

初版 dispatcher 在 inference_mode 下漏算矩陣乘法，原結果保留但不作 MAC 正文依據；報告統一使用 accounting-v2 的 no_grad CPU 計數，已通過已知尺寸浮點／整數矩陣與分組卷積的 3 項測試。latency／energy 仍在 inference_mode 下量測。

CPU 算量稽核曾與部分時延量測重疊。依工作時間保守圈定 j3_joint、pose_recovery、silu_e9、qsilu_e2，這 4 組另以無並行稽核的 isolated-recheck-v1 補測，其他成功量測不重跑。所有原紀錄保留；CSV 明列採用的來源。

## 原始證據

[精確比較 CSV](<comparison.csv>)／[比較 JSON](<comparison.json>)／[來源與原始量測](<../../experiments/benchmark/README.md>)。完整逐次 samples、每個能量區段與 checkpoint SHA 均可追溯。

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

## HOG

所有 AP 為既有完整驗證，範圍 0–1；— 表示此模型沒有該任務或此配對未驗證，不表示 0。Detect-only 的 COCO ball／bat 與 BBAT 二類 AP 不可混用。

| 案例／工作量 | COCO | person | COCO ball | COCO bat | BBAT box | BBAT Pose |
| --- | --- | --- | --- | --- | --- | --- |
| native control E3 / detect | 0.506052 | 0.626895 | 0.518051 | 0.484365 | — | — |
| HOG E3（部署移除輔助頭） / detect | 0.505733 | 0.626259 | 0.513622 | 0.486851 | — | — |

| 案例 | BBAT ball box | ball Pose | bat box | bat Pose |
| --- | --- | --- | --- | --- |
| native control E3 | — | — | — | — |
| HOG E3（部署移除輔助頭） | — | — | — | — |

### Params、Model size、MAC／FLOPs

Tensor MB 是本次 FP32 重建後參數＋buffers 的 payload，非已打包的低位元模型。來源檔 MB 是實際 checkpoint 大小；標記「snapshot」者含訓練狀態，不宜與推論檔直接比較。MAC／FLOPs 只包含下述可精確計數的 subtotal，二值位元乘積另列。

| 案例 | Params M | Tensor MB | 來源檔 MB／類型 | MAC G subtotal | FP FLOPs G subtotal | Binary bit-products M |
| --- | --- | --- | --- | --- | --- | --- |
| native control E3 | 21.897 | 87.815 | 427.206 / snapshot | 37.465 | 74.929 | 40.960 |
| HOG E3（部署移除輔助頭） | 21.897 | 87.815 | 427.226 / snapshot | 37.465 | 74.929 | 40.960 |

### Peak memory、CPU／GPU latency、target latency、energy/frame

B1／640／FP32、CPU 4 threads。延遲為 median；GPU 是 CUDA events 的模型核心時間。含模型內 decode／top-k，但不含外部 NMS、影像讀取與 H2D。GPU peak 為暖機後的 PyTorch allocated tensor 高水位，並非整張卡總使用量。

| 案例 | GPU peak MiB | CPU ms | GPU ms | target ms | GPU J/frame | target J/frame |
| --- | --- | --- | --- | --- | --- | --- |
| native control E3 | 448.2 | 229.076 | 25.158 | 未量測 | 4.4035 | 未量測 |
| HOG E3（部署移除輔助頭） | 447.9 | 220.623 | 25.480 | 未量測 | 4.4904 | 未量測 |

GPU energy/frame 是 NVML 整卡能量差，含桌面／idle 活動；不能視為純模型、整機或目標板能耗。target 欄位因未指定／連接目標設備而保留缺值。

## RepConv

所有 AP 為既有完整驗證，範圍 0–1；— 表示此模型沒有該任務或此配對未驗證，不表示 0。Detect-only 的 COCO ball／bat 與 BBAT 二類 AP 不可混用。

| 案例／工作量 | COCO | person | COCO ball | COCO bat | BBAT box | BBAT Pose |
| --- | --- | --- | --- | --- | --- | --- |
| Conv control E4 / detect | 0.508046 | 0.627352 | 0.513375 | 0.480253 | — | — |
| RepConv17 E4（已折疊） / detect | 0.508098 | 0.627501 | 0.513427 | 0.479946 | — | — |

| 案例 | BBAT ball box | ball Pose | bat box | bat Pose |
| --- | --- | --- | --- | --- |
| Conv control E4 | — | — | — | — |
| RepConv17 E4（已折疊） | — | — | — | — |

### Params、Model size、MAC／FLOPs

Tensor MB 是本次 FP32 重建後參數＋buffers 的 payload，非已打包的低位元模型。來源檔 MB 是實際 checkpoint 大小；標記「snapshot」者含訓練狀態，不宜與推論檔直接比較。MAC／FLOPs 只包含下述可精確計數的 subtotal，二值位元乘積另列。

| 案例 | Params M | Tensor MB | 來源檔 MB／類型 | MAC G subtotal | FP FLOPs G subtotal | Binary bit-products M |
| --- | --- | --- | --- | --- | --- | --- |
| Conv control E4 | 21.897 | 87.815 | 340.958 / snapshot | 37.465 | 74.929 | 40.960 |
| RepConv17 E4（已折疊） | 21.897 | 87.815 | 342.029 / snapshot | 37.465 | 74.929 | 40.960 |

### Peak memory、CPU／GPU latency、target latency、energy/frame

B1／640／FP32、CPU 4 threads。延遲為 median；GPU 是 CUDA events 的模型核心時間。含模型內 decode／top-k，但不含外部 NMS、影像讀取與 H2D。GPU peak 為暖機後的 PyTorch allocated tensor 高水位，並非整張卡總使用量。

| 案例 | GPU peak MiB | CPU ms | GPU ms | target ms | GPU J/frame | target J/frame |
| --- | --- | --- | --- | --- | --- | --- |
| Conv control E4 | 448.2 | 223.729 | 25.367 | 未量測 | 4.4268 | 未量測 |
| RepConv17 E4（已折疊） | 447.9 | 220.695 | 25.490 | 未量測 | 4.5229 | 未量測 |

GPU energy/frame 是 NVML 整卡能量差，含桌面／idle 活動；不能視為純模型、整機或目標板能耗。target 欄位因未指定／連接目標設備而保留缺值。

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

## Activation

所有 AP 為既有完整驗證，範圍 0–1；— 表示此模型沒有該任務或此配對未驗證，不表示 0。Detect-only 的 COCO ball／bat 與 BBAT 二類 AP 不可混用。

| 案例／工作量 | COCO | person | COCO ball | COCO bat | BBAT box | BBAT Pose |
| --- | --- | --- | --- | --- | --- | --- |
| SiLU E9 / both | 0.504309 | 0.626783 | — | — | 0.612832 | 0.889444 |
| qSiLU E2 / both | 0.503885 | 0.625887 | — | — | 0.618008 | 0.891329 |
| Hardswish zero-shot / both | 0.401393 | 0.543653 | — | — | 0.462790 | 0.803501 |
| PolyShift zero-shot / both | 0.500713 | 0.624503 | — | — | 0.592422 | 0.876398 |

| 案例 | BBAT ball box | ball Pose | bat box | bat Pose |
| --- | --- | --- | --- | --- |
| SiLU E9 | 0.493900 | 0.854713 | 0.731765 | 0.924174 |
| qSiLU E2 | 0.505192 | 0.859649 | 0.730823 | 0.923008 |
| Hardswish zero-shot | 0.357768 | 0.756817 | 0.567812 | 0.850185 |
| PolyShift zero-shot | 0.468226 | 0.836729 | 0.716618 | 0.916066 |

### Params、Model size、MAC／FLOPs

Tensor MB 是本次 FP32 重建後參數＋buffers 的 payload，非已打包的低位元模型。來源檔 MB 是實際 checkpoint 大小；標記「snapshot」者含訓練狀態，不宜與推論檔直接比較。MAC／FLOPs 只包含下述可精確計數的 subtotal，二值位元乘積另列。

| 案例 | Params M | Tensor MB | 來源檔 MB／類型 | MAC G subtotal | FP FLOPs G subtotal | Binary bit-products M |
| --- | --- | --- | --- | --- | --- | --- |
| SiLU E9 | 26.530 | 106.421 | 106.826 / inference/source | 47.883 | 95.766 | 40.960 |
| qSiLU E2 | 26.530 | 106.421 | 106.826 / inference/source | 47.883 | 95.766 | 40.960 |
| Hardswish zero-shot | 26.530 | 106.421 | 106.825 / inference/source | 47.883 | 95.766 | 40.960 |
| PolyShift zero-shot | 26.530 | 106.421 | 106.825 / inference/source | 47.883 | 95.766 | 40.960 |

### Peak memory、CPU／GPU latency、target latency、energy/frame

B1／640／FP32、CPU 4 threads。延遲為 median；GPU 是 CUDA events 的模型核心時間。含模型內 decode／top-k，但不含外部 NMS、影像讀取與 H2D。GPU peak 為暖機後的 PyTorch allocated tensor 高水位，並非整張卡總使用量。

| 案例 | GPU peak MiB | CPU ms | GPU ms | target ms | GPU J/frame | target J/frame |
| --- | --- | --- | --- | --- | --- | --- |
| SiLU E9 | 477.8 | 254.524 | 30.175 | 未量測 | 5.8741 | 未量測 |
| qSiLU E2 | 485.3 | 657.254 | 39.663 | 未量測 | 8.2207 | 未量測 |
| Hardswish zero-shot | 477.5 | 247.751 | 30.065 | 未量測 | 5.8264 | 未量測 |
| PolyShift zero-shot | 486.2 | 509.395 | 36.128 | 未量測 | 7.5411 | 未量測 |

GPU energy/frame 是 NVML 整卡能量差，含桌面／idle 活動；不能視為純模型、整機或目標板能耗。target 欄位因未指定／連接目標設備而保留缺值。

## KD

所有 AP 為既有完整驗證，範圍 0–1；— 表示此模型沒有該任務或此配對未驗證，不表示 0。Detect-only 的 COCO ball／bat 與 BBAT 二類 AP 不可混用。

| 案例／工作量 | COCO | person | COCO ball | COCO bat | BBAT box | BBAT Pose |
| --- | --- | --- | --- | --- | --- | --- |
| 雙教師 KD E4 / both | 0.502955 | 0.625193 | — | — | 0.613722 | 0.894121 |
| Pose-head KD E2 / both | 0.503885 | 0.625887 | — | — | 0.613915 | 0.892962 |

| 案例 | BBAT ball box | ball Pose | bat box | bat Pose |
| --- | --- | --- | --- | --- |
| 雙教師 KD E4 | 0.495441 | 0.859507 | 0.732003 | 0.928735 |
| Pose-head KD E2 | 0.501879 | 0.861716 | 0.725951 | 0.924207 |

### Params、Model size、MAC／FLOPs

Tensor MB 是本次 FP32 重建後參數＋buffers 的 payload，非已打包的低位元模型。來源檔 MB 是實際 checkpoint 大小；標記「snapshot」者含訓練狀態，不宜與推論檔直接比較。MAC／FLOPs 只包含下述可精確計數的 subtotal，二值位元乘積另列。

| 案例 | Params M | Tensor MB | 來源檔 MB／類型 | MAC G subtotal | FP FLOPs G subtotal | Binary bit-products M |
| --- | --- | --- | --- | --- | --- | --- |
| 雙教師 KD E4 | 26.530 | 106.421 | 106.825 / inference/source | 47.883 | 95.766 | 40.960 |
| Pose-head KD E2 | 26.530 | 106.421 | 106.825 / inference/source | 47.883 | 95.766 | 40.960 |

### Peak memory、CPU／GPU latency、target latency、energy/frame

B1／640／FP32、CPU 4 threads。延遲為 median；GPU 是 CUDA events 的模型核心時間。含模型內 decode／top-k，但不含外部 NMS、影像讀取與 H2D。GPU peak 為暖機後的 PyTorch allocated tensor 高水位，並非整張卡總使用量。

| 案例 | GPU peak MiB | CPU ms | GPU ms | target ms | GPU J/frame | target J/frame |
| --- | --- | --- | --- | --- | --- | --- |
| 雙教師 KD E4 | 485.3 | 664.323 | 39.749 | 未量測 | 8.2335 | 未量測 |
| Pose-head KD E2 | 485.3 | 654.054 | 40.671 | 未量測 | 8.2581 | 未量測 |

GPU energy/frame 是 NVML 整卡能量差，含桌面／idle 活動；不能視為純模型、整機或目標板能耗。target 欄位因未指定／連接目標設備而保留缺值。

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

## 主要量化判斷與配對差值

完整差值見[paired-deltas.csv](<paired-deltas.csv>)，包含比較範圍，不能只看差值忽略 parent／epoch。

- qSiLU 與 SiLU 的 Params 都是 26,529,701，MAC subtotal 都是 47.882752 G；但 CPU median 254.524 → 657.254 ms（2.582×），GPU 30.175 → 39.663 ms（1.314×），整卡遙測能量 5.8741 → 8.2207 J/frame。這正好說明 MAC subtotal 未涵蓋的 activation 算子与軟體 kernel 成本很重要。
- P3 MASF 比無 MASF 增加 75,777 個 registered parameters、0.475136 G Conv MAC；P2 增加同樣參數但 1.900544 G Conv MAC。配對 AP 仍未顯示足夠收益，所以不能以高成本本身推論模型較強。
- HOG 移除後與配對模型具有相同部署 Params／MAC；折疊 RepConv17 與原 Conv 亦相同。小幅 latency／energy 差沒有多輪隨機測試支持，不宣稱折疊後必然更快。
- KD、Pose-head KD 與關鍵點重組沒有新增部署 projector／teacher，與 qSiLU 主線的 Params／MAC 一樣；精度取捨仍需看 box 與 keypoint，不因成本相同就升版。
- FP → BinaryQK A0 的 GPU core 時間增加，不能將理論二值運算優勢直接套到此 bool-tiled PyTorch reference。FP／A0 是不同歷史權重，這裡只比較現存模型，不作單一修改的因果宣告。

## 如何解讀，不應得出的結論

BinaryQK 在這個軟體 bool-tiled reference 上不保證比 FP GPU 快；硬體友善性必須由目標實作驗證。A0／B100／FP 是歷史模型鏈，AP 差不是只改單一開關的完整重訓因果消融。

HOG 是 training-only，部署移除後成本應與同骨架接近；這不能解讀成 HOG 精度有效。RepConv 折疊消除多分支訓練成本，不代表參數或 latency 必然優於原 Conv。P2 MASF context 的空間面積增加使該模組 MAC 約為 P3 四倍，不是整網四倍。

KD 的 teacher 不進部署模型，所以同骨架 KD／非 KD 的 Params 通常相同；小幅 latency／energy 差不能歸因於 KD 算子增減。Hardswish／PolyShift 是既有 zero-shot，不與微調 10 輪的 qSiLU 作同訓練預算精度比較。

one2many 欄位是 Pose core，不含額外 class-aware NMS；本輪沒有宣稱完整 pipeline 變快。bat Pose 增益仍伴隨 ball Pose 下降，不能只因成本相近便採用。固定 8-scale selector、區域 KD、PTQ／QAT、target reciprocal 等未實施方案沒有可量測正式模型，不能編造其 Params 或 latency。

### 正式 target 補量流程

先指定板卡／時脈／runtime、權重量化格式與實際導出圖，再做同資料輸入的輸出等價性；分別量 model core 與完整 pipeline 的 median／P90。功耗需說明量測 rail 或插座、採樣率、idle 是否扣除，以及 E/frame 的積分區間。沒有這些資訊時，target 欄位維持未量測。

官方依據：[PyTorch peak memory](https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html)、[NVML 累計 GPU 能量](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html)。本報告的實測依據是本機輸出 JSON，而非網頁上的其他模型 benchmark。
