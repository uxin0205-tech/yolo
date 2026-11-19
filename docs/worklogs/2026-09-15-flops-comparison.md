# 2026-09-15：原始模型與最新模型 FLOPs 比較

## 任務與範圍

對原始 YOLO26m、YOLO11m、YOLO11m_bat 與最新 Rep17＋20 E5 BitTrue 做一致口徑的 CPU 前向計數。

## 變更內容與原因

新增可重現腳本與 JSON 結果。最新模型使用 model/load_model.py 正式載入入口，分別計算 Detect、Pose、both；共用骨幹在 both 只執行一次。所有模型 eval，輸入 1×3×640×640，未另做 fuse 或刪除 head 分支。無資料集讀寫。

## 驗證方式與結果

使用 Conv2d hook 與 TorchDispatchMode 計算實際 Conv 和矩陣乘加；每次乘加計 2 FLOPs。使用 no_grad 並停用 inference_mode，避免 dispatcher 漏算。所有輸出有限，YOLO26m 數值與既有 accounting-v2 完全相符，both 小於兩次獨立任務成本合計。

| 模型／任務 | 浮點運算小計 GFLOPs | 二元 bit products（十億） |
| --- | ---: | ---: |
| yolo26m | 75.011277 | 0.00000 |
| yolo11m | 68.104448 | 0.00000 |
| yolo11m_bat | 71.461837 | 0.00000 |
| rep17_20_E5_detect | 74.929357 | 0.08192 |
| rep17_20_E5_pose | 82.594406 | 0.08192 |
| rep17_20_E5_both | 95.765504 | 0.08192 |

每個 StaticDyadicScore 有原始 sign 與 Hadamard sign 兩組二元內積，故計 2 倍 QK bit products；兩個 attention site 合計 81,920,000 bit products。此數字不是 81,920,000 次 CPU 指令，也未換算為 FLOPs。

來源與腳本：[profile_models.py](assets/2026-09-15-flops-comparison/profile_models.py)、[results.json](assets/2026-09-15-flops-comparison/results.json)、[profile.log](assets/2026-09-15-flops-comparison/profile.log)。

## 困難與解法

一般模型 summary 對客製類別及函式形式矩陣乘法的計數可能不完整；統一採實際前向的 Conv／矩陣乘加計數，並把 BinaryQK 的二元內積另列。

## 未解事項或風險

這是主要算術運算小計，不含 BN、bias additions、activation、PWL、reciprocal、Hadamard transform、pooling、decode、sort、NMS、reductions 與記憶體搬移。不等同 Ultralytics fused summary 或完整硬體操作總量；不將 FLOPs 直接當成延遲或能耗。未使用 GPU，未修改模型權重。

## 參數量補充

直接載入上述相同權重，使用 sum(p.numel() for p in model.parameters()) 計算，未額外 fuse，包含 requires_grad=False 的參數，排除 buffers；共用參數不重複計數。此為整份已載入模型的參數量，不是各任務實際執行的參數子集。

| 模型 | Parameters |
| --- | ---: |
| YOLO26m | 21,896,248 |
| YOLO11m | 20,114,688 |
| YOLO11m_bat | 20,904,360 |
| Rep17_20_E5 | 26,529,713 |

驗證：四份模型均成功載入；最新模型由正式 loader 完成 SHA 與 strict load 核對。困難：無。未解事項：未另計 fused 模型或刪除未使用分支後的參數量。結果保存於 assets/2026-09-15-flops-comparison/parameters.json。
