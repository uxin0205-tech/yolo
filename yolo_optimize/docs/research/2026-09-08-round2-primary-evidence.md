# 第二輪優化研究：指定一手來源證據

- 日期：2026-09-08
- 範圍：只核對 QuaRot、SpinQuant、CrossKD、Localization Distillation（LD）、YOLOv10 原論文與 Ultralytics YOLO26 官方文件。
- 證據規則：外部資料只採原始論文、作者官方 repository 與 Ultralytics 官方文件；不引用二手文章，不新增本地 AP 或訓練結果。
- 閱讀邊界：以下「原文支持的機理」是來源可直接核對的摘要；「不可外推之處」明確保留架構、位元寬度、資料集與硬體差異。

## 1. QuaRot 與 SpinQuant：共同 Q/K 旋轉、量化目的及 fold／硬體限制

### 1.1 QuaRot（2024）

- 原始 URL： [論文（arXiv）](https://arxiv.org/abs/2404.00456)、[作者 repository](https://github.com/spcl/QuaRot)
- 原文支持的機理：論文在 attention 的 positional embedding（含 RoPE）之後，對每個 head 的 Q 與 K 同時右乘同一個正交 Hadamard `H`；因此 `(QH)(KH)^T=QHH^TK^T=QK^T`，作者明確說 full-precision final attention scores 不變。隨機 Hadamard 由 `H diag(s)` 構成，`s` 是 ±1 符號；Walsh–Hadamard 可用 `O(d log d)` 快速變換。
- 不可外推之處：這是旋轉後、量化前的 FP 等價性，不是 `sign(Q),sign(K)` 二值化後的 score 等價性；論文目標是 Llama 等 LLM 的 W/A/KV INT4，沒有 YOLO detection 或 binary-sign QK 證據。論文指出 positional embedding 使 Q/K 不能直接 fold，需 online head-wise transform；其特殊 CUDA kernel、power-of-two／分解維度與延遲不可當成本案硬體保證。查閱的原始段落只有 signed `H diag(s)`，未支持另有 permutation 的 Q/K 方法。

### 1.2 SpinQuant（2024）

- 原始 URL： [論文（arXiv）](https://arxiv.org/abs/2405.16406)、[作者 Q/K rotation code](https://github.com/facebookresearch/SpinQuant/blob/main/train_utils/apply_r3_r4.py)
- 原文支持的機理：SpinQuant 將 `R1`（residual）與 `R2`（head-wise value／output pair）學習為可吸收旋轉，`R3/R4` 保留 online Hadamard 以處理 activation／KV-cache outlier。論文的隨機 Hadamard 同樣是 `H diag(s)`；作者 code 的 `QKRotationWrapper` 對 Q、K 各套用同一個 `HadamardTransform`，並要求 `head_dim` 為 2 的冪，印證 shared Q/K orthogonal transform 可維持 dot-product 幾何（於未量化 FP 路徑）。
- 不可外推之處：SpinQuant 的收益是 Llama／Mistral LLM 在 W4A8、W4A4KV4 等 PTQ 設定的語言任務收益，不是本案 detector 或 binary-sign QK 的收益。`R3/R4` 因不能吸收而留在 forward；論文報告 online Hadamard 約增加 8% network latency，且 code 有 head-dim／group-size 契約，不能取代 target backend profile。學習的是連續正交 `R1/R2`，已查內容未支持把 permutation 或 learned binary basis 直接宣稱為本案創新或 AP 保證。

### 1.3 旋轉證據的共同界線

兩篇一手來源都支持「Q 與 K 使用相同正交變換時，未量化 attention dot product 保持不變」；兩篇也把 Hadamard 的 signed 版本（`H diag(s)`）放在量化友善化脈絡。它們沒有證明同一性在 sign nonlinearity、global scale、STE、detection assignment 或 YOLO26 DFL-free head 中仍成立。故本案 `{I, H D}`（共同固定 ±1 的 `D`）最多可視為已有 signed-Hadamard／shared-QK 家族上的 task-specific binary 試驗假說；是否有新穎性、收益或硬體可 fold，需另由本案 matched 實驗與 kernel 契約判定。

## 2. CrossKD 與 LD：teacher／student head、dense detection 與分布回歸契約

### 2.1 CrossKD（CVPR 2024）

- 原始 URL： [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html)、[論文（arXiv）](https://arxiv.org/abs/2306.11369)
- 原文支持的機理：給定 dense detector，CrossKD 把 student detection head 的中間特徵 `f_i^s` 送入 frozen teacher head 的下一層 `C_{i+1}^t`，形成 cross-head prediction `p̂^s`；蒸餾比較的是 `p̂^s` 與 teacher 原始 prediction `p^t`，不是 student 原始 `p^s`。偵測 loss 走 student head，KD gradient 經 teacher head 回到 student latent；classification 使用 QFL，直接 box regression 使用 GIoU，GFL 式 location distribution 則使用帶溫度的 KL。
- 不可外推之處：論文實驗以 GFL／RetinaNet／FCOS／ATSS 等 dense heads、COCO 與其 assigner／輸出圖為契約；需另對齊 YOLO26 的 o2m／o2o head、分支深度、輸出形狀與 assigner。CrossKD 的 GIoU 分支不要求 DFL，但這不等於可直接套入 YOLO26；論文沒有 YOLO26 或 binary QK 證據，不能宣稱已適用或有本地 AP 提升。

### 2.2 Localization Distillation（LD，CVPR 2022）

- 原始 URL： [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html)、[論文（arXiv）](https://arxiv.org/abs/2102.12252)、[作者 repository](https://github.com/HikariTJU/LD)
- 原文支持的機理：LD 將 box 的四個 edge `t,b,l,r` 從 Dirac／單值回歸改為離散機率分布；teacher／student localization logits 經溫度 `τ` 的 SoftMax 後，對每條 edge 以 KL divergence 蒸餾，四條 edge 求和。VLR 以每個 FPN level 的 anchor 與 GT 之 DIoU，選取 `γ α_pos ≤ DIoU ≤ α_pos` 的 valuable region；這些設計用來傳遞邊界歧義與分布暗知識。
- 不可外推之處：LD 的核心輸入是 GFocal/GFL 類 distribution regression 與 DFL-style bins；YOLO26 官方架構是 `reg_max=1`、DFL layer 為 Identity、直接回歸座標，因此不能原樣套用 LD 的 KL-over-bins。若加 auxiliary distribution head 或另定 box-logit 契約，那是新的適配實驗；原論文沒有 YOLO26、DFL-free loss 或 binary QK 結果，不能宣稱直接適用或有 AP 收益。

## 3. YOLOv10 與 YOLO26：一致匹配、雙 head 監督及既有 loss 能力

### 3.1 YOLOv10 原論文（2024）

- 原始 URL： [論文（arXiv）](https://arxiv.org/abs/2405.14458)
- 原文支持的機理：YOLOv10 把 one-to-many（每個 GT 多個正樣本）與 one-to-one（每個 GT 一個正樣本）放在兩個 head，訓練時 joint optimize、共享 backbone／neck 的豐富監督，推論丟棄 o2m 而用 o2o 免 NMS。兩者的 matching metric 是 `m=s·p^α·IoU( b̂,b)^β`；設定 `α_o2o=rα_o2m`、`β_o2o=rβ_o2m`（預設 `r=1`）使 o2m 最佳正樣本也是 o2o 最佳樣本，縮小 supervision gap。
- 不可外推之處：論文的結論是 YOLOv8-derived YOLOv10 在 COCO 上的 assigner、head 與 capacity 設定；它不是對所有 dual-head detector 的等價性證明。尤其不能僅因 YOLO26 也有 o2m／o2o 就假定採用相同 `α,β,r` 或相同 target。論文亦未研究 binary-sign QK、DFL-free YOLO26 或本案資料，不能提供本案 AP／優先順序。

### 3.2 Ultralytics YOLO26 官方文件（2026）

- 原始 URL： [YOLO26 model docs](https://docs.ultralytics.com/models/yolo26/)、[End-to-End Detection guide](https://docs.ultralytics.com/guides/end2end-detection)、[Training Recipe](https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/yolo26-training-recipe.md)
- 原文支持的機理：官方文件確認 YOLO26 訓練 one-to-many 與 one-to-one 兩個 head，兩者共享 backbone／neck 並共同最佳化；預設 prediction／validation 走 o2m+NMS，`nms=False` 選 o2o、免 IoU suppression。官方 architecture／recipe 亦明載 `reg_max=1`、DFL 變 `Identity`；既有 detection loss 仍含 IoU box、classification 與 box-distance 項，其中名為 `dfl` 的 gain 在 DFL-free YOLO26 實際加權 normalized-distance L1。這表示 YOLO26 已有 dual supervision 與直接 box／class loss 的接口能力。
- 不可外推之處：目前查到的官方頁面沒有公開 YOLO26 完整 o2m/o2o matching equation，故不能宣稱它精確等同 YOLOv10 的 consistent metric；也沒有 CrossKD／LD、binary QK 或 rotation 的適配結果。文件所說的 loss「已有能力」只支持接到現有 box/class/distance outputs 的工程可行性，不代表可直接使用 LD 的 distribution KL，也不代表能恢復本案量化精度。

## 4. 一手證據整理出的明確未解問題

- 未找到任何上述來源把 QuaRot／SpinQuant 的 FP shared-QK score invariance 證明延伸到 `sign(Q), sign(K)`、STE 或本案 BinaryQK；不得把正交等價式寫成二值化後等價式。
- 未找到任何上述來源在 YOLO26 或本案資料／評估器上報告 AP；本檔不提供本地數值，也不把 LLM、GFL、COCO 結果轉成本案預期增益。
- signed Hadamard（`H diag(s)`）與 shared Q/K rotation 已由既有一手來源明載；查閱段落未見 separate permutation 作為其 Q/K 核心。`{I,H D}` 是否對本案 binary basis 有效、是否可 fold、是否保留 target hardware gain，仍須本案 bit-true 與 matched latency／accuracy 驗證。
- YOLO26 官方文件確認 DFL-free direct regression，但沒有現成 localization distribution target；LD 需 bins／distribution 的缺口不能以把 `dfl` gain 改名來消除。

## 5. 子任務中文 worklog

### 變更內容與原因

- 新增本檔，集中記錄第二輪指定問題的一手來源、可核對機理與不可外推邊界；原因是主代理需要在分析／排序前先區分「來源證明」與「本案假說」。
- 納入 QuaRot／SpinQuant 的 shared Q/K、signed Hadamard、online／fold 條件，並補上 CrossKD／LD 的 teacher-student head 與 regression loss 契約，以及 YOLOv10／YOLO26 的 dual supervision 與 DFL-free loss 事實。

### 驗證方式與結果

- 以原始 arXiv HTML／abstract、CVPR Open Access 頁面、作者 official GitHub source 與 Ultralytics 官方 docs 交叉核對公式、head 流向、loss、年份與輸出路徑；結果可定位至本檔各來源 URL。
- 確認 QuaRot 與 SpinQuant 均有 shared Q/K 正交變換證據；確認隨機 Hadamard 的 signed `H diag(s)` 證據；未把 permutation、binary-sign score equality 或 YOLO26 AP 寫成已證明事實。
- 確認 CrossKD 同時描述 direct-box GIoU 與 distribution-regression KL；確認 LD 明確需要 edge probability distribution；確認 YOLO26 `reg_max=1`／Identity DFL 與 normalized-distance L1 recipe。
- 本次 GPU 工作數：0；未 import torch、未載入模型或 checkpoint、未訓練、未做資料／影像／標註變更、未產生 AP 或 latency 實驗。

### 困難與解法

- 部分 CVPR Open Access PDF／HTML 直接讀取受 403 限制；改以同一論文的 arXiv HTML／abstract 與官方 CVPR metadata 交叉核對，保留 CVPR 原始 URL，不使用二手摘要。
- 初始環境的 sandbox bwrap loopback 啟動失敗；唯讀查詢改用核准升權執行，檔案更新仍只用 `apply_patch`。
- 「SpinQuant 是否 shared Q/K」在論文摘要未直接列出；改核對作者 `apply_r3_r4.py` 的 QK wrapper，確認 Q、K 使用相同 `HadamardTransform` 並記錄 head-dim／group-size 限制。
- 其餘：無。

### 未解事項與風險

- YOLO26 完整 matching equation、正式 training branch 的 assigner 細節與 CrossKD／LD 直接 adapter 尚未由官方文件證明；不可自行補寫或假定等同 YOLOv10。
- shared orthogonal transform 只保證未量化 FP dot-product 幾何；sign、scale、STE 與 downstream detector loss 的誤差仍需本案 matched probe／實驗驗證。
- signed／permuted Hadamard 的硬體 fold 取決於本案 layout、RoPE／位置操作（若有）與 target kernel；論文的 CUDA／fast-Hadamard 成本不可當本案部署結果。
- 請主代理完成後續方向分析，並將本檔加入中央研究索引；本子任務不代主代理決定優先順序、創新性或 winner。
