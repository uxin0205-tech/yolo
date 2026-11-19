# Paper-TWN 與逐區三元權重量化：第一手文獻查核與實驗契約

日期：2026-09-04

範圍：Paper-TWN、TTQ、INQ、逐層／漸進式 ternary 與 optimal scaled-codebook baseline

限制：本報告只做第一手文獻研究與實驗設計；未使用 GPU、未啟動或中止任何訓練／validation

## 一、結論先行

1. **目前沒有第一手論文能直接告訴我們 Full35 的 backbone、neck、Detect head、Pose head 中哪一區可以安全替換成三元權重。**最接近的 detector 證據仍是全模型或一般 detector PTQ；因此「先確定 backbone，再固定它調 neck，最後調 head」應視為本專案要實測的 staged routing，而不是文獻已證明的安全順序。
2. **固定 `Paper-TWN` 必須釘選版本。**2016 年的 [TWN v1](https://arxiv.org/pdf/1605.04711v1)／[v2](https://arxiv.org/pdf/1605.04711v2) 使用 `Δ = 0.7 E|W|`；目前的 [arXiv v3](https://arxiv.org/pdf/1605.04711v3) 改為 `Δ = 0.75 E|W|`，Algorithm 1 並明寫逐 filter 計算。若專案現行實作是 `0.7`，正確名稱應是 `paper_twn_v1_v2_fixed`（或釘選實際採用的 v2），不可籠統稱為唯一的 Paper-TWN 定義。
3. **原論文 TWN 不是純 PTQ。**它在 forward／backward 使用三元權重，但更新保留的 FP32 權重；把既有 checkpoint 一次映射後直接驗證，只是「使用 Paper-TWN 投影公式的 fixed PTQ baseline」，不是重現原論文訓練結果。
4. **Paper-TWN PTQ 失敗時，第一個替代品應是 `exact-scaled ternary PTQ`，而不是立刻做 TTQ。**它維持同一個對稱三元 codebook `{-α,0,+α}`，但對固定 codebook 全域最小化 weight MSE，可辨別失敗是來自 `0.7/0.75` threshold heuristic，還是三元容量／任務敏感度本身。
5. **若 exact ternary 仍失敗，再進 TTQ 或 progressive ternary QAT。**TTQ 學習正、負兩個 scale 與 assignment；INQ／MLQ 提供逐步量化、凍結、重訓其餘權重或層的先例。但它們的訓練成本與模型格式不同，不能和 fixed PTQ 放在同一成本欄直接宣稱勝負。
6. [TWN v3](https://arxiv.org/pdf/1605.04711v3) 的 YOLOv5s／PASCAL VOC 實驗從 COCO 權重初始化後訓練 150 epochs，TWN 的 mAP50 為 76.8、FP 為 86.7，mAP50:95 為 51.5、FP 為 63.7，分別落後 **9.9 與 12.2 AP points**。這不代表本專案逐區 ternary 必敗，但足以否定「直接把 backbone＋neck＋head 全部 ternary 便可滿足 1.5 AP-point gate」這種預設。

## 二、Paper-TWN 的精確定義

### 2.1 固定對稱三元投影

對選定的一組權重 \(W\)，TWN 求下式的近似解：

\[
\min_{\alpha\ge 0,\,W^t_i\in\{-1,0,+1\}}
\lVert W-\alpha W^t\rVert_2^2.
\]

給定 threshold \(\Delta\) 後：

\[
W^t_i=
\begin{cases}
+1,&W_i>\Delta\\
0,&|W_i|\le\Delta\\
-1,&W_i<-\Delta
\end{cases},
\qquad
\alpha_\Delta=
\frac{1}{|I_\Delta|}\sum_{i\in I_\Delta}|W_i|,
\]

其中 \(I_\Delta=\{i:|W_i|>\Delta\}\)。也就是小權重歸零，正負尾端共用一個非負 scale。公式與推導見 [TWN v2 §2.1–2.2](https://arxiv.org/pdf/1605.04711v2)。

### 2.2 `0.7` 與 `0.75` 是版本差異，不可混用

- [TWN v1（2016-05-16）](https://arxiv.org/pdf/1605.04711v1) 與 [v2（2016-11-19）](https://arxiv.org/pdf/1605.04711v2) 從對稱 uniform 與零均值 Gaussian 的近似推導，皆採用 `Δ ≈ 0.7 E|W|` 的 rule of thumb。
- [TWN v3（2022-11-20）](https://arxiv.org/pdf/1605.04711v3) 改成 `Δ ≈ 0.75 E|W|`，Algorithm 1 對每一層的每個 filter 計算 threshold 與 scale。
- 因此實驗 artifact 至少要保存 `paper_version`、`threshold_multiplier`、`granularity`。`per-tensor`、`per-layer`、`per-filter/output-channel` 不是同一個方法成本，也不應共用同一名稱。

本專案最乾淨的命名建議是：

| ID | 定義 | 訓練 |
|---|---|---|
| `paper_twn_v2_fixed_ptq` | `Δ=0.7E|W|`、對稱單 scale、明定 granularity | 無 |
| `paper_twn_v3_fixed_ptq` | `Δ=0.75E|W|`、依 v3 逐 filter | 無 |
| `paper_twn_v2_qat` / `v3_qat` | 相同投影，但以 ternary forward/backward 更新 FP master | 有 |

若只保留一個 Paper-TWN 主 baseline，應保留與既有實作一致的 `v2_fixed_ptq`；`v3_fixed_ptq` 只需作低成本版本 sentinel。更強的 threshold 基線由 exact-scaled ternary 負責。

### 2.3 原方法的訓練與部署限制

[TWN v2 §2.3](https://arxiv.org/pdf/1605.04711v2) 與 [v3 §2.4 Algorithm 1](https://arxiv.org/pdf/1605.04711v3) 都是在 forward／backward 使用三元化權重、參數更新仍保留 FP 權重，並搭配 BN、SGD momentum 與 learning-rate decay。因此：

- one-shot fixed projection 是合理的 PTQ 壓力測試，但不能用原論文訓練成果替它背書；
- 原論文以 2 bits 儲存一個三元 index，聲稱最高約 16× 對 FP32 的 weight-code 壓縮；實際模型還有 scale、bias、BN、未量化 island、對齊與 packing metadata，必須另報真實 packed bytes；
- 零權重可跳過累加、scale 可移到輸入端，是潛在硬體優勢；沒有實際 ternary kernel 時，只能報 code／sparsity／BOP proxy，不能宣稱現成 GPU latency 加速。

## 三、哪些權重分布理論上較適合

以下嚴格區分「論文直接事實」與「由公式導出的專案假說」。

| 權重／區域特徵 | 判斷 | 證據邊界與應採方法 |
|---|---|---|
| 近似零中心、正負對稱，兩端幅度相近 | **較符合 Paper-TWN 假設** | v2 的 threshold 近似由對稱 uniform 或 \(N(0,\sigma^2)\) 推導；這只支持把它列為較可能候選，不保證 task metric 通過。 |
| 中央有可歸零的小權重，剩餘正負尾端可由單一幅度代表 | **較符合三元 codebook 形狀** | 由 `{-α,0,+α}` 直接推得。應同時報 zero／positive／negative occupancy 與 tail reconstruction error。 |
| 正、負尾端尺度明顯不對稱 | **固定 Paper-TWN 不利；TTQ 較合理** | [TTQ §4](https://arxiv.org/abs/1612.01064) 明確以兩個可學係數 `Wp`,`Wn` 建立 `{-Wn,0,+Wp}`，理由之一就是增加非對稱 codebook 容量。 |
| 非 Gaussian、重尾、離群或多峰 | **先用 exact-scaled ternary 檢驗 heuristic 問題** | [Optimal Quantization Using Scaled Codebook](https://openaccess.thecvf.com/content/CVPR2021/html/Idelbayev_Optimal_Quantization_Using_Scaled_Codebook_CVPR_2021_paper.html) 對任意資料分布、固定 codebook 求全域 MSE 最優 scale／assignment，不需要 TWN 的分布假設。 |
| weight MSE 很小，但 detector／pose metric 很敏感 | **不可只靠分布決策** | scaled-codebook 論文 §5.2 明寫 weight／activation MSE optimum 不等於 model-loss optimum；需要區域 output reconstruction 與最終任務指標。 |
| 參數量大但任務敏感度低 | **壓縮收益可能較好，仍屬待驗證假說** | 文獻沒有把這個條件映射到 Full35 的 backbone／neck／head；必須以逐區實驗決定。 |

[TTQ §6.1](https://arxiv.org/abs/1612.01064) 在 ResNet-20 的訓練實驗觀察到約 30%–50% sparsity 時誤差最低，但這是特定分類模型與 TTQ 訓練結果，不是所有層應固定成相同 zero ratio 的定律。實際上 TTQ 也觀察到各層 scale 與 sparsity 隨訓練演化不同，支持逐層記錄而非全模型共用一個 occupancy 目標。

## 四、first／last／predictor 敏感度：文獻可以與不可以說什麼

### 可以說

- [TTQ ImageNet 實驗 §5.1](https://arxiv.org/abs/1612.01064) 保留 AlexNet 第一個 convolution 與最後一個 fully connected layer 為 FP，其他層 ternary。
- [Reg-PTQ §5.1](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_Reg-PTQ_Regression-specialized_Post-training_Quantization_for_Fully_Quantized_Object_Detector_CVPR_2024_paper.html) 在 RetinaNet、YOLOF、Faster/Mask R-CNN 的 protocol 中將第一層設為 8-bit、最後 prediction layer 保持 FP，並針對 detector regression structure 的非均勻參數與全域校正另設方法。
- TWN v3 的 YOLOv5s 全模型結果顯示 detector 在極低位元 ternary 下仍有很大的 FP gap，即使做了 150 epochs fine-tuning。

### 不可以說

- TTQ 與 Reg-PTQ 的 first／last 保護是它們的實驗 protocol，沒有提供「Full35 predictor 必定最敏感」的逐區消融證明。
- TWN 沒有 backbone／neck／head 的 ternary sensitivity 表，也沒有 shared Detect＋Pose、BBAT ball／bat／pose 的證據。
- 因此不可先宣告 backbone 最適合或 predictor 一定不適合；可做的是把最後 predictor 設成保守 anchor，最後再拆成 Detect tower、Detect predictor、Pose tower、Pose predictor 逐項挑戰。

## 五、PTQ 失敗後的替代演算法順序

### A. Exact-scaled ternary PTQ：第一替代品

[CVPR 2021 scaled-codebook](https://faculty.ucmerced.edu/mcarreira-perpinan/papers/cvpr21.pdf) 解：

\[
\min_{\alpha>0,Z}\sum_n\sum_k z_{nk}(w_n-\alpha c_k)^2,
\quad C=\{-1,0,+1\},
\]

並給出對任意固定 codebook、任意資料分布的全域最優演算法，複雜度為 \(O(NK\log K)\)（不含預先排序成本）。它是純 PTQ，可作 Paper-TWN 的同格式強 baseline。

判讀方式：

- Paper-TWN fail、exact pass：主要是固定 threshold／assignment heuristic 不夠好；
- Paper-TWN 與 exact 都 fail，且 task metric 同樣差：問題較可能是對稱三元容量或該區 task sensitivity；
- exact 的 weight MSE 顯著較低但 task metric 仍 fail：不能再靠 weight-only objective，應進 output/task-aware recovery 或保留較高 precision。

對每個 output channel 個別執行 exact solver 是合理的工程延伸，但不再是原論文「single codebook per layer」的同一 metadata 成本；必須以不同 ID 與 scale bytes 報告。

### B. TTQ：需要 QAT 的非對稱三元

[TTQ](https://openreview.net/forum?id=S1_pAu9xl) 使用每層兩個可學 scale：

\[
w_l^t\in\{-W_l^n,0,+W_l^p\},\qquad
\Delta_l=t\max|\widetilde w_l|,
\]

論文實驗使用 `t=0.05`，訓練時保留 latent FP weights，反向更新正／負 scale 與 ternary assignment。它比 Paper-TWN 更能處理正負不對稱，但屬 QAT，且推論需要兩個 scale；必須另算 metadata 與硬體 datapath。

### C. INQ-style progressive ternary recovery

[INQ, ICLR 2017](https://arxiv.org/pdf/1702.03044) 對每層權重做「分組 → 量化並凍結一組 → 重訓其餘組」，逐步把累積量化比例推到 100%。其 2-bit ternary ResNet-18 使用累積比例 `{0.2,0.4,0.6,0.7,0.8,0.85,0.9,0.95,0.975,1}`，訓練達 30 epochs；Top-1 error 從 FP 的 31.73% 變成 33.98%，表示 progressive 也不是極低位元無損保證。

重要限制：INQ 原方法的 codebook 是 powers-of-two 加 zero；2-bit 時才退化成 ternary。它的 progressive schedule 是**各層內權重比例**，不是 backbone→neck→head 的區域順序。若本專案使用它，正確名稱應是 `inq_style_ternary_recovery`，並清楚標為適配方法。

### D. 明確的逐層 ternary 先例

- [Ternary Neural Networks for Resource-Efficient AI Applications](https://arxiv.org/abs/1609.00222) 使用 teacher–student 與 layer-wise greedy ternarization，並可在量化下一層前只重訓尚未 ternary 的層；但它同時 ternary weights 與 activations，任務是分類，不能直接當 weight-only YOLO baseline。
- [Multi-Level Quantization（arXiv preprint）](https://arxiv.org/abs/1803.03289) 明確為 ternary 引入 incremental layer compensation：逐步量化部分 layers，重訓其餘 layers；但 codebook 來自三群 k-means centroids，不是固定對稱 Paper-TWN。

這兩篇支持「一次只處理部分 layer、讓其餘區域補償」的研究合理性，不能證明本專案應固定使用 backbone→neck→head 的唯一順序。

## 六、建議採用的 backbone → neck → head 實驗鏈

### 6.1 核心原則

每一步只改當前區域，上一階段已通過的設定凍結，後續區域仍維持 parent precision。若當前區域失敗就回退到上一個合格 parent，不能帶著失敗設定繼續累積。

```text
P0：已鎖定 activation + W8/保護島 parent
  ↓ 只替換 backbone
PB：選定的 backbone ternary/W4/W8 policy
  ↓ 固定 PB，只替換 neck
PBN：選定的 backbone + neck policy
  ↓ 固定 PBN，依序替換 head tower / predictor
PBNH：最終 mixed-format policy
```

這個流程回答的是「在已確定上游設定下，下一區替換的邊際代價」。如果需要回答 region interaction，可在 finalist 只補一個 `neck-only` 或 `head-only` sentinel，不必把所有排列都跑完。

### 6.2 每一區的最小公平矩陣

| Cell | 當前區域 | 作用 |
|---|---|---|
| Parent control | 維持上一階段 W8／既定格式 | 計算本步 `Δ_step` |
| Uniform W4 | 同一組 exact paths | 4-bit 容量與常規量化對照 |
| Paper-TWN v2 fixed | `Δ=0.7E|W|` | 既有 fixed ternary baseline |
| Exact-scaled ternary | `C={-1,0,+1}` | 排除 threshold heuristic 偏弱 |
| Paper-TWN v3 sentinel | `Δ=0.75E|W|`、同 granularity 契約 | 只檢查版本差異；可由 static/probe 淘汰 |
| TTQ 或 INQ-style recovery | 只對 PTQ recovery-band candidate | 不讓所有失敗 cell 都消耗 QAT |

Head 不應一次合成一格；至少依序拆成：

1. Detect tower；
2. Detect predictor（class／box 若 graph 可分則分開）；
3. Pose tower；
4. Pose predictor（keypoint／visibility 若 graph 可分則分開）。

這不是預判 predictor 必敗，而是讓 regression／keypoint 的失敗可以被定位，並保留 W8／FP fallback。

### 6.3 通過與回退判定

每個 cell 同時報：

- `Δ_total`：相對同 activation 的 FP-weight／正式 total baseline；
- `Δ_step`：相對目前凍結 parent；
- 八個任務指標逐項值與 delta：COCO person box mAP50／mAP50-95、BBAT ball box、bat box、pose 各自的 mAP50／mAP50-95；
- weight NRMSE、當前區域 output reconstruction、正／零／負 occupancy、每層 scale、packed code bytes、scale/metadata bytes；
- 有真正 ternary kernel 才報 latency／energy，否則標記為 `not_measured`。

依目前專案目標，**每個 mAP50 的 total drop 需不超過 0.015，且這個 total 必須包含 activation 替換造成的下降**；mAP50-95 仍逐項報告並套用既定 companion gate，不能以平均分數掩蓋 ball、bat、pose 或 COCO person 的最差任務。靜態 MSE 與 output probe 只作 promotion，不取代 canonical validation。

決策規則：

- Paper-TWN 通過且 exact 無實質 Pareto 改善：保留較簡單的 Paper-TWN；
- exact 通過、Paper-TWN 失敗：選 exact ternary，並把結論寫成 threshold heuristic 問題；
- 二者 PTQ 都在 recovery band：只晉級較好的 route 做 TTQ／INQ-style 短 QAT；
- QAT 後仍失敗：該區保持 W4／W8／FP，不再以全區 ternary 為目標；
- 單區通過、累積後失敗：判為 interaction，回退最新區域，不把單區 delta 直接相加。

## 七、公平比較契約

所有 Paper-TWN、exact ternary、W4 與後續 QAT 比較必須固定：

1. 同一 activation parent、起始 checkpoint hash、BN-fold／deployment view、模型 graph 與 protected islands；
2. 同一組 exact module paths；region 名稱相同不等於 paths 相同；
3. 同一 granularity；若 per-filter／per-channel 不同，必須另列 cell 並計 metadata；
4. 同一 BBAT5 immutable split、COCO person split、calibration／search manifests；不得為某算法重抽較有利資料；
5. PTQ cell 完全不 fine-tune；QAT cell 使用相同資料比例、epochs、optimizer family、LR schedule、augmentation、seed 數、selector 與 sham control；
6. TTQ 兩個 scale、INQ 多階段訓練與 exact solver CPU 成本都需納入方法成本；
7. 同時報 theoretical code capacity 與 bit-true packed bytes，不以「三個值等於 1.58 bits」取代實際 2-bit encoding／metadata；
8. validation 只評估預註冊 cell；threshold、route 或 checkpoint selection 不可看 formal test 結果後再改。

## 八、對目前整體計畫的落點

- 目前正在進行的 W8 QAT／queue 不需停止，也不應為這份研究插入另一個 GPU job。
- Paper-TWN 現階段先完成 CPU static／route manifest 與版本釘選；等待現行訓練結束後，再排進 GPU validation queue。
- 下一輪三元實驗不再使用一次混多區的 `safe/balanced` route 作第一判斷；改用本文的逐區 parent chain，先 backbone，通過後固定，再 neck，最後細分 head。
- 特殊格式的總比較至少保留 `W8 parent / exact W4 / Paper-TWN v2 / exact ternary / Fixed-SD4`；TTQ 與 learned-SD4 是有訓練的 recovery／研究支線，必須用 matched QAT budget 比較。

## 九、第一手來源

1. Li et al., [Ternary Weight Networks v1（2016）](https://arxiv.org/pdf/1605.04711v1)；[v2（2016）](https://arxiv.org/pdf/1605.04711v2)；[v3（2022）](https://arxiv.org/pdf/1605.04711v3)；[作者公開程式碼](https://github.com/Thinklab-SJTU/twns)。
2. Zhu et al., [Trained Ternary Quantization, ICLR 2017](https://openreview.net/forum?id=S1_pAu9xl)；[arXiv 全文](https://arxiv.org/abs/1612.01064)。
3. Zhou et al., [Incremental Network Quantization, ICLR 2017（OpenReview）](https://openreview.net/forum?id=HyQJ-mclg)；[arXiv 全文](https://arxiv.org/pdf/1702.03044)。
4. Idelbayev et al., [Optimal Quantization Using Scaled Codebook, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Idelbayev_Optimal_Quantization_Using_Scaled_Codebook_CVPR_2021_paper.html)；[作者保存的全文](https://faculty.ucmerced.edu/mcarreira-perpinan/papers/cvpr21.pdf)。
5. Alemdar et al., [Ternary Neural Networks for Resource-Efficient AI Applications](https://arxiv.org/abs/1609.00222)。
6. Xu et al., [Deep Neural Network Compression with Single and Multiple Level Quantization（arXiv preprint）](https://arxiv.org/abs/1803.03289)。
7. Ding et al., [Reg-PTQ, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_Reg-PTQ_Regression-specialized_Post-training_Quantization_for_Fully_Quantized_Object_Detector_CVPR_2024_paper.html)；[CVPR 官方頁](https://cvpr.thecvf.com/virtual/2024/poster/30629)。

## 十、未解風險

- TWN v2 與 v3 的 threshold／granularity 已發生變更；既有 artifact 若未保存版本資訊，需由 config 與程式 hash 追溯，不能只看 `paper_twn` 名稱。
- 文獻沒有 Full35 shared Detect＋Pose 的 ternary 區域排序；任何「backbone／neck／head 哪個最好」都必須等本專案逐區 metrics，不能由分類論文外推。
- exact-scaled ternary 只保證 weight MSE 全域最優，不保證 output 或 mAP 最優。
- TTQ、INQ、MLQ 需要訓練；若最終部署端沒有對應 ternary sparse kernel，模型縮小不等於實際 latency 改善。
