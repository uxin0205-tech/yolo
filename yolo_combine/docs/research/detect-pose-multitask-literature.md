# Detect + Pose 多任務模型：學術與實務做法

> 研究範圍：以 YOLO26m 為基礎，共享 backbone + neck；Detect head 只做人員偵測，Pose head 做 ball / bat 的 bbox 與 2 個 keypoints。兩個任務使用不同資料集，先在 GPU 驗證，最終考慮 FPGA。
> 日期：2026-08-13。本文只採用論文、官方文件及官方原始碼。

## 結論先行

這個構想在學術上屬於 **hard parameter sharing（硬式參數共享）的 multi-task learning**：共享特徵抽取器，保留 task-specific heads。它不是罕見或不合理的架構；Mask R-CNN、MultiTask-CenterNet 等工作都已證明「共享特徵 + 平行任務分支」是偵測與 keypoint/pose 的主流整合方式。[Mask R-CNN](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html) 將 bbox 與其他 instance-level 分支放在同一框架；[MultiTask-CenterNet](https://openaccess.thecvf.com/content/ICCV2021W/ERCVAD/html/Heuer_MultiTask-CenterNet_MCN_Efficient_and_Diverse_Multitask_Learning_Using_an_Anchor_ICCVW_2021_paper.html) 更直接把 object detection、semantic segmentation 與 human pose 放入共享的 anchor-free 網路。

本專案真正的研究難點不是「能不能接兩個 head」，而是：

1. COCO-person 與 BBT ball/bat-pose 是 **disjoint datasets / partial task labels**。
2. 兩個 loss 的尺度、學習速度與對 shared trunk 的梯度可能不同，會造成 negative transfer。
3. FPGA 上的 MUX 是推論路由與硬體排程問題，不會自行解決訓練問題；若同一 frame 要兩種輸出，仍需執行兩個 heads。
4. 最終板卡未定，因此現在應維持可匯出、固定 shape、以 convolution 為主的 graph，並把 INT8/QAT 當成獨立實驗軸。

最有證據支持且風險最低的第一版，是：**YOLO26m shared backbone + PAN/FPN neck，後接獨立 Detect 與 Pose26 heads；採 task-balanced 的雙資料流、missing-task loss masking、兩任務合併成一次 optimizer update；先用固定 loss 權重建立 baseline，發現衝突後才加入 GradNorm 或 PCGrad。**

## 1. 方法分類

### 1.1 完全分開的兩個模型

```text
image -> YOLO26m Detect
image -> YOLO26m Pose
```

這是精度與工程風險的必要 baseline。優點是任務完全不干擾；缺點是 backbone 與 neck 被計算、儲存兩次。後續所有「省多少參數、MACs、顯存與延遲」都必須以這個雙模型系統為比較對象，而不是只和單一 YOLO26m 比較。

### 1.2 硬式共享：shared trunk + task-specific heads（本案首選）

```text
                         +-> person Detect head
image -> backbone -> neck|
                         +-> ball/bat Pose26 head
```

這是最直接的 multi-head MTL。共享部分同時接受兩任務梯度；每個 head 只接受自己任務的梯度。Mask R-CNN 的核心模式就是在共享特徵之上加入平行預測 branch，且顯示可以在同一框架處理 bbox 與 person keypoints。[Mask R-CNN paper](https://openaccess.thecvf.com/content_iccv_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)

對 YOLO26 而言，官方架構本身就是 backbone → FPN/PAN neck → head，並從多尺度特徵做預測，因此增加第二個 task head 有清楚的介面位置。[Ultralytics YOLO architecture guide](https://docs.ultralytics.com/guides/yolo-architecture/) 官方原始碼也顯示 `Pose` 繼承 `Detect`，在 box/class branches 外增加 keypoint branch；`Pose26` 再加入 RLE 訓練所需分支。[Ultralytics `head.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py)

### 1.3 單一 pose head 同時涵蓋 person、ball、bat

YOLO pose head 本身已包含 bbox 和 classification，所以理論上可做一個三類 head，再對沒有 keypoints 的 person instance 遮罩 keypoint loss。這會減少一組 box/class branch，但有三個問題：

- person 與 ball/bat 的 keypoint schema 不相同；標準 Ultralytics pose dataset 以全資料集共用一個 `kpt_shape`。
- COCO-person 與 BBT 不是同圖完整標註，仍要自行處理 missing labels。
- 本案需要 FPGA task selector；單一 head 會讓 person detect 與 ball/bat pose 的運算較難獨立關閉。

因此它適合作為後續壓縮 ablation，不適合作為第一版。

### 1.4 軟式／選擇性共享

當完全共享造成 negative transfer，可改成 task-specific neck、只共享 backbone，或用可學習的 feature mixing。Cross-stitch Networks 讓網路學習 shared 與 task-specific activations 的組合，說明最佳共享深度會隨任務而變。[Cross-stitch Networks](https://openaccess.thecvf.com/content_cvpr_2016/html/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.html)

這類方法通常增加參數、記憶體與 routing/mixing 操作，對 FPGA 第一版不划算。比較務實的 fallback 是從「共享 backbone + neck」退一步成「只共享 backbone」，而不是立刻導入複雜 adapter 或 MoE。

### 1.5 條件式執行／task routing

研究上的 routing networks 通常是模型依 input 動態選 module，以減少 task interference 或 conditional compute。[Routing Networks](https://openreview.net/forum?id=ry8dvM-R-) 與 [Dynamic Routing Networks](https://openaccess.thecvf.com/content/WACV2021/html/Cai_Dynamic_Routing_Networks_WACV_2021_paper.html) 都屬於此類。

本案的 MUX 更簡單：task 是外部已知控制訊號，選擇 `detect_only`、`pose_only` 或 `both`，不必再訓練 router。這種 static task routing 對可驗證性和 FPGA 都比較友善。

## 2. 代表性系統與它們提供的證據

| 系統／工作 | 共享方式 | 與本案的關聯 |
|---|---|---|
| [Mask R-CNN, ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html) | shared backbone/RoI features + parallel instance heads | 證明 detection 與 keypoint branch 可在同框架共享大部分計算。 |
| [YOLO-Pose](https://arxiv.org/abs/2204.06806) | one-stage、bbox 與 keypoints 聯合回歸 | 證明 YOLO 型單次 forward 可聯合 localize instance 與 keypoints，並用 OKS-aligned loss。 |
| [MultiTask-CenterNet, ICCVW 2021](https://openaccess.thecvf.com/content/ICCV2021W/ERCVAD/html/Heuer_MultiTask-CenterNet_MCN_Efficient_and_Diverse_Multitask_Learning_Using_an_Anchor_ICCVW_2021_paper.html) | anchor-free shared network + detection / segmentation / human-pose tasks | 與「共享 neck 特徵、不同 dense heads」最接近的公開案例。 |
| [UberNet, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Kokkinos_Ubernet_Training_a_CVPR_2017_paper.html) | shared CNN，多任務來自 diverse datasets | 證明任務不必擁有同一套資料；可按任務／資料集非同步訓練共享網路。 |
| [Partially Annotated MTL, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Learning_Multiple_Dense_Prediction_Tasks_From_Partially_Annotated_Data_CVPR_2022_paper.html) | shared encoder + task heads；每張圖只具部分 task labels | 明確定義 partial-label MTL；vanilla supervised masked-loss 是合理 baseline，更進階才用 cross-task consistency/pseudo supervision。 |
| [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) | 各 task 有官方獨立模型/head；Detect 是 end-to-end dual-head 訓練 | 提供可重用的 Detect、Pose26 與多尺度 neck，但官方 API 沒有現成的 Detect+Pose 多任務 trainer。 |

一個重要的名稱陷阱：YOLO26 文件所說的「dual-head detection」是 **one-to-one 與 one-to-many detection heads**，目的是 end-to-end/NMS-free 訓練與部署選擇；它不是本案的 Detect + Pose 兩任務 head。[YOLO26 end-to-end guide](https://docs.ultralytics.com/guides/end2end-detection/)

## 3. 不同資料集與缺失標註怎麼訓練

### 3.1 正確的基本原則：未知不是背景

對 COCO-person batch：

```text
features = shared(image)
loss = L_detect(person_head(features), person_targets)
# pose head 不 forward，或 forward 但 loss 權重為 0
```

對 BBT-pose batch：

```text
features = shared(image)
loss = L_pose(pose_head(features), ball_bat_targets)
# person head 不 forward，或 forward 但 loss 權重為 0
```

不能在 COCO-person 圖上把「沒標的 ball/bat pose」當負樣本，也不能在 BBT 圖上把「沒標 person」當背景來監督 person head。CVPR 2022 的 partial-label MTL 工作將「不是每張影像都有所有 task labels」明確視為 partially-supervised problem，並以只在可用標註上計算 supervised task loss 作為基準。[Li et al., 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Learning_Multiple_Dense_Prediction_Tasks_From_Partially_Annotated_Data_CVPR_2022_paper.pdf)

### 3.2 Sampling 不應照原始資料量

COCO 遠大於約 6k 張的 BBT train set。若直接 concat datasets，shared trunk 幾乎會被 person task 主導。實務上應先採 **task-balanced sampling**：例如每個 update 取一個 Detect batch 與一個 Pose batch，或以固定 1:1 task probability 交替。UberNet 專門處理 diverse datasets 的多任務訓練，重點之一就是對不同 task/dataset 的更新做顯式排程，而非假設資料完全對齊。[UberNet paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Kokkinos_Ubernet_Training_a_CVPR_2017_paper.pdf)

因 BBT 小很多，1:1 會重複看到 BBT；因此需另外控制 augmentation、epoch 定義和 overfitting。建議用「optimizer updates」而非「COCO epochs」作為統一訓練進度單位。

### 3.3 一次 update 看兩個 task，較適合做梯度診斷

推薦的第一版 update：

```text
zero_grad()
L_det  = forward(COCO batch)
L_pose = forward(BBT batch)
L_total = lambda_det * L_det + lambda_pose * L_pose
backward(L_total)
optimizer.step()
```

這需要共享 trunk forward 兩次，訓練成本不會像 inference 一樣只算一次，但每次更新能同時看到兩個任務。它也讓 `cos(g_det, g_pose)`、gradient norms、GradNorm、PCGrad 等方法有一致的比較基礎。若顯存不足，可用 gradient accumulation，但要確認兩個 task 的梯度在同一次 `optimizer.step()` 前累積。

單純一個 step Detect、下一個 step Pose 也可行，且更省顯存；但 shared BatchNorm statistics 與 optimizer momentum 會更受 task order 影響，PCGrad 也無法在同一步比較兩任務梯度。

## 4. Loss balancing 與 gradient conflict

建議由簡到難，不要第一天就同時加入多種方法。

### 4.1 固定權重 baseline

先將每個 task loss 在其單任務 baseline 的初期／穩定期尺度記錄下來，再設定 `lambda_det`、`lambda_pose`，使 shared layer 的 gradient norms 在相近量級。`1:1` 是實驗起點，不是自然定律。

### 4.2 Uncertainty weighting

Kendall 等人用每個任務的 homoscedastic uncertainty 學習 loss 權重，能處理 classification/regression 不同單位和尺度的問題。[Kendall et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html) 優點是容易加入；缺點是 YOLO task loss 本身已含 box/class/keypoint 等子項，需清楚決定是在 task 層或子 loss 層加權。

### 4.3 GradNorm

GradNorm 直接調整 task weights，讓各任務在 shared layer 的 gradient magnitudes 與相對訓練速度取得平衡；論文建議只在最後一個 shared layer 計算，降低成本。[GradNorm, ICML 2018](https://proceedings.mlr.press/v80/chen18a.html)

適合情況：Pose loss 長期被 Detect 梯度壓過，或兩任務 learning curve 速度差很多，但梯度方向不一定衝突。

### 4.4 PCGrad

PCGrad 在兩任務梯度 cosine similarity 為負時，把衝突分量投影掉；它只修改 shared parameters 的梯度，與架構無關。[PCGrad, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)

適合情況：量測顯示 shared neck/backbone 經常出現負 cosine similarity，且 joint model 相對兩個單任務 baselines 有明顯掉點。它需要分別取得 task gradients，因此訓練時間與記憶體會增加。

### 4.5 建議的啟用順序

1. 固定 task sampling + 固定 loss weights。
2. 記錄 shared 最後一層的 task gradient norm 與 cosine similarity。
3. 只有 magnitude 失衡：試 GradNorm。
4. 經常方向衝突且精度掉：試 PCGrad。
5. 仍有 negative transfer：減少共享範圍（例如 task-specific neck），再考慮 cross-stitch/adapters。

## 5. 初始化與訓練 schedule

學術與實務上常見的安全流程是「先單任務，再聯合微調」，因為它同時提供可靠 baseline 與較好的 head 初始化：

1. **Detect baseline**：YOLO26m → COCO-person。
2. **Pose baseline**：YOLO26m-pose → BBT ball/bat，`kpt_shape=[2,3]`。
3. **建立 combined model**：兩個 heads 分別載入自己的 baseline；shared trunk 只能選一份初始化，不能把兩份不同 trunk checkpoint 直接拼成同一份權重。
4. **短暫 freeze shared trunk**：只讓 heads 適應 combined wrapper/output contract。
5. **unfreeze neck，再 unfreeze backbone**：使用較低 learning rate 做 task-balanced joint fine-tuning。
6. **若 joint 精度低於 baseline**：才做 loss/gradient 方法與共享深度 ablation。

shared trunk 初始化至少要做兩個 ablation：

- 從 Detect baseline trunk 開始：可能有較強的通用 COCO/person 特徵。
- 從 Pose baseline trunk 開始：可能保留小球與細長球棒所需的 domain-specific features。

也可用官方 YOLO26m pretrained trunk 當中立第三組。不可只選一組後把結果解讀為架構本身的優劣。

### BatchNorm／資料分布

COCO 與棒球影像分布不同；共享 BatchNorm 的 running statistics 會混合兩域。第一版可保留共同 BN，但需記錄 task-wise validation。若不穩定，依序嘗試：

1. joint fine-tune 後 freeze BN statistics；
2. task-specific BN statistics/affine；
3. 改用不依 batch statistics 的 normalization。

task-specific BN 很便宜，但 FPGA MUX 需同時切換對應參數，會增加控制與驗證面。

## 6. MUX／條件執行真正能省什麼

設 shared 計算為 `C_shared`，兩 heads 分別為 `C_det` 與 `C_pose`：

```text
兩個獨立模型：2*C_shared + C_det + C_pose
combined, both：C_shared + C_det + C_pose
combined, detect_only：C_shared + C_det
combined, pose_only：C_shared + C_pose
```

因此：

- 同一 frame 要兩種結果：shared features 算一次，兩 heads 都要算；MUX 只是在輸出或時間上依序 dispatch，不能省掉其中一個必要 head。
- 一個 frame 只要一種結果：外部 task selector 可完全 clock/enable-gate 未選 head，這才有 head-level compute/power savings。
- 如果 FPGA 用一套 MAC engine 時分複用兩 heads，節省的是硬體面積，不一定減少 total operations 或 latency。
- 若 feature maps 寫入外部 DRAM 再依序跑 heads，可能把省下的算術成本換成 memory traffic；需量測 on-chip buffer 是否容得下 P3/P4/P5 features。

研究中的 dynamic routing 是 input-dependent learned decision，雖可降低平均 FLOPs，但會增加 router、控制流與最壞情況 latency 驗證。[Dynamic Routing Networks](https://openaccess.thecvf.com/content/WACV2021/html/Cai_Dynamic_Routing_Networks_WACV_2021_paper.html) 本案已有外部 task request，不需要學習式 router；靜態三模式 `detect_only | pose_only | both` 更合適。

## 7. YOLO26 特有的部署注意事項

YOLO26 Detect 訓練時同時含 one-to-many 與 one-to-one heads；推論預設使用 one-to-one、NMS-free 輸出。官方說明 `fuse()` 後會移除訓練用 one-to-many head，故參數與 FLOPs 應以 fused inference graph 計算。[YOLO26 model docs](https://docs.ultralytics.com/models/yolo26/)

Pose26 原始碼顯示：

- `Pose26` 的 box/class 結構繼承 `Detect`；另有 keypoint 與 sigma branches。
- sigma 與 RealNVP flow 用於訓練；`fuse()` 會移除 one-to-many、sigma 與 flow 等 inference 不需部分。
- end-to-end postprocess 會用 TopK 選取預測。

來源：[Ultralytics `Pose26` implementation](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py)

這代表 FPGA 比較必須使用 **fused/exported inference graph**，不可拿 training checkpoint graph 的參數／算量直接當部署成本。官方也明確指出某些 export backend 因 `torch.topk` 等算子限制會退回 traditional output；未知 FPGA toolchain 時，應同時保留：

- `end2end=True`：NMS-free，但需確認 TopK/gather 支援；
- `end2end=False`：dense output + CPU/FPGA NMS，graph 較傳統但後處理有成本。

來源：[YOLO26 end-to-end export notes](https://docs.ultralytics.com/guides/end2end-detection/)

## 8. 量化與 FPGA

### 8.1 建議精度階梯

1. FP32 PyTorch：確認 joint training 正確。
2. FP16 GPU/TensorRT：驗證實際 latency/VRAM，不把它誤稱為 FPGA 結果。
3. INT8 PTQ：以快速 feasibility check 為主。
4. INT8 QAT：若 small-object/keypoint AP 明顯下降，作正式候選。
5. 板卡與工具鏈確定後，再研究 INT4/混合精度。

Vitis AI 官方流程指出 PTQ 會用代表性樣本校正 activations，當精度不理想時可用 QAT 微調量化權重。[Vitis AI model development](https://xilinx.github.io/Vitis-AI/3.0/html/docs/workflow-model-development.html) FINN 則專為 quantized neural networks 的 dataflow FPGA 架構設計，搭配 Brevitas 與 QONNX；官方建議 1–8 bit integer、symmetric weights，支援 PTQ 與 QAT。[FINN documentation](https://finn.readthedocs.io/en/latest/) [FINN FAQ](https://finn-dev.readthedocs.io/en/latest/faq.html)

### 8.2 Calibration 必須涵蓋兩任務域

shared backbone 的 activation range 同時受到 COCO-person 與 BBT 場景影響；兩 heads 又各有不同輸出分布。因此 calibration set 應 task-balanced，至少涵蓋：

- COCO person 的尺寸、遮擋、背景分布；
- BBT 的小球、細長球棒、motion blur、室內外與不同曝光；
- `detect_only`、`pose_only`、`both` 三條實際 export/inference paths。

若只用 COCO calibration，Pose head 與棒球域 activation range 沒被代表；只用 BBT 也同理。

### 8.3 混合精度的優先位置

球很小、棒端點位置對 regression 誤差敏感。若全 INT8 掉點，先保留以下位置較高精度，而非把整網升位元：

- Pose head 最後的 coordinate regression conv；
- Detect/Pose head 的輸出 scale/decode；
- 必要時第一層與最後一層。

但能否採 mixed precision 取決於最終 FPGA compiler/IP；FINN 可探索任意整數位寬，Vitis AI 的既定 NPU/DPU 流則要遵循其 operator 與 datatype 支援。Vitis AI BYOM 會把 unsupported layers 切到 CPU subgraph，因此「模型能轉 ONNX」不等於「整圖都在 FPGA」。[Vitis AI BYOM](https://vitisai.docs.amd.com/en/gen-1/docs/byom.html)

### 8.4 Operator discipline

在板卡未定前，優先維持 Conv/BN(fused)/activation/concat/upsample 等常見算子，避免把自訂 Python control flow、動態 shape 或 learned router 放進 graph。真正選板後，先用 compiler report 查 supported/unsupported operator，再決定是否改 head/decode。FINN 對未支援 custom layer 需要自行實作 HLS/RTL operator，這可能比模型訓練本身更費工。[FINN FAQ](https://finn-dev.readthedocs.io/en/latest/faq.html)

## 9. 本專案建議實驗矩陣

### Phase A：單任務基準

| ID | 模型 | 資料 | 目的 |
|---|---|---|---|
| D0 | YOLO26m Detect, person-only | COCO | person AP 與成本上限基準 |
| P0 | YOLO26m Pose26, ball/bat, 2×3 kpts | BBT | ball/bat box AP、pose AP、端點誤差基準 |

### Phase B：共享模型

| ID | 共享範圍 | 訓練法 | 目的 |
|---|---|---|---|
| M0 | backbone + neck | 1:1 task-balanced，固定 loss weights | 最小可行 combined baseline |
| M1 | backbone + neck | GradNorm | 檢查 loss/learning-rate imbalance |
| M2 | backbone + neck | PCGrad | 檢查 gradient conflict |
| M3 | backbone only | 最佳 balancing 方法 | 檢查 neck 是否應 task-specific |

### Phase C：部署圖與量化

每個候選至少測：

- modes：`detect_only`、`pose_only`、`both`；
- precision：FP32、FP16、INT8 PTQ、INT8 QAT；
- output path：YOLO26 end-to-end 與 traditional/NMS；
- metrics：參數量、MACs/FLOPs、峰值 VRAM、batch=1 latency、吞吐、export graph operator list；
- accuracy：person box mAP50-95、ball/bat box mAP50-95、pose mAP50-95，以及以 pixel/normalized distance 計算的 keypoint error。

所有 latency 要固定輸入大小、batch=1、warm-up、同步計時，並分開報告 model forward 與 decode/postprocess。FLOPs 降低不保證 FPGA latency 等比例降低，因為 on-chip memory、DRAM traffic、並行度與 unsupported-op fallback 都可能成為瓶頸。

## 10. 對本案的最終建議

建議先做以下 architecture contract：

```text
SharedYOLO26m
  encode(image) -> [P3, P4, P5]
  detect(features) -> person detections
  pose(features) -> ball/bat boxes + 2x(x,y,v)
  forward(image, task="detect" | "pose" | "both")
```

訓練則不要讓 MUX 隨機猜任務；dataset 已知 task id，直接路由到對應 head，missing task loss 為 0。每個 optimizer update 使用一個 COCO batch 和一個 BBT batch，初始 task ratio 1:1；先做固定權重 M0，並記錄 task gradient norm/cosine。只有數據證明失衡或衝突，才分別啟用 GradNorm 或 PCGrad。

推論的 MUX 定義為外部控制：

- `detect_only`：shared + Detect head；
- `pose_only`：shared + Pose head；
- `both`：shared 一次 + 兩 heads；

這同時符合 GPU benchmark 與未來 FPGA 的硬體路由語意。FPGA 板卡未定前，不應先承諾某個 compiler 或低於 INT8 的位寬；但現在就應固定 tensor shapes、維持簡單 operators、提供 fused graph，並保留 end-to-end/TopK 與 traditional/NMS 兩種 export 路徑。

最後，論文式判定成功不能只看「總參數少了」：共享模型必須在可接受的 person AP 與 ball/bat pose AP 損失內，對比 `D0 + P0` 確實降低部署 graph 的參數、運算、記憶體與 batch=1 latency。若完全共享 neck 掉點明顯，`M3`（只共享 backbone）可能是較好的 Pareto point。

## Primary sources

- He et al., [Mask R-CNN](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html), ICCV 2017.
- Maji et al., [YOLO-Pose](https://arxiv.org/abs/2204.06806), 2022.
- Heuer et al., [MultiTask-CenterNet](https://openaccess.thecvf.com/content/ICCV2021W/ERCVAD/html/Heuer_MultiTask-CenterNet_MCN_Efficient_and_Diverse_Multitask_Learning_Using_an_Anchor_ICCVW_2021_paper.html), ICCVW 2021.
- Kokkinos, [UberNet](https://openaccess.thecvf.com/content_cvpr_2017/html/Kokkinos_Ubernet_Training_a_CVPR_2017_paper.html), CVPR 2017.
- Li et al., [Learning Multiple Dense Prediction Tasks From Partially Annotated Data](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Learning_Multiple_Dense_Prediction_Tasks_From_Partially_Annotated_Data_CVPR_2022_paper.html), CVPR 2022.
- Kendall et al., [Multi-Task Learning Using Uncertainty to Weigh Losses](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html), CVPR 2018.
- Chen et al., [GradNorm](https://proceedings.mlr.press/v80/chen18a.html), ICML 2018.
- Yu et al., [Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html), NeurIPS 2020.
- Misra et al., [Cross-stitch Networks](https://openaccess.thecvf.com/content_cvpr_2016/html/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.html), CVPR 2016.
- Cai et al., [Dynamic Routing Networks](https://openaccess.thecvf.com/content/WACV2021/html/Cai_Dynamic_Routing_Networks_WACV_2021_paper.html), WACV 2021.
- Ultralytics, [YOLO26 model](https://docs.ultralytics.com/models/yolo26/), [architecture](https://docs.ultralytics.com/guides/yolo-architecture/), [end-to-end export](https://docs.ultralytics.com/guides/end2end-detection/), and [`Pose26` source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py).
- AMD/Xilinx, [Vitis AI model-development workflow](https://xilinx.github.io/Vitis-AI/3.0/html/docs/workflow-model-development.html), [Vitis AI BYOM](https://vitisai.docs.amd.com/en/gen-1/docs/byom.html), [FINN](https://finn.readthedocs.io/en/latest/), and [Brevitas](https://github.com/Xilinx/brevitas).
