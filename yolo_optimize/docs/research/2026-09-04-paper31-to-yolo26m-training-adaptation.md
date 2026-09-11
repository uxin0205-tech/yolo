# 指定論文第 3.1 節如何真正適配目前 YOLO26M／Full35

- 日期：2026-09-04
- 狀態：研究完成／尚未執行實驗
- 範圍：逐項核對指定論文第 3.1.1–3.1.4、原始 MSFA 一手來源，以及本地 YOLO26M／Full35 的 graph、trainer、設定、checkpoint、結果與資料血緣
- 執行邊界：未啟動訓練或驗證；未修改 production code、dataset、label、split 或 checkpoint
- 指定材料：使用者提供論文，第 3.1.1–3.1.4 節（本機／歷史參照：`../references/papers/sar-yolox-multiprecision.pdf`；未隨本次報告發布）

## 決策摘要

論文有四個部分，但放到目前 YOLO26M 後，結論不是「四個都做」：

| 論文第 3.1 節 | 原做法 | 對目前 Full35 的判斷 | 本專案應改成 | 優先度 |
|---|---|---|---|---:|
| 3.1.1 模型／算力取捨 | 以約 33G operations/frame 預算選 YOLOX-S | 原則可用，數字與模型不可搬 | 鎖定現有 YOLO26M、3-channel RGB、BinaryQK/Bit-True 契約；新方法只准增加 training cost | 必守條件 |
| 3.1.2 短 LR 篩選 | 5 個固定 LR，各跑 9 epochs | 不應照抄；Full35 已有分層 LR、warmup、cosine、plateau 與 staged scope | 只在新增 trainable scope 不穩時做「犧牲性、stage-local LR diagnostic」；正式 arm 沿用現行 optimizer/LR | 條件式 |
| 3.1.3 COCO→DOTA→SARDet | 用光學遙測資料作 SAR domain/model bridge | DOTA 不適用；COCO person-only 與 COCO80 仍是同一 RGB/資料域 | 不新增 DOTA 類資料；若未來真要測，只能做保留 COCO80 labels 的 semantic replay sampler，且另案 | 延後／通常不做 |
| 3.1.4 WST/HOG/Canny | 灰階原圖與固定 filter channels 拼接後餵 backbone | 不適合直接搬；會改第一層、捨棄色彩且推論每張重算 | RGB 主路不變，在 P3 加「training-only、box-aware 9-bin HOG companion target」 | **本篇首選新候選** |

因此，本篇從論文衍生出的第一個、可單獨驗證的新方向是：

> **P3 Box-Aware HOG Companion Supervision**：HOG 只產生訓練標靶，不進 inference input；用一個暫時的 `1×1 Conv(256→9)` 從現有 P3 feature 預測局部梯度方向分布，J1／J2 early 有效、J3 前歸零並移除。

它和現有兩個方向必須分開：

- BinaryQK 的 FP-teacher／ranking model-gap bridge 已在 [Q0 BinaryQK 精度恢復](<../../proposals/binaryqk-accuracy-recovery/README.md>) 定義，本篇不重複建立 teacher/KD arm。
- Detect/Pose shared-gradient conflict 已在 [training-conflict-safe](<../../proposals/training-conflict-safe/README.md>) 定義，本篇只記錄 HOG auxiliary gradient，不在同一首輪加入 projection。
- COCO person-only 是另一個 head／資料視圖決策；目前正式 Full35 仍是 COCO80 Detect，不能先假定 person-only 已成為 baseline。

---

## 1. 證據標籤與不能混淆的名稱

本文使用四種標籤：

- **【指定論文】**：本地 PDF 第 3.1 節直接寫出的設定或結果。
- **【一手來源】**：原始論文、作者官方程式或正式框架程式。
- **【本地事實】**：目前工作區正式 artifacts／trainer／registry 可直接核對的內容。
- **【提案】**：尚未在本地訓練驗證的設計；只是假說，不是已知增益。

另須分清：

- `MSFA`：指定論文／SARDet 工作的 multi-stage + filter augmentation。
- `MASF`：本地 YOLO26M layer 16 的 P3 模組。

兩者只有縮寫相近，機制不同；本地 MASF 的正負結果不能當成 MSFA 訓練方法的證據。

## 2. 指定論文第 3.1 節原方法

### 2.1 第 3.1.1：先按部署預算選模型

【指定論文】論文假設 UAV 約有 `1 TOPS` 可分給 30 FPS 偵測，因而得到約：

\[
\frac{10^{12}\ \text{operations/s}}{30\ \text{frames/s}}
\approx 33.3\times10^9\ \text{operations/frame}.
\]

論文在多個 detector 間取捨後選 YOLOX-S；表中 YOLOX-S 為 `8.94M` parameters、`20.82G` FLOPs、`82.3 mAP@50`。這個段落真正可轉用的是「先固定部署圖與預算，再談訓練增益」，不是把目前模型換成 YOLOX-S。

### 2.2 第 3.1.2：5×9 的 LR 短跑

【指定論文】固定 batch size 16、AdamW `betas=(0.9,0.999)`，比較：

```text
{1e-4, 5e-5, 2.5e-5, 1.25e-5, 6.25e-6}
```

每個候選訓練 9 epochs，依 training loss 與 validation mAP50 選 `5e-5`；再由曲線約在 epoch 30 飽和，設定正式 30 epochs。初選成本實際是 `5×9=45 candidate-epochs`，它是小型 grid，不是一次低成本 LR range test。

【一手來源】Smith 的 LR range test 是在一段短跑中連續提高 LR，從 loss 行為估合理界線；原始證據主要是分類模型，不能把它當 detector AP 保證。[Cyclical Learning Rates](https://arxiv.org/abs/1506.01186)

【一手來源】AdamW 的核心是把 weight decay 與 loss-gradient update 解耦；它沒有證明「使用 AdamW 就一定比其他 optimizer 泛化好」。[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

### 2.3 第 3.1.3：COCO→DOTA→SARDet

【指定論文】比較：

```text
conventional：COCO → SARDet
multi-stage： COCO → DOTA → SARDet
```

表中 mAP50 由 `82.3` 到 `83.4`，即 `+1.1`。指定論文沒有在同一消融中等步數控制「多看資料／多做更新」與「DOTA bridge 語意」的各自貢獻，因此不能把 `+1.1` 當成本案預期收益。

【一手來源】DOTA 是 aerial image detection dataset，包含高解析航空影像與任意方向四邊形標註；它之所以可能橋接 optical remote sensing 與 SAR，並不代表它能橋接一般 RGB COCO 與 RGB 棒球資料。[DOTA 原始論文](https://arxiv.org/abs/1711.10398)

【一手來源】原始 MSFA 更精確的流程是 ImageNet backbone → optical remote-sensing detector → SAR detector，而且完整 detector 作 model bridge 通常比只搬 backbone 更有效；其問題設定是 optical/SAR domain 與 detector-structure gap。[MSFA／SARDet-100K 論文](https://papers.neurips.cc/paper_files/paper/2024/hash/e7eb8128eb26eafbe901348df1dbacdc-Abstract-Conference.html)、[論文 PDF](https://papers.neurips.cc/paper_files/paper/2024/file/e7eb8128eb26eafbe901348df1dbacdc-Paper-Conference.pdf)

### 2.4 第 3.1.4：WST、HOG、Canny

【指定論文】在 multi-stage 設定下表列：

| Filter | mAP50 | Params | GFLOPs |
|---|---:|---:|---:|
| WST | 83.4 | 8.94M | 35.38 |
| HOG | 83.9 | 8.94M | 22.11 |
| Canny | 83.7 | 8.94M | 21.63 |

該 YOLOX 實驗選 HOG；差距只有 `0.2–0.5 mAP50`，且未提供跨 seed 變異。原始 MSFA 主結果則偏向 WST，已說明 filter winner 依資料、backbone 與 recipe 而變。

【一手來源】官方 `MSFA.py` 在 filter 模式先做 `x.mean(1, keepdim=True)`，固定 filter 置於 `torch.no_grad()`，再把 gray identity 與 filter output concatenate。HOG、Canny、WST 分別增加 9、6、81 channels，backbone `in_channels` 也隨之改成 10、7、82。[MSFA 官方實作](https://github.com/zcablii/SARDet_100K/blob/main/MSFA/msfa/models/backbones/MSFA.py)

所以論文的 filter 不是「訓練時偶爾做、推論不用」的隨機 augmentation，而是模型輸入 representation；推論影像仍要重算。

---

## 3. 本地 YOLO26M／Full35 現況：不是空白 baseline

### 3.1 正式 graph

【本地事實】Factory report 已驗證：

- shared graph layers `0–22` 只算一次，layer 23 是 Detect／Pose 雙 head router；
- head inputs 為 `[16,19,22]`，即 P3/P4/P5；
- channels 為 `[256,512,512]`，strides 為 `[8,16,32]`；
- Detect 是 COCO80；Pose 是 ball/bat 2 類，`kpt_shape=[2,3]`；
- model `end2end=true`；
- shared model `26,529,701` parameters，兩個獨立模型合計 `45,580,762`，少 `41.796%`。

證據：[factory-report.json](<../../../yolo_combine/final/full35/outputs/training/factory-report.json>)、[fusion_model.py](<../../../yolo_combine/src/yolo_combine/fusion_model.py>)

目前 P3 不是原生裸 C3k2。建模程式把 layer 16 的 class 原地升級，`forward` 為：

```text
layer16 C3k2 output → p3_masf → layer16 output
                               ├→ P3 head input
                               └→ downstream P4/P5 graph
```

證據：[YOLO26M P3 graft](<../../../yolo_combine/final/full35/source_bundle/code/achitechure_1/model.py>)

完整現況可表示為：

```text
RGB(3ch)
  ↓
YOLO26M layers 0…15
  ↓
layer16：C3k2 → P3 MASF → F3 (256ch, stride 8) ─────┐
  │                                                  │
  └→ layers17…19 → F4 (512ch, stride16) ────────────┤
                    └→ layers20…22 → F5 (512ch,32) ─┤
                                                     ↓
             layer23 DualHeadPredictionModule([F3,F4,F5])
                     ├→ COCO80 Detect
                     └→ BBAT ball/bat Pose
```

這正好提供一個不改 inference stem 的 P3 training seam：`DualHeadPredictionModule.forward(features)` 已收到三尺度 list，`features[0]` 就是正式 head 看見的 post-MASF P3。

### 3.2 正式 J0→J3 不是論文的 dataset bridge

【本地事實】現行 stage 與實跑如下：

| Stage | 實跑 | task/scope | peak LR |
|---|---:|---|---|
| J0 | 8/8 | Pose only；只開 Pose head | Pose head `2e-4` |
| J1 | 20/20 | joint；neck + Detect/Pose heads | neck `7.5e-5`、heads `2e-4` |
| J2 | 25/80 | joint；layer9+ backbone、neck、MASF、heads | `1.5e-5/7.5e-5/1.5e-4/2e-4` |
| J3 | 11/20 | joint；full low-LR refinement，可微 attention 開啟 | backbone `3.8e-6`、neck `1.9e-5`、MASF `3.8e-5`、attention `5e-7`、heads `5e-5` |

證據：[stage_policy.py](<../../../yolo_combine/src/yolo_combine/stage_policy.py>)、[resolved-config.json](<../../../yolo_combine/final/full35/outputs/training/resolved-config.json>)、[最終分析](<../../../yolo_combine/final/full35/analysis/FINAL_ANALYSIS.md>)

J0→J3 改的是「可訓練參數範圍與 LR」，資料沒有依 stage 改成 DOTA 類中介域：joint stage 一直由 COCO Detect loader 與 BBAT Pose loader 組成。因此：

```text
論文 multi-stage = dataset/domain + model bridge
本地 J0→J3      = task-head adaptation + trainable-scope bridge
```

兩者不能說成完全相同；但 gradual unfreezing、head adaptation、discriminative LR 已經是本地 baseline，也不能重新命名成新創新。

### 3.3 optimizer、scheduler 與 loss 的真實行為

【本地事實】正式 joint optimizer 是 AdamW，`weight_decay=0.00027`、`betas=(0.948,0.999)`。參數同時按 semantic role 與 decay/no-decay 分組；shared BN running statistics 固定，head BN 維持 train。[joint.yaml](<../../../yolo_combine/final/full35/code/project/variants/full35/configs/joint.yaml>)、[stage_policy.py](<../../../yolo_combine/src/yolo_combine/stage_policy.py>)

重要修正：**同一正式鏈在 stage 轉換時不會丟掉 optimizer moments。** `update_optimizer_stage()` 明寫「without discarding state」，只更新各 group LR；每個 stage 會建立新的 per-macro warmup+cosine scheduler。J3 雖是另一個 run，也從 stage-complete J2 last 做 exact-resume，恢復 optimizer、scheduler、scaler、criterion 與 RNG。[stage_policy.py](<../../../yolo_combine/src/yolo_combine/stage_policy.py>)、[_formal_training_impl.py](<../../../yolo_combine/src/yolo_combine/_formal_training_impl.py>)、[_joint_trainer_impl.py](<../../../yolo_combine/src/yolo_combine/_joint_trainer_impl.py>)

`NativeTaskLossRouter.loss_for()` 目前先 forward，再把 Ultralytics native loss items 相加成 batch-summed `raw_total`；`MacroStepEngine` 依真實圖片數與 task weight 做一次 macro update。joint 時：

\[
L_{macro}
=64\,
\frac{
1.0\,(L_d/N_d)+0.25\,(L_p/N_p)
}{1.0+0.25}.
\]

正式 exposure 是每 macro `N_d=256`、`N_p=16`。J1/J2 的 Detect logical 128 由 physical 64 microbatches 累積；J3 改 physical 32，logical exposure 不變。最後才 unscale、finite check、clip norm 10、optimizer step、scaler/EMA update。[joint_loss.py](<../../../yolo_combine/src/yolo_combine/joint_loss.py>)、[最終分析](<../../../yolo_combine/final/full35/analysis/FINAL_ANALYSIS.md>)

### 3.4 checkpoint 與資料血緣

【本地事實】Factory 初始化來源：

| 角色 | SHA-256 |
|---|---|
| Float Full35-A2 Detect/source | `af390ec43cadb5aaac47a22ebf58c710106c3f8dd143fbd2bbcb0eb2b76ebf26` |
| Pose P3 head checkpoint | `7b46c6af29723cff8ffdad96cf1220fd75975c31f99def58cfec84e80c322f99` |
| 最終 J3 inference best_joint | `d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c` |
| 最終 J3 full-resume best_joint | `26e648432388060cb75a67875d956a022114e36f731ff9b5de04953e053b413d` |

Factory 對 shared/detect-head/pose-head 分別完整載入 587/240/411 tensors，沒有 missing、unexpected 或 shape mismatch。[factory-report.json](<../../../yolo_combine/final/full35/outputs/training/factory-report.json>)

新訓練 arm 必須從 **Float source 或完整 full-resume parent** 分支；Bit-True inference checkpoint 是部署／評估 artifact，不應當成有 optimizer state 的訓練入口。若要在 J1 起測本提案，所有 arm 必須使用同一個「已註冊但 J0 完全 dormant 的 auxiliary head」所產生的 J0 stage-complete experimental parent。原因是原正式 J0 optimizer 沒有 `hog_aux` group，直接塞入新 group 再宣稱 exact-resume 會改變 optimizer schema。

最安全作法是從相同 Float factory source 跑一次共同 J0：三臂在 J0 都不計算 auxiliary forward、`mu=0`、`hog_aux lr=0`，並先證明 native output/update 與原 J0 相等；再把同一 full-resume SHA 分給 F0/F1/F2。若未來實作 parameter-name state migration，則必須逐 group 證明原 optimizer moments逐值不變、新 aux state為空，不能依 PyTorch group順序猜配。從最終 J3 再短微調只能稱 post-final refinement，不能替代 J1/J2 causal test。

正式資料是：

- COCO2017 Detect：`118,287` train、`5,000` val、80 類；
- immutable BBAT5 v1：`6,647` images、`2,578` source groups，formal `5,964/683`，search `5,364/600` 且完全位於 formal train；沒有 test split。

證據：[COCO YAML](<../../../coco2017.yaml>)、[BBAT5 registry](<../../../configs/datasets/bbat5-v1.yaml>)、[BBAT5 資料契約](<../../../docs/agents/bbat5-datasets.md>)

任何 HOG target 只能 on-the-fly 產生；不得重切 BBAT5、另建 labels 或把 runtime view 變成新資料版本。

### 3.5 現有結果能支持什麼

【本地事實】目前只有 seed 0。最終 J3 相對 standalone：

| 指標 | Standalone | J3 | 差值 |
|---|---:|---:|---:|
| COCO overall box AP50-95 | 0.506401 | 0.498022 | -0.008379 |
| COCO person box AP50-95 | 0.626186 | 0.620381 | -0.005804 |
| BBAT box overall | 0.630964 | 0.630036 | -0.000928 |
| BBAT pose overall | 0.912161 | 0.903717 | -0.008443 |
| ball box | 0.510747 | 0.507437 | -0.003310 |
| bat box | 0.751180 | 0.752634 | +0.001454 |
| ball pose | 0.876263 | 0.859909 | -0.016354 |
| bat pose | 0.948058 | 0.947526 | -0.000532 |

J3 joint score 比 J2 高 `0.0011365`，但 ball box 與 ball pose 分別比 J2 低 `0.003429`、`0.000890`。這表示現行 staged training 有局部收益，但不能由 joint score 推論每項都改善。[FINAL_ANALYSIS.md](<../../../yolo_combine/final/full35/analysis/FINAL_ANALYSIS.md>)、[SUMMARY.json](<../../../yolo_combine/final/full35/analysis/SUMMARY.json>)

本地 durable gradient logs 另顯示：

| Stage | n | cosine mean | 負 cosine | Pose/Detect norm ratio median |
|---|---:|---:|---:|---:|
| J1 | 93 | 0.042682 | 30/93 | 3.6785 |
| J2 | 116 | 0.018512 | 38/116 | 0.7506 |
| J3 | 51 | 0.026841 | 21/51 | 0.6656 |

這是「另有 gradient-conflict 候選」的證據，不是 HOG 一定有效的證據。HOG 首輪若同時加入 projection，任何 AP 差都無法歸因，因此本篇只新增 auxiliary-gradient logging，不改 gradient merge。

---

## 4. 逐項適配判斷 A：模型／算力

### 原方法是否適用

適用的是約束方式，不是 YOLOX-S 本身。目前 Full35 已綁定：

- YOLO26M end-to-end graph；
- RGB 3-channel stem；
- P3 MASF；
- BinaryQK/Bit-True 硬體契約；
- P3/P4/P5 雙 head 一次共享抽 feature。

改成 YOLOX-S、改 input channels 或永久增加 filter branch，都會把「訓練方法」變成「新架構」，也會破壞現有 checkpoint tensor shape，無法用同一 baseline 歸因。

本地 Full35-A2 Detect source 在 imgsz 640 的既有 profile 是 `76.3789312 GFLOPs`，但這個計數口徑、模型圖和論文 YOLOX-S 的 `20.82G` 不同，不能直接相減或用來宣稱超過／符合論文 UAV 預算。[independent-history.json](<../../../yolo_combine/final/full35/metrics/independent-history.json>)

### 對新方法的硬 gate

HOG companion 可增加 training wall time／VRAM，但正式 materialized inference 必須滿足：

1. input 仍為 `[B,3,H,W]`；
2. layer 0–23 正式 parameter names、shapes 與 F0 baseline 一致；
3. auxiliary head／target generator 不在 inference state dict、ONNX/engine graph；
4. Float 與 Bit-True output schema不變；
5. 同設定 batch-1 latency、FLOPs、正式參數量不得因 HOG arm 增加。

這些不通過，就不能稱 training-only，也應立即停止。

## 5. 逐項適配判斷 B：LR screening／optimizer

### 5.1 為何不能搬 `5e-5`

論文的 `5e-5` 同時綁定 YOLOX-S、batch 16、單一 detector、其資料與 scheduler。Full35 是多 task、多 role LR：

```text
J1：neck 7.5e-5；heads 2e-4
J2：backbone 1.5e-5；neck 7.5e-5；MASF 1.5e-4；heads 2e-4
J3：backbone 3.8e-6；neck 1.9e-5；MASF 3.8e-5；
    attention 5e-7；heads 5e-5
```

把所有 group 改成 `5e-5` 會同時提高 J3 backbone/attention、降低 head LR，並摧毀既有 discriminative ratios；它不是單因子比較。

本地另有 attention-only LR sweep：`5e-6/1e-5/2e-5` 中最低者最好，但只比同輪 parent 高約 `0.000194`，最後仍保留 zero-train 正式 winner。它只證明該 attention scope 對高 LR 敏感，數值不能搬到 joint Full35。[Attention TRAINING.md](<../../../yolo_attention_final/final/TRAINING.md>)、[RESULTS.md](<../../../yolo_attention_final/final/RESULTS.md>)

### 5.2 本案應如何改

**第一輪不重掃 optimizer、不換 OneCycle、不全域重找 LR。** HOG treatment/control 都沿用現行 AdamW、role LR、warmup、cosine、plateau、task weight 與 macro exposure；這樣 F2−F0/F1 才是 auxiliary target 差。

新增的一個隨機初始化 `hog_aux` optimizer role，可先比照 J1/J2 task-head LR `2e-4`，J0/J3 為 0；這是待驗證初值，不是已知最佳值。其 weight 用現行 decay，bias no-decay。

只有下列任一情況出現，才啟動 LR diagnostic：

- `hog_aux` update ratio 比兩個正式 head 高或低一個數量級；
- auxiliary gradient 經常觸發 clip/overflow；
- native task loss 在開啟 auxiliary 後立刻非有限或劇烈偏離 matched control；
- 為 HOG 實作被迫改變 trainable scope／effective batch。

### 5.3 精確 diagnostic seam

在同一 J0 exact-resume parent 的**犧牲性 clone**中：

\[
\eta_{aux}(s)=\eta_{min}
\left(\frac{\eta_{max}}{\eta_{min}}\right)^{s/(S-1)}
\]

只掃 `hog_aux` group；既有 backbone/neck/MASF/head LRs 固定。記錄：

\[
u_l=\frac{\lVert\theta_l^{s+1}-\theta_l^s\rVert_2}
{\lVert\theta_l^s\rVert_2+\epsilon},
\qquad
r_{aux}=\frac{\lVert\mu g_{aux}\rVert_2}
{\lVert g_{native}\rVert_2+\epsilon}.
\]

實作 seam 是 [_formal_training_impl.py](<../../../yolo_combine/src/yolo_combine/_formal_training_impl.py>) 建立 `StageWarmupCosineScheduler` 與每 macro 呼叫 scheduler 的位置；diagnostic 暫時由單調 range scheduler 接管 `hog_aux`，完成後整個 clone 丟棄。

正式候選重新載入相同 J0 parent：

- 既有 groups 恢復相同 optimizer moments；
- 新 `hog_aux` group 以相同空 state 建立；
- 正式 per-stage scheduler 照舊；
- 不把 range-run 權重接著用。

這也修正「每 stage 都 fresh optimizer」的錯誤想像：只有隔離診斷 clone 可丟棄；正式 J-stage 語義仍保留 optimizer state。

### 5.4 optimizer 是否另做 AdamW vs MuSGD

不列入本篇首輪。Detect source Full35-A2 曾用 MuSGD 10 epochs，正式 joint run 用 AdamW；它們的 checkpoint、scope、loss、資料 exposure 與訓練長度不同，不能從歷史結果推論 optimizer 勝負。[independent-history.json](<../../../yolo_combine/final/full35/metrics/independent-history.json>)

若未來另案比較，需先各自做公平 LR calibration，再以相同 parent、scope、steps、scheduler family 與 seeds 比較；那已不是 HOG 單因子實驗。

## 6. 逐項適配判斷 C：multi-stage transfer

### 6.1 為何不建議 DOTA 類 bridge

目前 Detect source和持續訓練資料都是 COCO2017 RGB；使用者另考慮的 person-only 仍是 COCO 同一影像域，只是 label/head scope 改變。BBAT5 也是一般 RGB 棒球影像，不是 SAR。於是：

```text
論文：COCO natural RGB → DOTA aerial RGB → SAR
本地：COCO80 RGB Detect + BBAT RGB Pose，共享同一 YOLO26M trunk
```

本地不存在相同 optical→SAR modality gap。硬插 DOTA 會增加 aerial texture、rotated-box annotation、head mapping、額外 steps 與資料 lineage 五個變因，卻沒有與最終任務相符的理由。

此外，BinaryQK 的 FP→binary **model gap** 已由 Q0 的 site isolation、QAT、必要時 FP-teacher ranking KD 處理；不能把同一 KD 再包裝成本篇的 MSFA bridge。

### 6.2 本地 J0→J3 已吸收了哪些思想

論文希望避免模型從 source representation 一次跳到 target。Full35 已有最接近的 model-side adaptation：

```text
factory 初始化
  ↓
J0 Pose head 適應 Detect trunk
  ↓
J1 neck + heads
  ↓
J2 late backbone + MASF
  ↓
J3 full low-LR refinement
```

因此「再做一次 gradual unfreezing」沒有新意。論文能新增的只可能是 data curriculum，但目前沒有足夠 domain-gap 證據讓它排在 HOG 前面。

### 6.3 若未來仍要測：Replay-Preserving Semantic Curriculum

這是延後候選，不是首輪。不要重建三類 head，也不要刪 COCO 非相關 labels；只在 J1 的 Detect loader 改抽樣機率：

\[
p_t(i)=(1-\rho_t)\frac{1}{N}
+\rho_t\frac{\mathbf 1[i\in R]}{|R|},
\qquad
R=\{i:\text{image }i\text{ 含 person/sports ball/baseball bat}\},
\]

並令 `rho_t` 在 J1 結束前降到 0。如此 relevant images 被多看，但每張圖仍保留全部 COCO80 annotations，uniform replay 也保留沒有 person 的真實 negative images。

精確 seam 是 [_formal_training_impl.py 的 `_loaders()`](<../../../yolo_combine/src/yolo_combine/_formal_training_impl.py>)：只替 Detect loader 加 deterministic sampler；Pose loader、BBAT YAML、labels、macro ratio與總 optimizer steps不變。

最小矩陣必須有：

| Arm | Detect sampler | 用途 |
|---|---|---|
| B0 | 現行 uniform | baseline |
| B1 | 上述 relevant semantic mixture | 測語意 curriculum |
| B2 | 相同集合大小／重複曝光的 random-image mixture | 排除「只是多看某些圖／多做曝光」 |

只有 B1 同時穩定勝過 B0 與 B2，才可說 semantic bridge 有用。B1≈B2 只能說 resampling 有效；B1<B0 即停止。

這個方向仍有三個風險：

1. COCO Detect 仍是 80 類，過度偏向三類可能造成其他類 forgetting；
2. COCO sports ball/baseball bat 的尺度與 BBAT pose keypoint 語義不同；
3. 重複抽樣會改有效樣本多樣性，容易把「多曝光」誤認成 domain bridge。

所以它排在 HOG 之後，而且若 person-only head 已另案定稿，必須用那個已凍結 baseline 重寫 B0/B1/B2，不能在本篇偷換 head。

## 7. 逐項適配判斷 D：filter augmentation

### 7.1 為何直接 concat 不適合目前模型

若照官方 MSFA 搬：

```text
目前：
RGB(3ch) → pretrained YOLO26M stem → …

直接搬 HOG：
RGB → mean(gray,1ch)
       ├→ identity gray ─┐
       └→ HOG(9ch) ─────┴→ concat(10ch) → modified stem → …
```

會有六個具體問題：

1. 第一層從 3 channels 變 10，現有 pretrained tensor shape 無法完整載入；
2. RGB identity 被 gray 取代，制服、草地、皮膚、球棒/背景的 chroma cues 可能損失；
3. HOG 重述 early edge，可能與 backbone 已學 feature 冗餘；
4. HOG cell/block pooling 可能抹掉小球與細球棒；
5. 每張 inference image 仍要重算，硬體 graph/latency 改變；
6. WST 甚至增加 81 channels，與目前記憶體壓力和最小硬體路徑更不相容。

這些是可檢驗風險，不代表 HOG 本身無用；正確改法是把它從「永久輸入」移到「暫時標靶」。

### 7.2 首選新方法：P3 Box-Aware HOG Companion

【一手來源】經典 HOG 用局部梯度方向 histogram 與 block normalization 表示形狀，最早在人形偵測顯示效果；這只支持形狀先驗，不保證 YOLO26M AP 增益。[Dalal & Triggs, 2005](https://www.cs.princeton.edu/courses/archive/fall13/cos429/papers/Dalal05.pdf)

【一手來源】MaskFeat 顯示 HOG 可作 feature prediction target，並指出 local contrast normalization 很重要；該證據來自 masked representation pretraining，不是本地 detection/pose，因此本案需要負控制。[MaskFeat](https://openaccess.thecvf.com/content/CVPR2022/html/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.html)

本案改後：

```text
                    ┌──────────────── native Detect/Pose loss
RGB(3ch) → YOLO26M → F3/F4/F5 → Dual heads
   │                  │
   │                  └→ F3(post-MASF, 256×80×80)
   │                         └→ temporary 1×1 Conv(256→9)
   │                                      ↓ predicted orientation
   └→ luminance → fixed HOG9 target ──────┤
                 + GT object mask ────────┘ companion loss

J3／部署：刪掉 HOG target generator 與 temporary head；
          RGB→YOLO26M→雙 heads 完全回到原 graph。
```

在 imgsz 640 下，P3 是 `80×80`。首測固定 HOG cell `8×8 pixels`，剛好對齊 stride 8 的 P3 cell；使用 9 個 unsigned orientation bins 與 local block normalization。這是本專案工程假設，不是論文原設定的保證。

令：

\[
Y=0.299R+0.587G+0.114B,
\qquad
T=\operatorname{sg}\!\left(
\operatorname{HOG}_{9,cell=8}(Y)
\right),
\]

`T∈R^{B×9×80×80}`，每個 cell 的 9-bin histogram 正規化為分布。P3 side head：

\[
P=\operatorname{softmax}_{bin}(h_\phi(F_3)),
\qquad
h_\phi=\operatorname{Conv}_{1\times1}(256,9).
\]

若 `M_{buv}` 是由該 task train GT boxes 投影到 P3 的 soft filled mask，`A_{buv}` 是停止梯度的局部梯度能量，單張影像 loss：

\[
\ell_{hog}^{(b)}
=-
\frac{
\sum_{u,v,k}
M_{buv}\,A_{buv}\,T_{bkuv}\log(P_{bkuv}+\epsilon)
}{
\sum_{u,v}M_{buv}A_{buv}+\epsilon
}.
\]

每個 physical batch 必須回傳：

\[
L_{hog,raw}=\sum_{b=1}^{B}\ell_{hog}^{(b)},\qquad
L_{raw}=L_{native,raw}+\mu(t)L_{hog,raw}.
\]

要「每張先做 spatial/mask normalization，再對 batch 求和」，因為目前 router 的 native `raw_total` 是 batch-summed，後面的 `JointMacroPlan` 才依真實圖片數與 task weight 縮放。若先把 HOG loss 做 batch mean、又交給 macro scale，最後一個 batch與 microbatch切法就會錯權重。

Detect batch 的 `M` 使用全部 COCO80 GT boxes；Pose batch使用 ball/bat boxes。第一輪不另給 person 權重，否則會同時改 task objective。

### 7.3 精確插入 seam

需要改動的位置在未來實作時必須同時滿足：

1. [`DualHeadPredictionModule.forward(features)`](<../../../yolo_combine/src/yolo_combine/fusion_model.py>) 已接收 `[F3,F4,F5]`；只在 training mode 把 `features[0]` 交給 temporary `hog_aux`，不要在 RGB stem 前 concat。
2. [`NativeTaskLossRouter.loss_for()`](<../../../yolo_combine/src/yolo_combine/joint_loss.py>) 在 final augmented/preprocessed image 已到 device、native criterion 計算後，把 batch-summed auxiliary raw loss加進 `raw_total`。
3. [`build_joint_optimizer()`](<../../../yolo_combine/src/yolo_combine/stage_policy.py>) 目前早於 router 建立；因此 `hog_aux` 必須在 optimizer build 前完成註冊，並新增明確 semantic role，不能把參數偷偷塞進 detect_head/pose_head。
4. F0/F1/F2 都要註冊同一個 `hog_aux` 與 optimizer group；F0只關閉 forward/loss。full-resume snapshot 要保存 auxiliary weights、optimizer state、`mu` schedule state與 target config，才能 exact-resume。
5. 現有 inference saver拿同一 model 存 state；必須新增顯式 materialization/strip gate，否則 temporary head 會殘留在 inference checkpoint。

不建議用全域 forward hook + `last_feature` mutable cache，因為 AMP replay、雙 task sequential forward、exception retry與 future concurrency 容易拿到 stale feature。較安全的是該次 forward 明確回傳 training-only auxiliary output，再由同次 router 消費。

### 7.4 stage 位置與 `mu(t)`

```text
J0：mu=0；只有 Pose head 可訓練，shared P3 不需 companion
J1 warmup：mu=0
J1 warmup 後：由 0 線性升到 mu0，再保持
J2 前 8 epochs：mu 由 mu0 線性降到 0
J2 其餘／J3：mu=0
```

理由：

- J1 首次打開 neck，HOG 能直接約束 P3 representation；
- J2 打開 late backbone 與 MASF，保留短暫過渡即可；
- J3 是低 LR 任務 refinement，應讓 native task loss決定終點；
- aux 在 J3 前歸零，才能驗證移除後 inference graph不受依賴。

`mu0` 不做大 grid。先用固定 train-only probe 量：

\[
r=\operatorname{median}
\frac{\lVert\mu_0g_{hog}\rVert_2}
{\lVert g_{native}\rVert_2+\epsilon}
\]

把第一版鎖在約 `0.05–0.15` 的輔助梯度比例；這只是安全起點。probe 的 batch ids、seed與計算規則須先落盤，不能看 formal validation 再改。

### 7.5 成本與風險

單一 `Conv1×1(256→9)` 只有：

\[
256\times9+9=2,313
\]

個 training-only parameters；P3 auxiliary output 每張有 `9×80×80=57,600` scalars。正式 inference 若 strip 正確，增加的 params/FLOPs/latency應為 0；training 仍有 HOG target、side-head activation與 backward 成本，必須實測 wall time、peak VRAM與 OOM，不得稱「完全零成本」。

主要風險與對應診斷：

| 風險 | 最小診斷／停止 |
|---|---|
| 小球／細球棒在 P3 HOG target 塌成低能量 | 先量每類 mask內有效 HOG cells；ball/bat 有效率低於事前門檻就停止 P3-HOG，不臨時改 P2 |
| side head只學局部亮度，不是方向 | 必做 matched LUMA9 control |
| HOG gradient壓過 native task | 記 `r`、clip率、overflow與 module update ratio；超界停止 |
| HOG 和 Detect/Pose gradient衝突 | 記 `cos(g_hog,g_detect)`、`cos(g_hog,g_pose)`；首輪不加 projection |
| target與影像 augmentation錯位 | HOG 必須由 router看到的最終 transformed image計算；用 synthetic flip/resize alignment test |
| training checkpoint可 resume、inference卻夾帶 aux | full-resume/inference schema分別做 round-trip與 exact key diff |
| seed0偶然增益 | seed0只作 go/no-go；通過後補 paired seeds |

### 7.6 為何 HOG 先於 Canny/WST

- HOG 只有 9 bins，與 P3 cell自然對齊，且有人形形狀與 prediction-target 的一手先例。
- Canny 含 smoothing、threshold與 hysteresis；hard sparse target 對極小物體敏感，又多兩個 threshold超參數。
- WST 官方配置多 81 channels、訓練計算與依賴較重；指定論文 YOLOX 表中也最高 FLOPs。
- 第一輪同時跑三種 filter會把小型、必要實驗擴成無界搜尋。HOG失敗就先結束這個假說，不以 Canny/WST 連續救結果。

---

## 8. 最小必要消融：只跑三個正式 arm

### 8.1 訓練前工程 probe，不作 AP 宣稱

先以少量固定 macro steps 驗證：

1. `mu=0` 時 native loss、gradients、optimizer update與 F0 在 tolerance內一致；
2. HOG target shape固定為 `B×9×80×80`，finite、local normalization與 flip/resize alignment正確；
3. Detect/pose 最後不完整 microbatch 的 auxiliary batch-sum語義正確；
4. AMP overflow replay不會重用 stale feature／target；
5. J0/J3 `hog_aux` 沒有更新，J1/J2 ownership符合 manifest；
6. full-resume能逐值續訓，inference checkpoint完全不含 aux keys；
7. Float/Bit-True正式 outputs、input schema、parameter count與 baseline相同；
8. training wall time／VRAM在預先同意預算內。

任一不變量失敗，先修工程；不能進正式 AP 比較。

### 8.2 三臂矩陣

所有 arm 從同一個 dormant-aux J0 stage-complete full-resume SHA、同一 auxiliary 初始化、資料順序、seed、optimizer state、macro exposure與 code revision開始：

| Arm | Temporary head | Target | 用途 |
|---|---|---|---|
| F0-MATCH | 已註冊但不執行 forward/loss | 無，`mu=0` | 功能上等同目前 J1→J3 的 matched baseline |
| F1-LUMA9 | 同一 `256→9` head | 9-bin soft-quantized local low-pass luminance | 排除 generic deep supervision／亮度重建 |
| F2-HOG9 | 同一 `256→9` head | 9-bin normalized HOG orientation | 測 gradient-orientation prior |

F1 與 F2 使用相同 object mask、loss family、`mu(t)`、optimizer role與 stage window。只有 F2 同時勝 F0、F1，才可把收益歸因於 HOG orientation；F2≈F1 只能說 companion supervision可能有用。

若算力只允許兩臂，可先跑 F0/F2，但結論只能寫「HOG auxiliary recipe有無觀測增益」，不能宣稱方向先驗是原因。為了研究上的可歸因性，三臂才是最小完整集合。

### 8.3 判定 gate

現行 `maximum_map_drop=0.08` 是部署可接受上限，對「優化是否有效」太寬。新 arm 使用相對 matched F0 的 tighter、事前鎖定 gate：

- joint score 至少 `+0.001`；
- 八項正式指標任一項不得低於 F0 超過 `0.001`；
- COCO person 或 ball pose 至少一項改善 `0.002`，否則沒有證據處理到現有主要缺口；
- F2 必須優於 F1，否則不能宣稱 HOG-specific；
- Bit-True/Float差異、finite、checkpoint、data digest、inference equivalence全部通過。

這些是工程上的 practical thresholds，不是由現有單 seed估出的統計顯著性。正式流程：

```text
seed0：F0/F1/F2
  ├─ F2 未過 tight gate → rejected，停止
  └─ F2 通過
       ├─ 最小產品判定：F0/F2 補 paired seeds 1/2
       └─ 若要主張 HOG-specific／論文創新：F0/F1/F2 全部補 seeds 1/2
```

三 seeds 前只能標 `provisional`。若 F2只在 seed0贏、或任一 protected metric反向，就不調更多 bins、cell size、位置、filter或 `mu` 來追 validation。

### 8.4 不要把 LR、bridge、conflict projection 疊進來

首輪唯一 treatment 是 target：

```text
固定：graph、dataset、head、optimizer、role LR、stage、macro、augmentation、
      BinaryQK、MASF、task weights、selector

變動：none vs LUMA9 vs HOG9 companion
```

LR diagnostic只在工程異常時觸發且不產生候選 checkpoint；semantic curriculum、Q0 teacher KD、gradient projection都不與 F2 同跑。

---

## 9. 資料血緣與 selection leakage

1. **BBAT5 immutable。** 只使用 canonical `pose.yaml`；不建立新 split、抽樣版、label版或影像版。
2. **Filter target不物化成新 dataset。** on-the-fly 記錄 HOG cell、bins、normalization、mask、epsilon、程式 digest；若為效能做 cache，cache只可由 image hash+transform/config hash重建，不能成為資料版本。
3. **先 augmentation、後 target。** HOG由實際進 model 的 tensor計算，不能從未翻轉／未縮放原圖產生。
4. **GT只存在 training loss。** object mask不得進 validation/inference prediction，也不能用預測框替代後再暗中調 threshold。
5. **不以 formal val無界調參。** cell=8、bins=9、P3、三臂與 `mu`規則先鎖定；失敗就停止本方向。
6. **COCO labels保持完整。** 現行 Detect 是 80 類；HOG mask使用所有 annotation。person-only若未來成立，另以該方向的 frozen manifest重開實驗。
7. **checkpoint角色清楚。** full-resume可續訓，inference不可；Float是訓練 source，Bit-True負責正式 gate。
8. **相同 optimizer steps與 exposure。** auxiliary多耗時不能藉由少跑 steps、改 batch或少看 BBAT來補成本。

---

## 10. 最終排序與答案

### 10.1 目前最合理的順序

| 排名 | 動作 | 決策 |
|---:|---|---|
| 1 | F0/F1/F2：P3 training-only box-aware HOG companion | 本篇首測；最直接保留論文 filter思想，又不改部署圖 |
| 2 | HOG arm 的條件式 aux-LR diagnostic | 只有 update/overflow異常才做；不是 accuracy arm |
| 3 | COCO semantic replay curriculum | 延後；同域、收益機理弱，需 B0/B1/B2 才能判斷 |
| 4 | AdamW vs MuSGD／新 scheduler | 另案；會改第二個主要因素 |
| 5 | 直接 gray+HOG/Canny/WST input concat | 不做；改 stem、損 RGB、推論重算 |
| 6 | DOTA/SAR 類中介資料 | 不做；domain/head/data lineage不匹配 |

### 10.2 一句話結論

指定論文第 3.1 節對目前 YOLO26M 最有價值的不是照搬 `5e-5`、DOTA 或十通道 gray+HOG input，而是把 HOG 改造成 **只在 J1／J2 early 約束現有 post-MASF P3、最後可完全移除的 object-aware gradient-orientation target**；先用 LUMA9 負控制證明它不是一般 deep supervision，再用現行八項 Bit-True metrics與 paired seeds決定是否保留。

## 11. 一手來源與本地證據索引

外部一手來源：

- [MSFA／SARDet-100K，NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/hash/e7eb8128eb26eafbe901348df1dbacdc-Abstract-Conference.html)
- [MSFA 官方 repository](https://github.com/zcablii/SARDet_100K)
- [MSFA 官方 filter/input 程式](https://github.com/zcablii/SARDet_100K/blob/main/MSFA/msfa/models/backbones/MSFA.py)
- [Cyclical Learning Rates／LR range test](https://arxiv.org/abs/1506.01186)
- [AdamW](https://arxiv.org/abs/1711.05101)
- [DOTA](https://arxiv.org/abs/1711.10398)
- [HOG](https://www.cs.princeton.edu/courses/archive/fall13/cos429/papers/Dalal05.pdf)
- [MaskFeat](https://openaccess.thecvf.com/content/CVPR2022/html/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.html)

本地主要證據：

- 指定論文 PDF（本機／歷史參照：`../references/papers/sar-yolox-multiprecision.pdf`；未隨本次報告發布）
- [Full35 factory report](<../../../yolo_combine/final/full35/outputs/training/factory-report.json>)
- [Full35 resolved config](<../../../yolo_combine/final/full35/outputs/training/resolved-config.json>)
- [Full35 最終分析](<../../../yolo_combine/final/full35/analysis/FINAL_ANALYSIS.md>)
- [Full35 machine-readable summary](<../../../yolo_combine/final/full35/analysis/SUMMARY.json>)
- [joint loss／macro engine](<../../../yolo_combine/src/yolo_combine/joint_loss.py>)
- [stage／optimizer policy](<../../../yolo_combine/src/yolo_combine/stage_policy.py>)
- [formal trainer](<../../../yolo_combine/src/yolo_combine/_formal_training_impl.py>)
- [dual-head P3/P4/P5 seam](<../../../yolo_combine/src/yolo_combine/fusion_model.py>)
- [P3 MASF graft](<../../../yolo_combine/final/full35/source_bundle/code/achitechure_1/model.py>)
- [BBAT5 registry](<../../../configs/datasets/bbat5-v1.yaml>)
- [BBAT5 immutable data contract](<../../../docs/agents/bbat5-datasets.md>)
- [既有 Attention LR 消融](<../../../yolo_attention_final/final/TRAINING.md>)
- [Q0 BinaryQK 計畫](<../../proposals/binaryqk-accuracy-recovery/README.md>)

## 12. 限制、困難與未解風險

- 本輪只研究與唯讀稽核，沒有產生新的 AP、latency或 VRAM結果。
- 指定論文的 mAP50不能直接和本地 mAP50-95比較。
- 本地 Full35目前只有 seed0；所有 practical gate仍需 paired seeds。
- HOG對 P3 tiny ball/bat 的有效 target occupancy尚未實測，是首個工程 go/no-go。
- `mu0`、LUMA9 encoding與 aux serialization仍是提案，需先做數值/恢復測試。
- 未在 scoped 正式 Full35 artifacts 找到既有 HOG/WST/Canny matched ablation；不能宣稱本案已有正面證據。
- 困難：PDF 的頁碼與檔案頁索引不同，已按章節標題完整定位第 3.1.1–3.1.4；其餘無。
