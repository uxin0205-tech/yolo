# 論文第 3.1 節轉用研究：以「衝突安全訓練」取代照抄 MSFA

日期：2026-09-04

狀態：`research / proposed`

範圍：只做論文、官方程式與本地既有證據稽核；未啟動訓練、未修改模型／資料／checkpoint。
建議方向代號：`OPT-TRAIN-CONFLICT-SAFE`（名稱可在建立方向資料夾時再定稿）。

> **2026-09-04 定位更正。** 本報告保留為「由本地 Detect/Pose 梯度證據衍生的獨立方向」，不再作為
> 指定論文第3.1節對 YOLO26M 的直接適配答案。逐項重讀 3.1.1–3.1.4 後，最直接且可驗證的轉法是
> [P3 training-only、box-aware HOG companion](<2026-09-04-paper31-to-yolo26m-training-adaptation.md>)：
> filter只產生訓練標靶，不進推論輸入。兩個方向首輪不疊加，以免無法歸因。

## 結論先行

論文第 3.1 節真正值得借用的不是「COCO → DOTA → SARDet」或「把 HOG 永久接到輸入」本身，而是三個設計原則：先固定部署預算、用短程診斷縮小訓練搜尋、再用一個中間橋接過程降低突然轉移造成的 feature shock。指定論文的完整依據見使用者提供論文，第 3.1.1–3.1.4 節（本機／歷史參照：`../../Design%20and%20Implementation%20of%20a%20Multi-Precision%20Deep%20Learning%20Accelerator%20for%20YOLOX-Based%20Object%20Detection%20in%20Synthetic%20Aperture%20Radar%20Images.pdf`；未隨本次報告發布）。

本專案不應直接照抄，原因是：

1. 論文處理的是 RGB／遙測 RGB／SAR 的成像域差異；本專案正式資料是 COCO80 RGB Detect 與 BBAT5 RGB ball／bat Pose／Detect，不存在相同的 SAR modality gap。
2. 本地正式 Full35 已經有 J0→J3 的 head adaptation、分層 LR、逐步解凍、early stop 與 rollback；再把「multi-stage」換名字重跑，沒有新增可辨識的機制。
3. 本地真正已量到的訓練問題是雙任務負遷移：durable logs 在 J1／J2／J3 分別有 `30/93`、`38/116`、`21/51` 個 shared-gradient cosine 為負；即使 J2、J3 的 Pose／Detect norm ratio 中位數已降到 `0.7506`、`0.6656`，方向衝突仍未消失。最終 J3 相對 standalone 的 COCO person 與 ball pose 分別仍低 `0.005804`、`0.016354`。這比猜測 DOTA 類中介資料或只重調 LR 更直接。
4. 原始 MSFA 官方程式會把影像先灰階化，再串接固定 filter channels，並修改 backbone `in_channels`；這會成為推論資料路徑的一部分，不符合本專案希望保留 RGB 與既有硬體圖的條件。[MSFA 官方實作](https://github.com/zcablii/SARDet_100K/blob/main/MSFA/msfa/models/backbones/MSFA.py)

因此第一順位改為 **Detect-priority Asymmetric Conflict Projection（Detect 優先的非對稱衝突投影）**：只在 active shared parameters 上、且 `g_detect · g_pose < 0` 時，移除 Pose gradient 中反向於 Detect gradient 的分量；Detect gradient 與兩個 task-specific heads 都保留原樣。第一輪只需要 matched control 與投影 arm 兩個 jobs，沒有 teacher forward，也不改推論圖。

這裡必須誠實命名：目前 durable logs 量到的是完整 COCO Detect loss 對 Pose loss 的 gradient，不是 class-0
person-only gradient；所以證據應寫 `g_detect`，不能冒稱 `g_person`。person head會先在另一方向以H1/H2固定，
之後G0/G1兩臂共同使用同一winner並重新screen。即使當時只有一類，`g_detect`仍包含box、assignment與
classification，不應縮寫成單純的person classification gradient。

若投影已消除方向衝突、protected AP 卻仍未補回，第二順位才測 **Lineage-Preserving Anchor Bridge**：用既有 standalone Detect／Pose 模型作短暫 frozen-teacher feature anchor。濾波概念則保留為第三順位的 **training-only、box-aware structure target**，不堆進首臂。

濾波器若要利用，第三順位建議改成 **training-only、box-aware structure target**：用 Sobel／HOG 產生固定 target，由暫時性的 P2/P3 side head 預測；正式推論時移除 side head。它不改 RGB 輸入、不增加部署 FLOPs，也比直接串 HOG/Canny/WST 更容易做負控制。這仍是假說，不得預先寫成已有效。

優先順序如下：

| 排名 | 方向 | 首輪是否執行 | 理由 |
|---:|---|---|---|
| 1 | Matched control vs. Detect-priority asymmetric projection | 是；2 jobs | 直接處理 durable logs 已量到的方向衝突；不加 teacher forward |
| 2 | Dual-teacher anchor bridge | 條件式；投影結束後另案 | 若衝突已解但遺忘仍在，才測函數錨定 |
| 3 | Box-aware Sobel/HOG companion loss | 條件式；獨立另案 | 借用 filter 概念，但不改 RGB／推論圖 |
| 4 | 局部 scope LR range test | 只有 trainable scope 或 optimizer 改變時 | 本地已有 LR sweep；不需再做五組完整短訓 |
| 5 | GradNorm | 目前不先做 | J2/J3 norm ratio 已小於 1，衝突率卻仍約 33–41%；不是單純量級問題 |
| 6 | DOTA／SAR 類外部 domain bridge、永久 filter input | 不建議 | 任務域不匹配、資料血緣與推論成本變複雜 |

**可獨立歸檔的首輪邊界。** 已建立
[`optimizations/training-conflict-safe/`](<../../optimizations/training-conflict-safe/README.md>)，其中只收
`G0-MATCH`、`G1-APC-DETECT`的frozen spec、shared-parameter manifest、數值probes、run manifests與報告；
dual-teacher anchor、Sobel/HOG companion與person head不得放進首輪處理arm。如此資料夾只有一個可推翻的
假說：「移除Pose對Detect的當次反向shared-gradient分量，是否能補回protected AP？」本輪只建立研究與計畫，
沒有修改production code或啟動實驗。

## 一、證據標籤與術語邊界

下文使用四種標籤：

- **論文事實**：指定 PDF 第 3.1 節直接報告的設定或數字。
- **官方一手證據**：原作者論文、官方程式或正式框架文件。
- **本地事實**：目前工作區的 spec、metrics、registry 或 checkpoint lineage。
- **本專案提案**：尚未訓練驗證的設計；只能稱 `proposed`。

此外必須區分兩個容易混淆的名稱：

- `MSFA`：指定論文的 Multi-Stage with Filter Augmentation 訓練／輸入框架。
- `MASF`：本地放在 P3 的架構模組。兩者不是同一機制，不能把 P3 MASF 的結果當成 MSFA 訓練證據。

## 二、指定論文第 3.1 節逐段稽核

### 2.1 第 3.1.1 節：先用部署預算選模型

**論文事實。** 論文假設 UAV 約有 `1 TOPS` 可分配給 30 FPS 偵測，換算約 `33G operations/frame`，比較 F-RCNN、Deformable DETR、YOLOv5 與 YOLOX。其表 3.1 報告 YOLOX-S 為 `82.3% mAP@50`、`8.94M Params`、`20.82G FLOPs`，因而選作後續 baseline。

**可轉用原則。** 每個訓練創新都必須先聲明部署圖是否改變，並同時登錄 Params、FLOPs、同卡 latency、峰值記憶體與 Bit-True 指標。若方法只存在於 training，正式 export 應與 baseline 保持相同 state schema／輸入 channels，並做逐 tensor 或逐輸出的 equivalence gate。

**不能照抄之處。** `FLOPs`、`operations` 與實際 30 FPS latency 不是同一量；論文的預算推算不能取代本機 target-kernel／FPGA 量測。本訓練方向也不應順手換 YOLOX、head 或 P3 MASF，否則無法歸因。

### 2.2 第 3.1.2 節：短 LR 篩選

**論文事實。** batch 固定 16；五個初始 LR `{1e-4, 5e-5, 2.5e-5, 1.25e-5, 6.25e-6}` 各跑 9 epochs，以 early loss 與 validation mAP@50 選 `5e-5`；optimizer 為 AdamW、`betas=(0.9, 0.999)`，再依曲線約在 30 epochs 飽和而定總長。

**官方一手證據。** Leslie Smith 的 LR range test 是在短時間內逐步提高 LR，找出合理上下界，不是把每個 LR 都完整訓練一次；原論文也只把它當界線診斷。[Cyclical Learning Rates／LR range test](https://arxiv.org/abs/1506.01186) Hyperband 則提供把少量資源先分配給多個候選、逐步淘汰的多保真搜尋框架，但它不能保證非常短的 detection curve 排名就是最終排名。[Hyperband 原始論文](https://www.jmlr.org/papers/v18/16-558.html)

**本地事實。** 本地 Attention block 已跑 `5e-6 / 1e-5 / 2e-5`，最低的 x1 反而最好；較高 LR 逐步變差。Full35 舊 J1 又曾在 warmup 附近由 COCO delta `-0.0813` 惡化到 `-0.1855`，降 LR 後才恢復。詳見[正式 Attention 訓練說明](<../../../yolo_attention_final/final/TRAINING.md>)與[Full35 最終分析](<../../../yolo_combine/reports/full35/FINAL_ANALYSIS.md>)。

**本專案提案。** 不重做論文式 `5×9` grid：

- 若 optimizer、batch、trainable scope 都沿用正式 J recipe，直接沿用現行 LR。
- 只有 scope 改變時，從同一 parent 的犧牲性 clone 做一次 batch-wise LR range diagnostic；結束後丟棄 clone，重新載入 parent 並重建 optimizer／scheduler。
- range test 只看平滑 loss、gradient norm、`update_ratio` 與 finite safety，不用 formal val 選最後 winner。
- 分組相對更新量可記為：

\[
r_l=\frac{\eta_l\lVert g_l\rVert_2}
          {\lVert\theta_l\rVert_2+\epsilon}.
\]

若 early backbone 的 `r_l` 明顯高於既有安全 run，才降低該 group LR 或 freeze；不能先假定「前 N 層一定該凍結」。

### 2.3 第 3.1.3 節：COCO → DOTA → SARDet 多階段轉移

**論文事實。** 論文把 DOTA 遙測 RGB 當 COCO 自然影像與 SARDet 雷達影像之間的 domain bridge；表 3.2 報告 conventional `82.3%`、multi-stage `83.4% mAP@50`，差 `+1.1`。

**官方一手證據。** 原始 SARDet-100K 工作指出其核心問題是 RGB pretraining 與 SAR finetuning 之間的 data-domain／model-structure gap，並提出 MSFA；該證據屬 SAR 任務，不等於一般 RGB 棒球任務也會受益。[NeurIPS 2024 SARDet-100K／MSFA 論文](https://papers.neurips.cc/paper_files/paper/2024/hash/e7eb8128eb26eafbe901348df1dbacdc-Abstract-Conference.html)

**本地事實。** 現行 joint model 本來就有 J0 Pose-head adaptation、J1 neck＋heads、J2 late-backbone、J3 full low-LR refinement；模型轉移 seam 已存在。[正式 joint-training spec](<../../../yolo_combine/docs/design/2026-08-24-yolo26-joint-training-spec.md>)

**因果限制。** 指定論文表 3.2 沒有在同表證明額外 DOTA 樣本、額外 optimizer steps 與「domain bridge 語意」各自貢獻多少；`+1.1` 不可直接移植成本專案的預期收益。對本專案加入 DOTA 或 SAR 甚至會創造無關 domain shift。

**本專案提案。** 把「資料域 bridge」改成「功能 bridge」：兩個 standalone teachers 錨定 shared trunk 在各自任務上的既有函數，讓 joint student 漸進離開 parent，而不是突然把兩個任務壓進同一組 shared weights。Learning without Forgetting 提供用 distillation 約束既有函數、降低遺忘的原始先例；它只支持這個機理方向，不保證本案 AP 必升。[Learning without Forgetting](https://arxiv.org/abs/1606.09282)

### 2.4 第 3.1.4 節：WST／HOG／Canny filter augmentation

**論文事實。** 指定論文表 3.3 報告：WST `83.4% / 35.38G`、HOG `83.9% / 22.11G`、Canny `83.7% / 21.63G`，三者 Params 都是 `8.94M`；該 YOLOX 實驗選 HOG。表中是 mAP@50，不能和本地 mAP50-95 直接比較。

**官方一手證據。** 原始 MSFA 官方實作不是一般意義的隨機 augmentation：啟用 filter 時先 `x.mean(1, keepdim=True)`，再在 `torch.no_grad()` 下把灰階與 filter output 串接；HOG、Canny、WST 分別增加 9、6、81 channels，最後改寫 backbone `in_channels`。因此 filter 是固定、推論期也要執行的 input representation。[官方 `MSFA.py`](https://github.com/zcablii/SARDet_100K/blob/main/MSFA/msfa/models/backbones/MSFA.py) 官方主論文／repository 的主要 MSFA 結果以 WST 為代表，指定論文的 YOLOX 消融卻選 HOG，正好說明 filter winner 依資料、架構與 recipe 而變，不能把「HOG 最好」視為普遍定律。[官方結果與 configs](https://github.com/zcablii/SARDet_100K)

**本專案提案。** 保留 3-channel RGB raw path；filter 只作訓練 target 或 teacher view，絕不永久串到輸入。MaskFeat 證明「預測 HOG target」可作 representation-learning objective，且其消融強調 HOG local contrast normalization；但該結果來自 masked image/video pretraining，不是 YOLO26 detection，因此本案仍必須做 matched ablation。[MaskFeat 原始論文](https://openaccess.thecvf.com/content/CVPR2022/html/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.html)

AugMix 的一手設計則是混合多個 augmentation views 並加 consistency loss；它提供「augmentation 不一定要成為推論輸入」的先例，但原始證據主要是 image classification robustness，不能直接當本案 detection 精度證明。[AugMix 論文](https://openreview.net/forum?id=S1gmrxHFvB)、[官方程式](https://github.com/google-research/augmix)

## 三、本地正式訓練與資料血緣稽核

### 3.1 目前已經具備、不可冒充新創新的部分

現行 Full35 joint recipe 已包含：

- `J0`：只讓 Pose head 適應 Detect trunk。
- `J1`：shared neck＋兩 heads。
- `J2`：late backbone＋neck＋MASF＋兩 heads。
- `J3`：full low-LR refinement，最後才開 attention 可微部分。
- 分層 LR、每 stage 新 optimizer、BN policy、Bit-True validation、early stop、rollback 與八項 hard gate。

因此「head→neck→backbone gradual unfreezing」是本地 baseline，不是新方向的處理組。Full35 J3 的正式結果與訓練設定見[最終分析](<../../../yolo_combine/reports/full35/FINAL_ANALYSIS.md>)。

### 3.2 新方向要解的已知症狀

正式 Full35 durable gradient logs 的 stage 彙整如下；`norm ratio` 是 Pose／Detect，共享梯度以當時 active shared scope 計：

| Stage | 記錄數 `n` | cosine mean | `cosine < 0` | 負衝突率 | norm ratio median |
|---|---:|---:|---:|---:|---:|
| J1 | 93 | `0.042682` | `30/93` | `32.258%` | `3.6785` |
| J2 | 116 | `0.018512` | `38/116` | `32.759%` | `0.7506` |
| J3 | 51 | `0.026841` | `21/51` | `41.176%` | `0.6656` |

另有舊版 early probe 顯示 Pose／Detect norm ratio 平均約 `19.69×`、`7/24` 次 cosine 為負；它可作歷史交叉檢查，但 durable stage logs 才是本提案的主要診斷。[Full35 durable training logs](<../../../yolo_combine/final/full35/outputs/training/logs>)

| 最終量測 | 數值 | 對新方向的含義 |
|---|---:|---|
| J3 person internal AP vs standalone | `-0.005804` | joint trunk 對 person 有可量測遺忘，作 Detect-side tight gate |
| J3 ball pose vs standalone | `-0.016354` | 最大已知 protected-task gap，不能為保 Detect 而放棄 |
| J3 joint score vs J2 | `+0.0011365` | staged refinement 有效，但不是全指標同升 |

關鍵判讀不是「平均 cosine 仍為正」，而是每個 stage 都有約三分之一以上的更新事件方向衝突，J3 更達 `41.176%`。同時 J2/J3 的 norm ratio 中位數已小於 1，負衝突率卻沒有下降，表示後段的主要可測症狀是**方向**，不能只歸因於 Pose gradient 太大。因此：

- 全域降低 LR 只會縮小合成 gradient，通常不會改變兩 task 的夾角；它不能直接消除負 dot product。
- GradNorm 主要調量級／相對訓練速率，對這組 J2/J3 證據不是第一個最小 intervention。
- 非對稱 conflict projection 直接在負 dot product 發生時才介入，與現有量測一一對應，故應排在 scheduler、GradNorm 與 teacher anchor 前面。

這些數字仍來自單一正式 run，能定位與設計實驗，不能宣稱統計顯著，也不能證明 gradient conflict 是 AP 下降的唯一根因。現有 mean／count 也沒有給出每個負事件的幅度，所以首臂必須新增 `correction_ratio`，用來判斷究竟是「高頻但很淺」或「高頻且有實質修正量」。

### 3.3 正式資料入口

| 任務 | 正式入口 | 目前契約 |
|---|---|---|
| COCO80 Detect | `/home/uxin/yolo/coco2017.yaml` | 118,287 train／5,000 val；80 類，person 是 class 0 |
| BBAT5 Pose | `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml` | 5,964 formal train／683 formal val；ball/bat、2 keypoints |
| BBAT5 Detect | `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml` | 和 Pose 共用相同影像與 assignment，只換 label view |
| BBAT5 search | `pose-search.yaml`／`detect-search.yaml` | 只在 formal train 內的 5,364／600 grouped split |

registry 規定 6,647 images、2,578 source groups，formal／search val 與 COCO train overlap 都是 0；不得新增 split、抽樣、改 label 或把 runtime view 當新資料版本。[BBAT5 registry](<../../../configs/datasets/bbat5-v1.yaml>)、[資料契約](<../../../docs/agents/bbat5-datasets.md>)

本輪唯讀重新計算的 YAML SHA-256：

```text
COCO2017                 1ed971749ec00b84db8ffe850b05e34e0acfd63a9654a9224ac8d80f5e068068
BBAT5 registry           208e0f9e5bfded11f5739e9c598d98b4e5414699c64b9faf6f9625680988754a
BBAT5 pose formal        dece3f4665fad86c2b0b3e3941b1cde9668cbfc6db1208c0e3d10638b3f5b888
BBAT5 detect formal      222f61f7800be52086218bc4e6880f23917abc33382714f8a737013b75aa6aa5
BBAT5 pose search        a9069992cf21bff802466fa8adb46cba9bd55585f5adc8f3c183d33c8ede0b5d
BBAT5 detect search      7b9fbeff82eed0c72f1db2aea003909c8e517c7252574c12ddebfeff69c35016
```

上述統計來自COCO80歷史training，只能作開案證據。正式執行先由獨立的
[person-only head方向](<../../optimizations/coco-person-specialized-head/README.md>)固定H1或H2，再讓G0/G1共同使用
同一head與同一person Runtime Dataset View。person specialization不是G1 treatment；只要兩臂的上游winner完全
相同，仍可把G1−G0歸因為projection。若使用者保留COCO80，也可以同法執行，但報告語意是保護完整Detect。

## 四、主提案：Detect-priority Asymmetric Conflict Projection

### 4.1 精確定義：保留 Detect，只修正衝突的 Pose 分量

令 `theta_s` 是該 stage 已解凍、而且 Detect 與 Pose 都實際使用的 shared parameters。先讓 `L_d`、`L_p` 包含現行 recipe 已有的 task normalization 與 task weights，再分別取得：

\[
g_d=\nabla_{\theta_s}\widetilde L_d,
\qquad
g_p=\nabla_{\theta_s}\widetilde L_p,
\qquad
d=g_p^Tg_d.
\]

首測只在 `d<0` 時修改 secondary Pose gradient：

\[
g_p^{safe}=
\begin{cases}
g_p-\dfrac{g_p^Tg_d}{\lVert g_d\rVert_2^2+\epsilon}g_d,
& g_p^Tg_d<0,\\[6pt]
g_p,& g_p^Tg_d\ge 0.
\end{cases}
\]

shared update 使用：

\[
g_{shared}=g_d+g_p^{safe}.
\]

Detect head 仍只接原生 `g_d`，Pose head 仍只接原生 `g_p`；projection 不可套到 task-specific heads，也不可套到只被單一 task 使用的參數。這不是 PCGrad 原論文的對稱／隨機順序版本，而是依本案保護需求設計的**單向變體**；PCGrad 只提供「衝突 gradient 可投影」的一手機理先例。[PCGrad 原始 NeurIPS 論文](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)

### 4.2 為何它比先改 LR／scheduler 更直接

當 `d<0` 時：

\[
g_d^Tg_p^{safe}
=d\frac{\epsilon}{\lVert g_d\rVert_2^2+\epsilon}
\approx 0.
\]

在純 SGD 的一階局部近似下，`Delta theta=-eta(g_d+g_p^safe)`，所以：

\[
\Delta L_d
\approx g_d^T\Delta\theta
=-\eta\left(\lVert g_d\rVert_2^2+g_d^Tg_p^{safe}\right)
\approx-\eta\lVert g_d\rVert_2^2.
\]

也就是只有 Pose 中「會在一階近似下抵銷 Detect」的投影分量被移除；其與 Detect 正交的學習分量仍保留。相反地，全域 LR 只把 `g_d+g_p` 同比例縮小，並未使負 `d` 變成非負。J2/J3 在 Pose norm 已不占優勢時仍有 `32.759%`／`41.176%` 衝突事件，因此先做方向修正比再猜 scheduler 或只做 norm balancing 更可歸因。

但上式**不是 AP 保證**，也不是實際 AdamW parameter step 的嚴格保證：momentum、adaptive preconditioning、weight decay、gradient clipping、非線性曲率和不同 task batch 都會改變真實更新。它只保證投影後的 raw shared Pose gradient，在數值 tolerance 內不再直接反向於當次 raw Detect gradient。因此 ball pose 仍需 tight gate，不能因為數學上保了 Detect 方向就假定兩任務都會上升。

### 4.3 為何首臂是 `g_detect`，不是冒稱 `g_person`

目前記錄的 dot product 是完整 COCO Detect criterion 對 BBAT Pose criterion；COCO batch 也含 80 類。現有證據沒有把 box、DFL／localization、object assignment 與 class-0 classification 全部分解成一個已驗證的 `L_person`。因此：

```text
已量到且首輪可驗證：g_detect ↔ g_pose
尚未定義／不可冒稱：g_person ↔ g_pose
```

不可在G1 arm內臨時把COCO80改成person-only；那會同時改supervision、normalization、positive assignment與資料
語意。正確順序是先在T0方向固定H1/H2，之後G0與G1從相同新parent重跑。此時仍稱`g_detect`，以canonical
person AP `delta >= -0.001`作關鍵gate；只有未來真的分解出獨立、驗證過的person classification criterion，
才可研究嚴格定義的`g_person`。

### 4.4 插入 J0→J3 的位置與數值順序

```text
J0：Pose head only
    └─ 沒有兩任務 active shared update，完全維持現行

J1／J2／J3 每個 macro-step：
Detect microbatch ──> g_detect(shared) ─────────────────┐
                                                       ├─ dot/cosine
Pose microbatch   ──> g_pose(shared) ── if dot<0 投影 ─┘
                                   │
                                   └─ g_pose_safe
                                          │
              shared：g_detect + g_pose_safe ─> clip once ─> AdamW step
          Detect head：原生 g_detect ──────────┘
            Pose head：原生 g_pose ────────────┘
```

實作契約必須鎖定：

1. 從 config／parameter ownership 產生 deterministic shared-parameter manifest；每 stage 記錄名稱與 digest，不能把「目前有 gradient」誤當 ownership。
2. 先完成兩 task 的 macro accumulation與 AMP finite/unscale，再在 FP32 accumulation buffer 計算 dot／projection；套用既有 task weights一次且只一次。
3. `d>=0` 時 shared gradient、兩 heads、clip input 與 matched baseline 必須在預定 tolerance 內一致；`d<0` 時只容許 shared Pose 分量改變。
4. projection 後才把 task gradients合併，之後只 clip 一次、step 一次；overflow 時整個 macro-step 一起跳過，不可只留下其中一個 task。
5. 每 stage 記 `pre/post cosine`、`projection_rate`、`norm ratio`、`correction_ratio=||g_p^safe-g_p||/(||g_p||+epsilon)`、missing-gradient count、AMP overflow與 wall time。
6. `||g_d||` 小於事前數值門檻時 fail/skip 並記錄，不可讓除法放大；門檻與 `epsilon` 在 probe 前固定。

### 4.5 這個方法能與不能回答什麼

若投影 arm 相對 matched control 同時降低 post-projection negative rate、恢復 person/ball-pose gate，便支持「可觀測 shared-gradient conflict 是負遷移的可干預來源」。若 cosine 被修正而 AP 不變或 Pose 更差，只能說 raw direction conflict 不是充分原因；此時停止調投影強度，轉測下一節的功能錨定。若 projection 根本很少觸發，先查 control 的 scope／記錄定義是否與歷史 logs 一致，不可擴大投影範圍來追求漂亮數字。

## 五、條件式第二提案：Lineage-Preserving Anchor Bridge

這一臂不與 gradient projection 同跑。只有首臂已通過數值 probe、確實把衝突項壓到近零，但 protected AP 仍未恢復時，才有證據懷疑問題還包含較長時間尺度的 representation drift／forgetting，值得支付兩個 teacher forward 與 `lambda` 搜尋成本。

### 5.1 目前 objective 為什麼可能遺忘

令 shared trunk 參數為 `theta_s`，Detect／Pose task loss 分別為 `L_d`、`L_p`，現行 joint loss 的 shared gradient 為：

\[
g = w_d g_d + w_p g_p,
\qquad
g_q=\nabla_{\theta_s}L_q.
\]

只調 `w_d:w_p` 可以縮放 norm，卻無法在 `g_d^T g_p < 0` 時同時保證兩個 task 都前進。當其中一個 task 的 norm 又遠大於另一個 task，shared update 更容易離開弱 task 的 standalone optimum；這和本地早期 norm ratio／negative cosine 方向一致，但仍只是機理解釋，不等於唯一根因。

### 5.2 用兩個 standalone teachers 建立功能 bridge

保留既有 standalone Detect teacher `T_d` 與 standalone Pose teacher `T_p`，全部 frozen、eval、no-grad。對 task `q` 的輸入 `x_q`，在 shape 已確認相同的 shared P3/P4/P5 features 建立物件區域 mask `M_{l,q}`：

\[
L_{anchor}^{(q)}=
\sum_{l\in\{P3,P4,P5\}}
\frac{
  \left\|M_{l,q}\odot
  \left(\hat F_l^S(x_q)-\operatorname{sg}(\hat F_l^{T_q}(x_q))\right)
  \right\|_1
}{\sum M_{l,q}+\epsilon}.
\]

`hat F` 表示 channel-wise normalized feature；`sg` 是 stop-gradient。mask 由既有 GT box 投影到 feature grid，適度包含 box 周邊，不生成 pseudo label，也不更動資料。偵測蒸餾研究指出，vanilla 全圖 imitation 容易被大量背景主導，而 near-object／knowledge-dense locations 更適合 detector feature imitation；這支持 object mask，但不保證本案的 P3/P4/P5 權重。[Fine-Grained Feature Imitation 原始論文](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Distilling_Object_Detectors_With_Fine-Grained_Feature_Imitation_CVPR_2019_paper.html)

完整 objective：

\[
L(t)=\frac{w_dL_d+w_pL_p}{w_d+w_p}
     +\lambda_d(t)L_{anchor}^{(d)}
     +\lambda_p(t)L_{anchor}^{(p)}.
\]

`lambda_q(t)` 只在 shared layers 剛解凍的 bridge window 非零，然後依預先固定 schedule 衰減到 0：

\[
\lambda_q(t)=\lambda_{q,0}\max\left(0,1-\frac{t}{T_{bridge}}\right).
\]

如此 early update 不會立刻拋棄 standalone functions；後段 `lambda=0` 又允許 joint optimum 不被 teachers 永久鎖死。這是本專案的新組合提案，不應宣稱為全球首創；在投稿前仍需更廣泛的新穎性檢索。

### 5.3 插入現有 J0→J3 的位置

```text
immutable standalone Detect teacher ──┐
                                       ├─ object-masked feature anchor
immutable standalone Pose teacher ────┘
                                       ↓
J0：Pose head only（維持現行，無需 anchor）
  ↓
J1 early：neck + heads + anchor bridge，lambda 逐步下降
  ↓
J1 late：現行 task loss only
  ↓
J2 early：late backbone 解凍 + 短 anchor bridge
  ↓
J2 late／J3：lambda=0，維持現行 selector／Bit-True gate
```

teacher forward 應在每個 task microbatch 以 no-grad 順序執行並立即釋放 features，不把兩個 teacher graph 同時留在記憶體。若 wall time 過高，可預先固定「每 `k` 個 macro 才 anchor 一次」的 sparse cadence；`k` 必須是 config、不可依 validation 結果臨場改。

### 5.4 這個設計和論文的差別

| 指定論文 | 本專案提案 |
|---|---|
| DOTA 是實體中介資料集 | 不新增資料；standalone functions 是 bridge |
| filter channels 成為模型輸入 | raw RGB 輸入不變 |
| 單一 SAR detection task | 同時保護 COCO Detect 與 BBAT Pose |
| 以 mAP@50 選 filter | 用八項現行 metrics、person 與 ball pose tight gates |
| HOG/WST/Canny 影響推論 FLOPs | anchor 只增加 training cost；正式 graph 不變 |

## 六、第三順位：Box-aware structure companion loss

### 6.1 不把 HOG/Canny 接進輸入

先從 RGB luminance `Y(x)` 產生固定 Sobel magnitude：

\[
E(x)=\operatorname{norm}\left(
\sqrt{(K_x*Y(x))^2+(K_y*Y(x))^2+\epsilon}
\right).
\]

在 early/P2 或 P3 feature 掛暫時性 side head `h`，只在物件 soft mask `M` 內預測 downsampled edge target：

\[
L_{struct}=
\frac{\left\|M\odot
\left(h(F)-\operatorname{sg}(\operatorname{down}(E(x)))\right)
\right\|_1}{\sum M+\epsilon}.
\]

最終 objective 加 `mu(t)L_struct`，最後 20% training schedule 令 `mu(t)=0`；export 前刪除 side head，並要求 raw inference graph 與 matched baseline 的輸入 shape、正式參數集合及硬體路徑一致。

選 P2 或 P3 不能憑直覺：本地 ball 在 P3 只有約 2 cells，因此第一個工程 probe 應先比較 target downsample 後的 object-mask 非零率。若 P3 target 大多塌成 0，直接選 P2；這不是新增 P2 Detect head，推論時不保留。

### 6.2 為何先 Sobel、後 HOG／WST

- Sobel target 是連續、逐像素、低成本，適合先驗證「結構輔助 loss」這個機制。
- HOG 有方向 bins 與 pooling，可能對人形輪廓有利，但也可能讓極小 ball 在降採樣後消失；通過 Sobel 後再作單一替換。
- Canny 有 hard threshold，對參數與影像噪聲敏感；不作第一輪。
- WST 在指定論文 FLOPs 最高，官方實作又增加 81 channels；只有便宜方案已顯示結構 target 有效後才有理由研究。

### 6.3 必要負控制

若 structure arm 初步過 gate，正式主張「edge/HOG 特異性」前必須補一個相同 side head、相同 loss scale 的 luminance／low-pass target control：

```text
S0-LUMA：temporary head 預測 downsampled luminance
S1-EDGE：同 head 預測 Sobel/HOG target
```

只有 `S1 > S0` 才能把收益歸因給 structure target；若兩者相同，只能說額外 deep supervision 有用。AugMix 式雙 view consistency或 filter-privileged teacher 是更昂貴的後續，不列首輪。Generalized Distillation 說明 training-only privileged representation 可蒸餾進 test-time student，但它只是後續機理依據。[Generalized Distillation／Privileged Information](https://arxiv.org/abs/1511.03643)

## 七、為何首輪不再加入其他 optimizer／schedule 變因

- **不跑對稱 PCGrad。** 本案首測是 PCGrad-inspired 的單向投影；若同時讓兩邊互相投影，就失去「保留 Detect、只修 secondary Pose 衝突分量」的明確假說，也多了 task-order semantics。
- **不跑 GradNorm。** GradNorm 會依相對訓練速率動態調 task weights；它適合主要問題是 norm imbalance 時。[GradNorm 原始 ICML 論文](https://proceedings.mlr.press/v80/chen18a.html) 但本案 J2/J3 ratio median 已為 `0.7506/0.6656`，負衝突仍有 `32.759%/41.176%`，所以首輪先處理方向。
- **不重掃 LR／scheduler。** 全域 LR 不改 task angle；而本地已有多段低 LR、分層 LR 與既有 sweep。只有 projection 實作迫使 trainable scope、effective batch 或 optimizer 改變時，才需犧牲性 LR range probe，不能把它混成首輪第三因素。
- **不先加 teacher anchor 或 HOG。** 兩者各自改 objective 且增加計算，會讓首輪無法判斷收益是否來自 conflict removal。

## 八、最小實驗矩陣與執行順序

### 8.1 必跑矩陣：只有兩個 training jobs

兩臂必須從同一個 immutable parent、同一份 code revision、同一 seed 與同一 optimizer-step sequence 開始：

| Arm | 現行 J0→J3 | J1–J3 active shared scope | Teacher/filter/其他變因 | 用途 |
|---|---|---|---|---|
| `G0-MATCH` | 完全相同 | 原生 `g_d+g_p` | 無 | 排除重訓、driver 或 seed 差異 |
| `G1-APC-DETECT` | 完全相同 | 只在 `g_d·g_p<0` 投影 `g_p` | 無 | 測方向衝突修正的邊際效果 |

J0 沒有 active joint shared update，兩臂都原樣執行。兩臂使用同一 immutable parent、同一 code revision（只差 config feature flag）、seed、sampler manifests、batch／macro-step 順序、optimizer-step 數、LR／scheduler、stage selector與 validation gates。existing J3 只是歷史參考，不可取代 `G0-MATCH`。

### 8.2 訓練前工程 probe，不作 AP 宣稱

先固定少量 macro steps，只檢查：

1. 合成向量單元測試：負 dot 時 `g_d` 不變、post dot 在 tolerance 內約為 0；非負 dot 時 `g_p^safe==g_p`；另測 zero norm、missing grad、NaN/Inf。
2. shared manifest 的 names、numel、stage ownership固定；task heads 在正／負 dot case 都逐 tensor 比對原生 task gradients。
3. `projection_enabled=false` 時 loss、gradient、clip input、optimizer state與 parameter update 必須和既有 path matched 到預定 tolerance。
4. AMP unscale、task weight、macro accumulation、projection、single clip、single optimizer step次序符合 4.4；刻意觸發 overflow 時兩 task 一起 skip。
5. `pre/post cosine`、projection rate、correction ratio、norm ratio與 durable log schema 能完整重播；正 dot 分支不可被「順便正規化」。
6. checkpoint／export schema、正式 inference output與 latency不變；training wall time／VRAM 增量實測並登錄，超過事前預算就停止，不用降低 batch偷換契約。

### 8.3 seed-0 篩選後才補 seeds

工程 probe 通過後各跑一個 paired seed。`G1-APC-DETECT` 必須相對 `G0-MATCH` 同時符合：

- joint score `>= +0.001`；
- COCO person 與 ball pose 都不得低於 G0 超過 `0.001`；
- 八項既有 metrics 任一項不得低於 G0 超過 `0.001`；
- 至少 person 或 ball pose 其中一項改善 `>= +0.002`，否則沒有證據說投影解決了目標問題；
- 所有觸發事件的 post dot 都在事前數值 tolerance 內，且 Detect gradient／兩 task heads 通過不變量；
- Bit-True／Float 差、NaN/Inf、資料 digest與訓練成本全部過 gate。

任一 metric／不變量失敗即停止本首臂，不調 projection strength、不換成 per-layer projection，也不把 anchor／structure loss 混進來救結果。只有 seed 0 通過才補兩組 paired seeds，以 paired mean/std 套用相同 gate；三 seeds 前狀態只能是 `provisional`。

### 8.4 只有首臂無效後才開條件式候選

若 `G1` 的數值機制正確但 AP gate 失敗，先把它標為 `rejected`。若特別呈現「post conflict 已消失、person/ball-pose 遺忘仍在」，才另開 `A0-MATCH` vs `A1-ANCHOR`；anchor 首測也必須單獨相對 baseline，不與 `G1` 疊加。這樣才能檢驗長時間尺度的 function drift，而不是無界堆方法。

filter 子矩陣更後面，且不和 projection／anchor 混跑：

只有主方向已完成、且仍希望驗證 filter 概念時才跑：

| Arm | 暫時 side head | Target | 推論保留 | 用途 |
|---|---|---|---|---|
| `F0-CTRL` | 無 | 無 | 無 | matched baseline |
| `F1-STRUCT` | 有 | Sobel magnitude | 否 | structure hypothesis |

`F1` 初步過 gate 後，才補 `F-LUMA` 負控制；若 Sobel 確認有效，再只替換 target 為 HOG，不同時改 LR、位置或 mask。

## 九、停止條件與不可接受的解讀

立即停止條件：

- shared-parameter manifest／stage ownership 和 matched control 不一致；
- 非負 dot 分支不能重現 baseline gradient，或任一 task-specific head gradient 被改動；
- 負 dot 分支的 post dot 超出事前 tolerance、zero-norm／missing gradient 未被 fail closed；
- 非 finite loss／gradient／checkpoint／metrics；
- 任一 BBAT5 YAML、assignment、label hash 或 source-group overlap 不符合 registry；
- seed-0 tight metric gate 失敗；
- 需要用新的 split、DOTA/SAR 資料、調 validation 門檻或改 architecture 才能讓 arm 看起來有效；
- training-only 方法無法在 export 時完全移除，或 formal inference schema／latency 出現未登錄差異。
- 若後續進入 anchor：teacher／student shape 或 owner 不一致，或任一 teacher tensor、BN running stat／checkpoint digest 改變。

不得作以下解讀：

- 短 LR probe loss 低，不等於最終 AP 一定高。
- 單 seed `+0.001` 不等於穩定改良。
- joint score 上升不能抵銷 person／ball pose 退化。
- post cosine 接近零只證明 projection 實作生效，不等於 AP 改善或根因已證實。
- 完整 Detect gradient 受保護不能寫成「person gradient 已受保護」；person 只能由獨立 AP gate 評估。
- 平均 cosine 為正不能抹去 32–41% 的事件衝突；反之，事件衝突也不能單獨證明 AP 下降全由它造成。
- filter arm 上升不能在沒有 LUMA control 時宣稱 HOG/edge 特異性。
- 原論文 SAR 的 `+1.1 mAP@50` 不能當成本案預期收益。
- training-only 增加不等於「零成本」；只能說正式 inference cost 可保持不變，training wall time／VRAM 仍須量測。

## 十、資料血緣與模型選擇風險

1. **BBAT5 不可變。** 第一輪若做 joint Pose，只使用 `pose.yaml`；若另做 ball/bat Detect，使用 `detect.yaml`。search YAML 只能作 formal-train 內工程篩選，不可取代 formal validation。
2. **不建立中介資料版本。** filter target 可 on-the-fly 生成，必須記 filter kernel、normalization、mask、seed 與 code digest；不物化成另一套 images／labels。
3. **不把 val 當 optimizer controller。** projection 的 scope、方向、`epsilon` 與 zero-norm 門檻事前鎖定；若後續進入 anchor，`lambda(t)`、bridge window 與 sparse cadence也須預先鎖定，不能看 formal val逐 epoch手調。
4. **COCO／BBAT 任務保持分離。** 不把兩種 labels 合成同一 dataset；沿用現行雙 loader／macro-step。
5. **checkpoint lineage 完整。** 首輪保存 student parent SHA-256、code/config digest、shared manifest、Float／Bit-True role；若後續進入 anchor，再保存兩個 teacher SHA-256、來源 commit與 task-head schema。
6. **額外算力匹配。** 首輪 projection 會增加 per-task gradient buffer／運算；anchor 才有 teacher forward。所有 arm 都匹配 optimizer steps與 exposure，並實測 wall time／VRAM，否則會混入算力與 batch 改變。
7. **formal val 重複挑選。** 候選數保持最小，先 seed 0 淘汰，再只對單一 finalist 補 seeds；不做無界 hyperparameter sweep。

## 十一、如果未來要研究「語意資料 bridge」

相關類別 COCO bridge（person／sports ball／baseball bat）有直覺吸引力，但不列首輪：現行 joint Detect head 是 COCO80，BBAT Pose head是二類，直接把三類 COCO 轉成 BBAT head 會混入 class mapping、head initialization 與背景標註完整性問題。若日後獨立立案，至少需要：

```text
B0：直接 fine-tune
B1：相關類別 bridge
B2：相同影像數、相同 steps 的隨機 COCO 類別 bridge
```

只有 `B1 > B0` 且 `B1 > B2` 才能說相關語意是原因。任何 COCO subset 都必須保存 deterministic manifest；它不是 BBAT5 新版本，也不能取代 canonical BBAT5。這條路比直接使用既有 standalone teachers 多一個資料與 class-remapping 變因，所以優先級較低。

## 十二、創新邊界

本提案的各元件都有既有先例：LR range test、distillation、object-focused detector imitation、HOG prediction target、multi-task gradient surgery 都不是新的。可能形成「本專案方法貢獻」的是以下受控組合：

1. 不是泛用地套對稱 PCGrad，而是依 durable stage logs 建立 Detect-primary、只投影 secondary Pose 衝突分量的單向假說。
2. projection 僅作用於 active shared-parameter manifest，兩個 task heads 維持原生 gradient，且用 person／ball-pose tight gates反制單邊偏袒風險。
3. 若方向衝突不是充分原因，不疊方法，而以兩個 task-specific standalone functions另作短暫 anchor bridge。
4. 把 filter 從推論 input modality 改成可移除的 object-masked P2/P3 training target，並以 LUMA control區分 structure 與一般 deep supervision。
5. 所有結論綁定 immutable BBAT5 lineage、matched steps、paired seeds、數值不變量與推論圖等價。

這可以稱為「新提案」或「本專案創新方向」，但在完成更廣泛 related-work／專利檢索和三 seed 實驗前，不應寫成學術上的全球首創。

## 十三、建議最終決策

```text
現行 Full35 J0→J3 + immutable datasets
                  │
                  ▼
        engineering invariants probe
                  │
        ┌─────────┴─────────┐
        │失敗               │通過
        ▼                   ▼
     修實作／停止      G0-MATCH vs G1-APC-DETECT
                            │
               ┌────────────┴────────────┐
               │ seed-0 tight gate 失敗  │通過
               ▼                         ▼
          G1 = rejected             補 paired seeds 1/2
               │                         │
   conflict 已消失但 AP 仍掉？        三 seed gate
        ┌──────┴──────┐             ┌────┴────┐
        │否           │是           │失敗     │通過
        ▼             ▼             ▼         ▼
      結束       另開 A0/A1      rejected  validated
                 anchor 單因子

HOG/Sobel companion 永遠另案，不能堆進 G1 或 A1 首輪。
```

最值得先做的不是 DOTA、WST、五組 LR、GradNorm 或多方法疊加，而是 **一組 matched baseline 加一組 Detect-priority asymmetric conflict projection**。現有衝突雖然 stage mean cosine 略為正，卻有 `32.258%–41.176%` 的更新事件為負；目前尚不能由 count 判斷負向幅度強弱，但其頻率不低，而且在 J2/J3 norm ratio 已低於 1 時仍存在，足以成為首輪最小、可直接推翻並同步量出 correction magnitude 的訓練假說。此方案不改 dataset／inference graph；若它只修正 cosine、無法補回 AP，就按計畫停止並把 dual-teacher anchor 降為下一個獨立候選。

## 一手來源

- 使用者提供論文，第 3.1.1–3.1.4 節（本機／歷史參照：`../../Design%20and%20Implementation%20of%20a%20Multi-Precision%20Deep%20Learning%20Accelerator%20for%20YOLOX-Based%20Object%20Detection%20in%20Synthetic%20Aperture%20Radar%20Images.pdf`；未隨本次報告發布）
- [SARDet-100K／MSFA，NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/hash/e7eb8128eb26eafbe901348df1dbacdc-Abstract-Conference.html)
- [SARDet-100K 官方 repository](https://github.com/zcablii/SARDet_100K)
- [MSFA 官方 filter／input 實作](https://github.com/zcablii/SARDet_100K/blob/main/MSFA/msfa/models/backbones/MSFA.py)
- [Cyclical Learning Rates／LR range test](https://arxiv.org/abs/1506.01186)
- [Hyperband](https://www.jmlr.org/papers/v18/16-558.html)
- [Learning without Forgetting](https://arxiv.org/abs/1606.09282)
- [Fine-Grained Feature Imitation for Object Detection](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Distilling_Object_Detectors_With_Fine-Grained_Feature_Imitation_CVPR_2019_paper.html)
- [MaskFeat](https://openaccess.thecvf.com/content/CVPR2022/html/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.html)
- [AugMix](https://openreview.net/forum?id=S1gmrxHFvB)
- [Generalized Distillation／Privileged Information](https://arxiv.org/abs/1511.03643)
- [PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
- [GradNorm](https://proceedings.mlr.press/v80/chen18a.html)

## 本輪限制

- 沒有啟動 GPU training／validation，也沒有新增結果。
- 沒有修改 production code、模型 YAML、dataset、label、split、checkpoint 或 queue。
- 指定論文的數字是該論文自己的 YOLOX／SARDet 設定；未自行重現。
- 本地 Full35 結果目前只有 seed 0，故所有新 gate仍須 paired seeds 才能升格。
- 困難：PDF 禁止 copy但可由本機 `pdftotext` 完整讀取；未影響第 3.1 節表格與文字稽核。其餘無。
