# OPT-TRAIN-CONFLICT-SAFE：Detect 優先的衝突安全訓練

> 2026-09-08 本輪依[方向1 master plan](<../integrated-roadmap/direction1-master-plan.md>)；COCO80＋BBAT5 是現任務，person-only 為 `deferred_by_user`、不是前置。G 僅依 master 的 trigger（同一 stage 負 cosine≥20% 且 correction median≥0.02）開啟，與 HOG 不疊；`training_ready=false`，GPU 0。詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

| 欄位 | 內容 |
|---|---|
| 狀態 | `proposed` |
| 全域序位 | 依方向1 master 的 G trigger；COCO80 現任務；person-only 暫緩，不是前置 |
| 主要問題 | Detect 與 Pose 在共享參數上反覆出現負梯度內積，joint model 的 person／ball-pose 仍低於 standalone |
| 首輪唯一變因 | `g_detect · g_pose < 0` 時，只投影 Pose 的衝突分量 |
| 首輪成本 | `G0-MATCH`、`G1-APC-DETECT` 共 2 個 matched training jobs |
| 推論成本 | 0；不改模型 graph、輸入、checkpoint inference schema |
| 詳細研究 | [本地梯度衝突研究](<../../docs/research/2026-09-04-msfa-section31-training-transfer-innovation.md>) |
| 架構圖 | [目前與修改後訓練資料流](<architecture-report.md>) |
| 執行方式 | [最小實驗計畫](<plan.md>) |

> 定位更正：本方向是由 Full35 的 Detect/Pose shared-gradient紀錄衍生的獨立優化，不是指定論文第3.1節
> 最直接的移植。該章逐項適配後的首選是
> [P3 training-only HOG companion](<../p3-hog-companion-training/README.md>)；兩者首輪不可疊加。

## 方向結論

本方向保留「先定位可量測問題、再用最小處理降低 feature shock」的一般原則，但主要證據來自本地
Detect/Pose shared-gradient logs，而不是論文的 filter ablation。指定論文也不應照搬
`COCO → DOTA → SARDet`、五組固定 LR，或把 HOG/WST/Canny 永久接進 RGB 輸入。

目前 Full35 已經具備 J0→J3 分階段解凍、分組 LR、early stop 與 rollback，所以再做一次普通 gradual
unfreezing 不是新的處理。反而 durable logs 已直接量到 Detect／Pose 共享梯度衝突：

| Stage | 記錄數 | 負 cosine | 負衝突率 | Pose／Detect norm ratio median |
|---|---:|---:|---:|---:|
| J1 | 93 | 30 | 32.258% | 3.6785 |
| J2 | 116 | 38 | 32.759% | 0.7506 |
| J3 | 51 | 21 | 41.176% | 0.6656 |

J2／J3 的 Pose norm 中位數已低於 Detect，負衝突卻仍存在，表示這不只是「Pose 梯度太大」，而是方向問題。
因此首測採 Detect-priority Asymmetric Conflict Projection；它只修正當下會抵銷 Detect 的 Pose shared
gradient，兩個 task-specific heads 完全不動。

這仍是待驗證假說。歷史負 cosine 多半幅度不大，事件數不等於 AP 根因；首輪必須同步量
`correction_ratio`，若只修好 cosine 而 AP 沒改善，就停止，不再調 projection strength。

## 原本與建議做法

目前每個 J1–J3 macro-step：

~~~text
Detect microbatches ──> backward ──> g_detect(shared) ───────────┐
                                                                  ├─> 相加 ─> clip ─> AdamW step
Pose microbatches   ──> backward ──> g_pose(shared) ─────────────┘

Detect head：只收到 Detect gradient
Pose head：  只收到 Pose gradient
shared：     g_detect + g_pose，即使兩者 dot < 0 仍直接相加
~~~

建議首輪：

~~~text
Detect microbatches ──> g_detect(shared) ─────────────────────────────┐
                                                                       ├─> dot/cosine
Pose microbatches   ──> g_pose(shared) ── dot<0 才投影 ─> g_pose_safe ┘
                                                        │
shared：     g_detect + g_pose_safe ──> clip once ──> AdamW step
Detect head：原生 g_detect，不投影
Pose head：  原生 g_pose，不投影
~~~

令 `theta_s` 為當前 stage 的 active shared parameters：

\[
g_d=\nabla_{\theta_s}\widetilde L_d,\qquad
g_p=\nabla_{\theta_s}\widetilde L_p.
\]

若 `g_p^T g_d < 0`：

\[
g_p^{safe}=g_p-
\frac{g_p^Tg_d}{\lVert g_d\rVert_2^2+\epsilon}g_d,
\qquad
g_{shared}=g_d+g_p^{safe}.
\]

否則 `g_p_safe=g_p`。這是受 PCGrad 機理啟發的單向版本，不是原論文的對稱／隨機投影。

## 為何要在 person head 定案後執行

現有 logs 的 `g_detect` 來自完整 COCO80 Detect loss，不能冒稱 `g_person`。新的
[COCO person-only head 方向](<../coco-person-specialized-head/README.md>)會先決定 H1 標準 Detect1 或 H2
Detect1-Lite64；之後再用勝出 head 產生新的 `G0-MATCH` 梯度 baseline。此時 `g_detect` 是單類 person
detection 的完整 box／assignment／classification gradient，但仍應叫 `g_detect`，不能縮寫成只有 class-0
classification 的 `g_person`。

若使用者決定維持 COCO80 head，本方向仍可執行，但其語意是保護整個 COCO Detect 任務；報告不得寫成只保護人。

## 首輪範圍

首輪納入：

- 在現有 `MacroStepEngine` 的 AMP unscale 後、global clip 前插入一次 shared-vector projection。
- shared parameter manifest 由 stage ownership 產生，保存 names、numel 與 digest。
- `G0-MATCH` 與 `G1-APC-DETECT` 沿用同一 parent、sampler 順序、optimizer steps、LR、BN 與 gate。
- 記錄 pre/post cosine、projection rate、norm ratio、correction ratio、missing gradients、overflow、wall time與
  peak VRAM。

首輪不納入：

- 對稱 PCGrad、per-layer projection、projection strength sweep 或 GradNorm。
- 新 LR/scheduler、teacher、HOG/Sobel side loss、P2、MASF、RepConv、BinaryQK或資料變更。
- 把現有 J3 歷史結果直接當 `G0-MATCH`；控制組必須由同一新程式版本重跑。

## 從論文保留與捨棄什麼

| 論文第 3.1 節元素 | 本方向判定 |
|---|---|
| 部署預算先行 | 保留；本法正式推論 graph 必須和 control 等價 |
| 五個 LR 各跑 9 epochs | 不照搬；只有 trainable scope 改變才做犧牲性 LR range probe |
| COCO→DOTA→SARDet | 不照搬；DOTA/SAR 的 modality bridge不符合 RGB棒球任務 |
| staged fine-tuning | 已是 Full35 baseline，不冒充新方法 |
| HOG/WST/Canny input | 不放首輪；若日後做，只能另案作 training-only structure target |
| 降低轉移 shock | 轉化為逐 macro-step、可量測的 shared-gradient conflict removal |

## 風險與誠實邊界

- 投影只保證 raw shared gradients 的一階幾何關係，不保證 AdamW 真實參數步或 AP 一定改善。
- Detect 優先可能傷 Pose，所以 ball pose與全部既有 BBAT5 gates都必須比 matched control緊密監控。
- 需要保存 Detect gradient snapshot，training VRAM與 wall time不會是零；只有正式 inference成本為零。
- person head、資料 view 或 joint recipe一旦再改，舊梯度統計失效，必須重新 screen。
- BBAT5 仍只能使用不可變的 `/home/uxin/yolo/original/pose/derived/bbat5-v1/` 與正式
  `configs/pose.yaml`／`configs/detect.yaml`，不得重切或改標註。

返回[優化方向索引](<../README.md>)。
