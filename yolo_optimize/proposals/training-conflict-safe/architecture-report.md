# 衝突安全訓練：目前、修改後與推論架構

> 2026-09-08 本輪維持COCO80+BBAT5，person-only不是前置；同一stage負cosine≥20%且correction median≥0.02才開；與HOG不疊，parent/順序以總計畫為準。詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

> 狀態：訓練機理提案；尚未修改 production trainer，尚未跑 matched GPU 實驗
>
> 方向 ID：`OPT-TRAIN-CONFLICT-SAFE`

在 `yolo_optimize` 目錄可用下列命令於 terminal 重看：

~~~bash
sed -n '1,360p' optimizations/training-conflict-safe/architecture-report.md
~~~

## 0. 一句話判斷

不要再把「多階段訓練」重做一次；Full35 已經有 J0→J3。現在可直接干預的訊號是：J1–J3 約有
32–41% 的 sampled shared updates 出現 `g_detect · g_pose < 0`。建議只在這些事件中移除 Pose 沿
`-g_detect` 的分量，Detect 與兩個 task heads保持原樣。

## 1. A：目前的 joint macro-step

~~~text
COCO Detect batch ──> L_detect ──> backward ──> g_detect(shared) ──────┐
                                                                        │
BBAT5 Pose batch ───> L_pose ────> backward ──> g_pose(shared) ────────┤
                                                                        ↓
                                                          g_shared = g_detect + g_pose
                                                                        │
                                                                        ▼
                                                        global clip → AdamW → EMA

Detect-specific params：g_detect_only ────────────────────┘
Pose-specific params：  g_pose_only   ────────────────────┘
~~~

現行 [`MacroStepEngine.run()`](<../../../yolo_combine/src/yolo_combine/joint_loss.py>)會先依序完成
Detect、Pose backward，再 unscale、finite check、global clip與單次 optimizer step。它已能在抽樣 cadence保存
Detect snapshot，並從 joint gradient回推出 Pose shared gradient；因此工程 seam存在，不需重寫 loader或 loss。

若 `g_d^Tg_p<0`，直接相加會讓兩個 task互相抵銷：

\[
\lVert g_d+g_p\rVert^2
=\lVert g_d\rVert^2+\lVert g_p\rVert^2+2g_d^Tg_p.
\]

負交叉項會縮小合成步，甚至讓其中一個 task的一階 loss方向惡化。降低全域 LR只會縮短同一個方向，不能把負內積
變成非負。

## 2. B：預計的 Detect-priority 非對稱投影

~~~text
COCO/person Detect batch ──> L_detect ──> g_detect(shared) ────────────────────────┐
                                                                                   ├─> global dot
BBAT5 Pose batch ──────────> L_pose ────> g_pose(shared) ──────────────────────────┘
                                                        │
                                    ┌───────────────────┴──────────────────┐
                                    │ dot >= 0                             │ dot < 0
                                    ▼                                      ▼
                         g_pose_safe = g_pose          移除 Pose 沿 -g_detect 的分量
                                    │                                      │
                                    └───────────────────┬──────────────────┘
                                                        ▼
                                  g_shared = g_detect + g_pose_safe
                                                        │
                                                        ▼
                                       global clip once → AdamW → EMA

Detect-specific params：原生 g_detect_only，不進 projection
Pose-specific params：  原生 g_pose_only，  不進 projection
~~~

精確公式：

\[
g_p^{safe}=
\begin{cases}
g_p-\dfrac{g_p^Tg_d}{\lVert g_d\rVert_2^2+\epsilon}g_d,&g_p^Tg_d<0,\\[6pt]
g_p,&g_p^Tg_d\ge0.
\end{cases}
\]

當 `epsilon` 相對 `||g_d||²` 很小：

\[
g_d^Tg_p^{safe}\approx0.
\]

這表示 Pose 和 Detect正交的學習分量仍保留，只移除當次直接反向的部分。它不是「Pose gradient設為0」，也不是
把 Detect loss weight放大。

## 3. 插入現有 J0→J3 的位置

~~~text
J0：Pose head only
    └─ shared trunk frozen；維持原生 path，不做 projection

J1：Neck + two heads
    └─ 只對 J1 active shared manifest 做 global projection

J2：layer9+ backbone + Neck + MASF + two heads
    └─ 重新產生 J2 manifest；不可沿用 J1 tensor list

J3：full low-LR refinement + 可微 attention
    └─ 重新產生 J3 manifest；Binary sign／硬體固定項仍遵守既有 freeze policy
~~~

person-only H1/H2會先改 Detect head與 supervision，所以正式 G0/G1應在該方向決選後重新開始。舊 logs只證明
值得開方向，不能替代新 head下的 trigger screen。

## 4. 數值執行順序

~~~text
1. optimizer.zero_grad(set_to_none=True)
2. Detect macro accumulation（現行 task weight已包含）
3. snapshot Detect shared grads（仍帶 AMP scale）
4. Pose macro accumulation（現行 task weight已包含）
5. scaler.unscale_(optimizer)
6. non-finite / overflow gate
7. 以 FP32 buffer取得 g_detect、g_pose與 global dot
8. dot < 0 才建立 g_pose_safe並回寫 shared .grad
9. global clip一次
10. scaler.step、scaler.update、zero_grad、EMA update各一次
~~~

不能在 Detect backward後先 step，也不能分別 clip兩個 task。projection若放在 AMP unscale前，scale與 overflow
行為容易錯；若放在 global clip後，方向與預定公式又不一致。

## 5. 為何是 global shared vector，不是逐 layer

首輪把所有 active shared tensors視為一個向量：

\[
d=\sum_l\langle g_{p,l},g_{d,l}\rangle,\qquad
n_d=\sum_l\lVert g_{d,l}\rVert^2.
\]

再以同一係數 `d/(n_d+epsilon)`修正每層。逐 layer投影會讓大量局部負 dot各自觸發，通常修正更強，也多出
scope/granularity這個變因。若 global版本失敗，不能在同一方向臨場改 per-layer把結果救回來。

## 6. 目前證據表示「值得測」，不是「一定有效」

| Stage | cosine mean | 負事件率 | 負事件 `|cos|` median | 負事件 `|cos|` max |
|---|---:|---:|---:|---:|
| J1 | 0.042682 | 32.258% | 0.027935 | 0.182819 |
| J2 | 0.018512 | 32.759% | 0.039848 | 0.107548 |
| J3 | 0.026841 | 41.176% | 0.041183 | 0.126632 |

對 global orthogonal projection，忽略 epsilon時：

\[
\frac{\lVert g_p^{safe}-g_p\rVert}{\lVert g_p\rVert}=|\cos(g_d,g_p)|.
\]

所以歷史典型修正量約 2.8–4.1%，並不巨大。這正是要先做 matched兩臂、而不是直接宣稱能補 AP的原因。

## 7. C：正式推論 graph完全不變

~~~text
訓練結束：只保存 student weights

RGB image → shared YOLO26 layers0–22 ─┬─> person Detect([P3,P4,P5])
                                      └─> BBAT5 Pose26([P3,P4,P5])

不存在：gradient snapshot、dot、projection、teacher、HOG、P2 side head
~~~

因此可以說「inference graph理論零增量」，不能說整個方法零成本；訓練仍多出 gradient buffer、dot/rewrite與
可能的 VRAM／wall-time成本。

## 8. 最小決策圖

~~~text
person head winner + same-lineage parents
                    │
                    ▼
       gradient trigger screen + numerical tests
                    │
       ┌────────────┴────────────┐
       │訊號／工程 gate失敗       │通過
       ▼                         ▼
  不啟動／修實作           G0-MATCH vs G1-APC-DETECT
                                      │
                       ┌──────────────┴──────────────┐
                       │ seed-0 metric gate失敗       │通過
                       ▼                             ▼
                   rejected                  補 paired seeds 1/2
                                                     │
                                              三-seed gate
                                              ┌──────┴──────┐
                                              │失敗         │通過
                                              ▼             ▼
                                          rejected       validated
~~~

若 post cosine已正確歸零但 AP不回升，代表 raw conflict不是充分根因。本方向到此結束；dual-teacher anchor或
Sobel/HOG companion各自另案，不堆進 G1。

返回[方向說明](<README.md>)、[最小計畫](<plan.md>)或[優化方向索引](<../README.md>)。
