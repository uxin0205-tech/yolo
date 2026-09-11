# 2026-09-01：YOLO26 P3 MASF 無實質增益專項診斷

本報告的可執行方向已整理到
[P3 MASF Detect-entry 優化資料夾](<../../proposals/p3-masf-detect-entry/README.md>)，實作、實驗矩陣、
gate 與停止條件見同目錄的[執行計畫](<../../proposals/p3-masf-detect-entry/plan.md>)。

## 結論

目前最有力的原因不是 MASF 沒有被使用，而是它的**插入邊界與「只增強 P3」的設計意圖不一致**。
現行實作把 MASF 直接接在 `model.16` 的輸出，因此它同時改變：

~~~text
model.16 P3 + MASF ─┬─> Detect 的 P3 輸入
                    └─> model.17 -> model.19 P4
                                      └─> model.20 -> model.22 P5 -> Detect
~~~

正式 checkpoint 的 CPU alpha-on／alpha-zero 差分證實，改動會由 P3 傳到 P4、P5；Partial75 在
P4 的差異甚至大於 P3。換言之，現在的 P3 MASF 實際上是「整個 top-down pyramid 的共同擾動源」，
不是只供 P3 Detect 使用的 enhancement branch。

第二層原因是資訊來源不匹配。MASF 的 DW3/DW5 只在 stride-8 P3 上重新混合既有 context，不能補回
stride-8 之前已丟失的高解析度細節。既有 BBAT5 field-check 顯示典型 ball 約 `16×17 px`，在 P3
只有約 `1.88 cells`。這個模組可以改變 context，但不能創造 P2 級空間資訊。

第三層原因是模組與選模目標都不夠針對小物件：Full35 處理所有通道；Partial75 固定處理前 64 個
通道，沒有 learned selector；DW3/DW5 沒有 input-dependent branch weighting，最後只有一個全域 scalar
`alpha`。正式選模又以 COCO80 overall mAP 為主，局部 small／sports-ball 收益會被 baseball-bat 或其他
類別退化抵銷。

## 一、症狀與紅燈

同機 COCO2017 internal AP50-95：

| 模型 | AP50-95 | 相對 A0 |
|---|---:|---:|
| A0 | 0.506753642 | — |
| Full35 A2 | 0.506391251 | -0.000362391 |
| Partial75 A2 | 0.506754491 | +0.000000849 |

以預先使用的 `+0.001` material-gain gate 判斷，最佳 Partial75 只增加 `0.000000849`，穩定觸發
「沒有實質增益」紅燈。Full35 的下降也小於 `0.001` tie band，因此不能把它宣稱為已證實的穩定回歸。

## 二、已確認：MASF 不是只作用於 P3

正式圖結構為：

| Layer | `from` | 角色 |
|---:|---|---|
| 16 | `-1` | P3 C3k2 + MASF |
| 17 | `-1` | P3→P4 stride-2 Conv |
| 19 | `-1` | P4 C3k2 |
| 20 | `-1` | P4→P5 stride-2 Conv |
| 22 | `-1` | P5 C3k2，內含 attention |
| 23 | `[16, 19, 22]` | Detect(P3, P4, P5) |

`graft_p3_masf()` 以 in-place class upgrade 把 `model.16.forward()` 改成
`p3_masf(super().forward(x))`，沒有建立只供 Detect 使用的分支。因此 layer 17 的 `from=-1` 也直接
吃到 MASF 輸出。

以正式 Float A2 checkpoint、固定 seed `20260901`、`1×3×128×128` synthetic input，在 CPU 上將同一
checkpoint 的 `alpha` 暫時設為零，量到：

| 模型 | alpha | P3 layer 16 | P4 layer 19 | P5 layer 22 |
|---|---:|---:|---:|---:|
| Full35 A2 | +0.170532 | 0.271428 | 0.174071 | 0.085438 |
| Partial75 A2 | -0.361084 | 0.210857 | 0.234279 | 0.098079 |

表內是 `mean(abs(on-zero)) / mean(abs(on))`。這個 synthetic probe 只證明 dependency 與傳播範圍，
不代表真實影像 AP。

另一個 module/output probe 顯示：

| 模型 | MASF output 相對 input L1 | cosine | 模型輸出 alpha dependency L1 |
|---|---:|---:|---:|
| Full35 A2 | 0.295976 | 0.952471 | 0.118066 |
| Partial75 A2 | 0.241157 | 0.863623 | 0.120828 |

所以「alpha 太小、branch 沒學到、MASF 沒進 forward」均可排除。它已大幅改變特徵和輸出，只是改動沒有
轉成 AP。

## 三、指標顯示局部收益被其他目標抵銷

Canonical COCO API 相對 A0：

| 模型 | Overall AP | AP_S | AP_M | AP_L | Sports-ball AP | Baseball-bat AP |
|---|---:|---:|---:|---:|---:|---:|
| Full35 A2 | -0.000223 | -0.001226 | -0.000234 | -0.000412 | -0.001871 | -0.002900 |
| Partial75 A2 | +0.000107 | +0.001559 | -0.000141 | +0.000322 | +0.002763 | -0.005135 |

Partial75 的方向很有資訊量：它不是完全無效，small 與 sports ball 有很小的局部收益；但 baseball bat
下降更多，overall 幾乎歸零。這支持「固定通道／固定 context 對部分尺度有用，但不是穩定的整體特徵
改良」。Full35 全通道混合則沒有出現同樣的局部收益。

歷史 `bbt5-detect-valid` 上 Full35／Partial75 A2 也都低於 A0，但該入口不是現行 canonical
`bbat5-v1`，只能當舊結果的相對佐證，不能替代新的正式 BBAT5 實驗。

## 四、其他高機率原因

### 1. P3 context 不能補回 P2 detail

既有 field-check 的 valid ball 中位數為 `16×17 px`；在 P2／P3／P4 的最短邊約為
`3.75 / 1.88 / 0.94 cells`。DW3/DW5 在 P3 上可擴大 context，但輸入本身已是 stride 8。
若主要錯誤來自 tiny ball 沒被保留下來，單純在 P3 加卷積沒有足夠資訊恢復它。

「高頻被平滑」目前仍是假設；尚未做真實影像 feature-spectrum 或 ball-centered activation probe，
不能把它寫成已證實根因。

### 2. Partial75 的前 64 channels 是任意固定切片

`torch.split(x, (64, 192), dim=1)` 沒有判斷哪些 channels 對 ball／tiny object 有用。Phase A1/A2 又只
訓練 MASF，產生這 256 channels 的 parent projection 是 frozen，無法重新排列語意來配合固定前 64
通道。這限制了 Partial75 的上限。

### 3. 目前不是 input-adaptive multi-scale selection

DW3 與 DW5 的輸出直接相加，再經 dense 1×1 projection；沒有依影像、位置或通道動態選擇 kernel。
單一 scalar `alpha` 對所有 spatial positions／channels 共用。它比較接近 residual multi-kernel context
block，不是能針對小球與球棒分別選尺度的 adaptive fusion。

### 4. 訓練 scope 與選模目標不對齊

- A1／A2 只訓練 MASF，Detect 與其他 Neck frozen；但 trained MASF 已讓輸出差約 12%。
- Full-data Phase B 同時解凍全部 Neck／Detect 後，Full35／Partial75 相對各自 A2 又下降
  `0.002889 / 0.002745`，而且都 rollback。
- Phase B 沒有只解凍 P3 predictor，也沒有 matched no-MASF 同排程 control，因此不能判斷是共同適應
  不足、解凍範圍太廣，或排程本身造成 parent drift。
- Formal selection 主要看 COCO80 overall mAP，不是 BBAT5 ball／bat 的 joint objective。

### 5. 因果 control 不足

A0 沒有用同一 staged schedule 重訓，目前又只有 seed 0。`±0.0004` 級差值很可能包含 validation／
training variance。可確定的是「未達 material gain」，不能僅憑這一組宣稱 MASF 必然下降。

## 五、根因排序

| 排名 | 原因 | 證據強度 |
|---:|---|---|
| 1 | in-place graft 讓 MASF 同時擾動 P3/P4/P5，違反 P3-only 意圖 | 已確認 |
| 2 | P3 stride-8 context 沒有引入新的高解析度資訊 | 已確認的結構限制 |
| 3 | Full 全通道與 Partial 固定前 64 通道缺乏 task-adaptive selection | 已確認設計；效果機制高機率 |
| 4 | COCO overall 目標將 small/ball 局部收益與 bat／其他類別 trade-off 混在一起 | 指標已確認 |
| 5 | A 階段 MASF-only、B 階段一次解凍整個 Neck/Detect，不利於隔離 P3 共同適應 | 高機率；缺 matched control |
| 6 | MASF 平滑掉 ball 高頻細節 | 尚未證實 |

## 六、優先解法

### 1. 第一優先：把 enhancement 改成 Detect-only fork

~~~text
p3_raw = model.16(...)
p4 = model.17(p3_raw) -> ... -> p5
p3_det = p3_raw + gate * Context(p3_raw)
Detect([p3_det, p4, p5])
~~~

這樣 MASF 的梯度與特徵改動只服務 P3 Detect，P4/P5 仍沿用 parent 的 `p3_raw`。目前的 in-place
class upgrade 做不到這個隔離，必須建立明確 graph node／Detect input adapter。

### 2. 第二優先：保留 Partial，改成可學 selector

- 從 25% context 起步，不再先測 Full35。
- 使用 exact-zero per-channel gate，而不是所有 channels 共用 scalar `alpha`。
- DW3/DW5 使用 normalized learned weights；若要 input-adaptive，再加極輕量 branch gate。
- 不固定「前 64 channels」；改用 learned `1×1` reducer/expander 或 channel mask。

每次只改一個因子，RepConv 與 BinaryQK 在此矩陣中固定不動。

### 3. 若真正瓶頸是 tiny ball，另測 P2→P3 lateral

例如只在 Detect fork 使用：

~~~text
p3_det = p3_raw + gate3 * Context(p3_raw) + gate2 * Downsample(P2)
~~~

它能帶入 P2 detail，但成本與 assignment 風險較大，必須當獨立實驗，不能和 MASF fork 同輪混改。

### 4. 改訓練與 gate

1. C0：無 MASF，以完全相同 staged schedule 重訓。
2. C1：現行 in-place Partial75，作為位置 control。
3. C2：Detect-only Partial75 fork，其他條件與 C0/C1 完全相同。
4. C3：只有 C2 過 gate 才加入 per-channel gate／learned branch weighting。
5. Stage A 改為 MASF + P3 predictor 共同低 LR 訓練，P4/P5 路徑保持 parent；必要時加 parent
   feature/logit distillation。
6. 工程篩選可先用固定 seed；正式 winner 與 C0 至少做 3 個 paired seeds。

選模至少同時要求：overall AP 不低於 matched control `0.001`、AP_S／sports-ball 不退化、baseball-bat
不被局部收益犧牲。若最終目標是棒球任務，新實驗必須使用不可變
`/home/uxin/yolo/original/pose/derived/bbat5-v1/` 與正式 `configs/detect.yaml`／`configs/pose.yaml`，
不得沿用歷史資料入口或重切 split。

## 七、最便宜的下一個決策點

在改架構前，先對目前正式 J3 做完整 M0/M1：

| Arm | 操作 | 用途 |
|---|---|---|
| M0 | checkpoint 原樣跑完整八項 validation | 目前部署基準 |
| M1 | 同一 checkpoint，只在 eval 暫設 `alpha=0` | 判斷現有權重對 MASF 的即時依賴 |

CPU probe 已證明模型依賴 MASF，但不能替代 full validation。M1 若持平／改善，才進移除與 recovery；
若下降，現有 checkpoint 必須先保留 MASF，另由 Detect-only fork 做新 lineage。

## 證據來源與限制

- 圖與 graft：`achitechure_1/src/achitechure_1/model.py`、`artifacts/preflight.json`。
- 模組：`achitechure_1/src/achitechure_1/masf.py`。
- 訓練 scope：`achitechure_1/src/achitechure_1/phases.py`、`EXPERIMENT_SPEC.md`。
- 指標：`achitechure_1/final/reports/full35-partial75-ap.csv`。
- ball cell 統計：`yolo_masf/field_check/context_rf_cpu/report.md`。
- 更早 YOLO11/P2 因果拆解見[既有 MASF 診斷](<2026-09-01-masf-regression-diagnosis.md>)。

本輪沒有跑 GPU training/validation，沒有修改 checkpoint、模型、資料、標註或 split。CPU 差分使用
`torch.load(weights_only=True)` 與明確 trusted-global allowlist；結果只證明計算圖 dependency，不證明
任何新架構會提高 AP。

## 八、原本怎麼做，建議改成怎樣

### 8.1 插入 seam：從共享 P3 node 移到 Detect entry

目前 `model.py` 的核心等價於：

~~~python
class C3k2P3MASFPartial75(C3k2):
    def forward(self, x):
        return self.p3_masf(super().forward(x))

graph = inspect_yolo26_graph(model)
layer = model.model[graph.p3_index]  # model.16
layer.__class__ = C3k2P3MASFPartial75
layer.add_module("p3_masf", P3MASFPartial75(channels))
~~~

問題不在 `super().forward()`，而在 MASF owner 是共享的 `model.16`。layer 17 的 `from=-1` 使這個
output 同時成為 P3 Detect feature 與 P4/P5 的上游。

第一輪建議只移動 seam，不同時修改 MASF 公式：

~~~python
class DetectP3MASFPartial75(Detect):
    def forward(self, features: list[torch.Tensor]):
        if len(features) != 3:
            raise ValueError("expected [P3, P4, P5]")
        detect_features = list(features)  # 不改 caller 的 list
        detect_features[0] = self.p3_masf(detect_features[0])
        return super().forward(detect_features)


graph = inspect_yolo26_graph(model)
detect = model.model[graph.detect_index]  # 現行固定圖是 model.23
p3_source = model.model[graph.p3_index]   # model.16，只用來推導 channels

detect.__class__ = DetectP3MASFPartial75
detect.add_module("p3_masf", P3MASFPartial75(channels))
~~~

Detect 的 `f` 仍是 `[16, 19, 22]`，model index 不需要重排；差別是 MASF 只處理 Detect 收到的 list
副本中第 0 個 tensor。新 state-dict 路徑應為 `model.23.p3_masf.*`，不再是
`model.16.p3_masf.*`。

`GraphReport` 應新增 `detect_index`，並提供單一 accessor，避免其他模組繼續硬編 owner：

~~~python
def get_p3_masf(model: nn.Module) -> nn.Module:
    graph = inspect_yolo26_graph(model)
    detect = model.model[graph.detect_index]
    masf = getattr(detect, "p3_masf", None)
    if not isinstance(masf, nn.Module):
        raise TypeError("Detect entry does not contain P3 MASF")
    return masf
~~~

`checkpoint.py`、`evaluation.py`、`profiling.py`、`queue_worker.py`、`cli.py` 與測試都應改用此
accessor。`phases.py` 以 `.p3_masf.` 判斷 role，本身不依賴 owner index，可繼續使用。

舊 `C3k2P3MASFFull35/Partial75` class 不能直接刪除，否則既有 checkpoint 反序列化會失敗。它們應保留
為 legacy loader；新的 graft 只建立 `DetectP3MASF*`。舊 trained MASF state 也不應默默搬到新 seam
當正式結果，因其 head、P4、P5 已對 shared-seam 行為共同適應；新實驗必須建立獨立 lineage。

### 8.2 MASF 公式：先保持不變，再做第二因子優化

現行 Partial75：

~~~text
x_c = x[:, :64]
x_b = x[:, 64:]
F(x_c) = Project(DW3(x_c) + DW5(x_c))
y = concat(x_c + alpha * F(x_c), x_b)
~~~

第一輪 C2 只搬 seam，公式完全不變，才能回答「位置是否為主因」。C2 通過後，C3 才測：

~~~text
u = Reduce1x1(x)                         # learned channel subspace，不固定前 64 維
(w3, w5) = softmax(BranchGate(u))        # 至少 per-channel；需要時才 input-adaptive
delta = Expand1x1(w3 * DW3(u) + w5 * DW5(u))
y = x + g ⊙ delta                        # per-channel gate，初值精確為 0
~~~

若擔心 gate 無限制放大，可使用 `g = g_max * tanh(g_raw)`，並記錄
`mean(abs(g*delta))/mean(abs(x))`。這比只看 `alpha` 數值更有意義；projection 可翻轉符號，所以
Partial75 的負 alpha 不等於它一定在「扣除 context」。

### 8.3 訓練 scope：從兩個極端改成 P3 predictor joint stage

目前：

~~~text
A1/A2: 只有 *.p3_masf.* trainable
B:     MASF + model.11 之後幾乎整個 Neck/Detect trainable
~~~

建議的 C2 第一輪：

~~~text
Stage A:
  trainable = MASF
            + model.23.cv2.0.*
            + model.23.cv3.0.*
            + model.23.one2one_cv2.0.*
            + model.23.one2one_cv3.0.*

  frozen    = P4/P5 predictors、model.16/17/19/20/22、attention、backbone
  LR        = MASF 3.8e-4；P3 predictor 1.9e-4
~~~

這沿用現有 2:1 LR 比例，只縮小解凍範圍。不要第一輪就解凍 model.16，因為即使 MASF 已搬到 Detect，
改動 model.16 權重仍會再次傳到 P4/P5。

exact-zero gate 的數學副作用是第一個 update 時 context weights 的梯度為零：

~~~text
y = x + g F_theta(x)
dL/dtheta = g * dL/dy * dF_theta/dtheta
~~~

當 `g=0`，先只有 gate 收到梯度；gate 離開零後 context 才開始學。這是 parent-preserving 的代價，
不是 bug。可用短 warmup、較高 gate LR，或在前幾 epochs 加 parent feature/logit distillation，但不要為了
立即讓 context 有梯度而放棄 epoch-0 等價性。

### 8.4 測試：原本缺什麼，修改後要新增什麼

原本測試有檢查：shape、alpha 初值、Partial bypass、gradient、parent tensor preservation、checkpoint
reload。它沒有檢查 enhancement 的影響邊界，因此 shared-seam 也會全部通過。

新的必要 regression tests：

1. `gate=0` 時 parent 與 grafted full-model output 必須 `torch.equal`。
2. gate 開啟前後，layer 16 raw P3、layer 19 P4、layer 22 P5 必須 `torch.equal`。
3. gate 開啟後，送入 P3 predictor 的 feature 必須改變。
4. parent state tensors 全部保留，新 tensors 只能出現在 `model.23.p3_masf.*`。
5. Float save/reload、Bit-True materialization、deep-copy early-stop 都必須保留新 MASF owner。
6. legacy `model.16.p3_masf.*` checkpoints 仍可唯讀載入，但不可與新 lineage 混排名。

## 九、為什麼目前位置會抵銷精度：計算圖推導

令 layer 16 的原始 P3 feature 為 `z3`，MASF residual 為：

~~~text
delta = alpha * F(z3)
M(z3) = z3 + delta
~~~

現行 shared-seam：

~~~text
p3 = M(z3)
p4 = N4(p3)
p5 = N5(p4)
L  = loss(H3(p3), H4(p4), H5(p5))
~~~

一階近似：

~~~text
Delta p3 = delta
Delta p4 ~= J_N4 * delta
Delta p5 ~= J_N5 * J_N4 * delta

dL/dalpha = [dL/dp3
             + J_N4^T * dL/dp4
             + J_N4^T * J_N5^T * dL/dp5]^T F(z3)
~~~

所以 MASF 的更新方向同時受到 P3、P4、P5 路徑牽引。它可能改善小球的 P3 feature，同時改壞中型
bat 或其他尺度，這正好符合 Partial75 的 `AP_S +0.001559`、sports-ball `+0.002763`、bat
`-0.005135`。

Detect-entry：

~~~text
p3_raw = z3
p4 = N4(z3)
p5 = N5(p4)
p3_det = M(z3)
L = loss(H3(p3_det), H4(p4), H5(p5))

dp4/dalpha = 0
dp5/dalpha = 0
~~~

loss assignment 仍可能在預測層級耦合三個 scales，但 alpha 已沒有改寫 P4/P5 feature 的計算圖路徑。
這才是「只改 P3 Detect feature」能做到的最強隔離。

## 十、為什麼只做 P3 context 有上限：資訊與通道推導

### 10.1 空間資訊

對典型 `16×17 px` ball：

~~~text
P2 stride 4:  約 4.00 × 4.25 cells
P3 stride 8:  約 2.00 × 2.13 cells
P4 stride 16: 約 1.00 × 1.06 cells
~~~

field-check 以最短邊統計得到約 `3.75 / 1.88 / 0.94 cells`。`F(P3)` 是 P3 的確定性函數；它能
重新組合 context、提高可分性，但不能重建 stride-8 之前已不存在的像素細節。若主要錯誤是 tiny ball
在 P3 前已消失，只有 P2 lateral／P2 head 或更高輸入解析度能提供新資訊。

因此順序應是：先修 Detect-only seam；若 ball recall 仍卡住，再把 P2→P3 當獨立因子，不能直接把
「P3 MASF 無增益」推導成「加更大 P3 kernel」。

### 10.2 固定前 64 channels

Partial75 使用固定 selection matrix `S` 取出前 64 維：

~~~text
x_c = Sx
delta = S^T F(Sx)
~~~

Phase A 上游 frozen，產生 P3 channels 的 parent projection 無法旋轉 feature basis 配合 `S`。若 ball
訊號分散在其餘 192 channels，MASF 永遠碰不到它；若前 64 維同時承載 bat／背景語意，context 又會
產生 trade-off。learned reducer `R x` 的作用就是讓模型學習應處理的低維 subspace，而不是假定 channel
順序有小物件語意。

### 10.3 固定 branch sum 與單一 alpha

現行 `DW3(x)+DW5(x)` 可透過 kernel、BN、projection 學習全域權重，但沒有依輸入、位置或目標動態
選擇 receptive field；單一 alpha 又同時縮放所有 channels。球與球棒的尺寸、形狀和所需 context 不同，
一個全域方向能幫 ball、傷 bat 是合理的結果。這是 C3 learned weighting 的理由，但仍應排在 seam 修正
之後，否則兩個因子無法歸因。

## 十一、與 MASF-YOLO 論文的差異

原論文不是在 YOLO26 P3 Neck 單獨放一個 DW3/DW5 block。論文使用 YOLOv11-s／VisDrone2019，
建立 P2 high-resolution detection layer 與 shallow-to-deep skip connections；MFAM 位於 backbone，並與
Fusion、IEMA、DASI 組成完整系統。論文 MFAM 也描述四種尺度輸出，包含 `k=7,9` 的 `1×k`／`k×1`
strip convolutions與平行 3×3，而本地版本只有 DW3、DW5、dense 1×1 和 outer residual。
[MASF-YOLO 原始論文](https://arxiv.org/abs/2504.18136)

論文 VisDrone ablation 的 mAP50:95 依序約為：

| 組合 | mAP50:95 | 相對上一列 |
|---|---:|---:|
| YOLOv11-s baseline | 0.294 | — |
| + P2 layer | 0.307 | +0.013 |
| + MFAM | 0.319 | +0.012 |
| + Fusion | 0.321 | +0.002 |
| + IEMA | 0.324 | +0.003 |
| + DASI（完整 MASF-YOLO-s） | 0.329 | +0.005 |

這能證明 MFAM 在該 P2-enabled、VisDrone、YOLOv11-s lineage 有增益；不能證明「簡化 DW3/DW5 放在
YOLO26 model.16」必然有相同效果。當前實驗其實同時改了 dataset、baseline、module formula、插入位置、
head scales 與訓練契約。

## 十二、Detect-entry prototype 驗證

本輪只在 CPU 記憶體建立 prototype，沒有修改正式程式或 checkpoint：

| 性質 | 結果 |
|---|---|
| parent tensors 全保留 | 通過 |
| 新 state path | `model.23.p3_masf.*` |
| gate=0，P3 enhancement output 與 raw P3 bit-exact | 通過 |
| gate=0，完整模型 output 與 parent bit-exact | 通過 |
| gate 開啟，P3 Detect input 改變 | 通過 |
| gate 開啟，layer 16 raw P3 不變 | 通過 |
| gate 開啟，layer 19 P4 不變 | bit-exact |
| gate 開啟，layer 22 P5 不變 | bit-exact |

同一 parent、同一組 Partial75 權重、`alpha=0.2` 的位置 A/B：

| 指標 | 現行 model.16 shared-seam | Detect-entry |
|---|---:|---:|
| P4 相對 gate=0 feature change | 0.006724 | 0 |
| P5 相對 gate=0 feature change | 0.004886 | 0 |
| 兩位置完整輸出相對差 | 0.043793 | — |

在真正的 MASF module input/output 邊界比較，同 parent／同權重的兩個位置皆 bit-exact，證明 A/B 差異
來自 downstream routing，不是初始化或浮點執行差異。這證實 proposed seam 的結構性質；是否提高 AP
仍必須由 C0/C1/C2 matched GPU experiment 回答。

## 十三、修改優先級與停止條件

| 優先級 | 修改 | 成功訊號 | 失敗時動作 |
|---:|---|---|---|
| P0 | C0/C1/C2：只比較 shared vs Detect-entry seam | C2 paired mean 過 gate | 不再增加 MASF 複雜度 |
| P1 | MASF + P3 predictor joint stage | AP_S/ball 上升且 bat 不退 | 調 scope/KD，不解凍整 Neck |
| P2 | learned reducer + per-channel gate | 超過固定前 64 control | 保留簡單 Partial75 |
| P3 | learned DW3/DW5 weights | 對不同尺寸穩定改善 | 移除動態 gate |
| P4 | 獨立 P2→P3 lateral | tiny-ball recall 明顯改善 | 接受 P3 資訊上限 |

任何一層若未超過 matched control，就停止往後疊模組。RepConv、BinaryQK、資料 split、augmentation、
optimizer、seed 在 C0/C1/C2 中必須固定；正式 winner 才跑至少三個 paired seeds。
