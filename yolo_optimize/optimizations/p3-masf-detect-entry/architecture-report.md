# P3 MASF：目前、預計修改與條件式後續架構

> 2026-09-08 按總計畫S5執行，no-MASF FP-QK parent；新整合版固定5epoch recovery（待驗證提案），不同於舊未定長度Stage A；實際DualHead owner須解析。詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

> 狀態：架構分析／尚未修改 production model／尚未以 GPU AP 證實
> 方向 ID：`OPT-P3-MASF-DETECT-ENTRY`

在 repository 根目錄可用下列命令於 terminal 重看本報告：

~~~bash
sed -n '1,360p' optimizations/p3-masf-detect-entry/architecture-report.md
~~~

## 0. 一句話判斷

不是「MASF 不應融合」，而是：

- 若目標是**只提升 P3、小物件或 ball**，不要把 MASF 直接串在 P3/P4/P5 共用的 `model.16`
  output；先改成只供 P3 predictor 使用的 Detect-entry fork。
- 若目標是**讓 MASF 重塑整個 feature pyramid**，現行 shared-seam 才是合理候選，但必須明確把它
  當作三尺度改造、配合全尺度共同適應與 matched control，不能稱為「只增強 P3」。

依目前設計目標與量測證據，第一種情境才符合本專案需求。因此最值得先做的是 Detect-entry
`MASF-P3`；現行直接融合只保留為歷史證據與 legacy loader，不再新增訓練 arm。

## 1. A：一開始／目前的直接融合架構

~~~text
layer16：p3_raw → MASF → p3_masf ──────┬─> layer17 → layer19：p4_affected → layer20 → layer22：p5_affected
                                       │
                                       └───────────────────────────────────────────────────────────┐
layer19：p4_affected ──────────────────────────────────────────────────────────────────────────────┤
layer22：p5_affected ──────────────────────────────────────────────────────────────────────────────┤
                                                                                                  ↓
                                                         Detect([p3_masf, p4_affected, p5_affected])
~~~

圖中的 `affected` 不是表示 layer 19／22 各有一個 MASF，而是表示它們的上游已由 `p3_raw` 變成
`p3_masf`。MASF owner 與 state path 仍是 `model.16.p3_masf.*`。

### 1.1 這個圖真正代表什麼

現行 graft 不是建立一條 P3-only branch，而是把 `model.16.forward()` 的正式輸出改成：

~~~text
p3 = M(z3)
p4 = N4(p3)
p5 = N5(p4)
L  = L3(H3(p3)) + L4(H4(p4)) + L5(H5(p5))
~~~

令 MASF 參數為 `θ`，它收到的梯度包含三條路徑：

~~~text
∂L/∂θ =
    ∂L3/∂p3 · ∂M/∂θ
  + ∂L4/∂p4 · ∂N4/∂p3 · ∂M/∂θ
  + ∂L5/∂p5 · ∂N5/∂p4 · ∂N4/∂p3 · ∂M/∂θ
~~~

所以 MASF 不只在學「怎麼幫 P3」，它同時要滿足 P3、P4、P5 的 loss。三條梯度若方向衝突，就可能
出現 P3 small-object 有局部收益，但 P4/P5 或其他類別把 overall AP 抵銷的情況。

### 1.2 已量到的跨尺度影響

正式 checkpoint 的 alpha-on／alpha-zero CPU dependency：

| 模型 | P3 layer 16 | P4 layer 19 | P5 layer 22 |
|---|---:|---:|---:|
| Full35 A2 | 27.1% | 17.4% | 8.5% |
| Partial75 A2 | 21.1% | 23.4% | 9.8% |

Partial75 對 P4 的 dependency `23.4%` 比 P3 的 `21.1%` 還高，已直接排除「目前只改 P3」的解讀。
這些數值證明 dependency，不等於真實影像 AP。

## 2. 為什麼目前精度沒有如預期增加

~~~text
設計意圖：P3 enhancement → small object / ball 改善
實際行為：P3 enhancement → P3 + P4 + P5 共同改變
                                        │
                                        ├─ P3 局部收益
                                        ├─ P4/P5 分布漂移
                                        └─ 類別 trade-off
                                                │
                                                ▼
                                      overall AP 幾乎沒有增加
~~~

目前 canonical 結果正符合這個症狀：

| Partial75 A2 相對 A0 | 差值 |
|---|---:|
| overall AP | +0.000107 |
| AP_S | +0.001559 |
| sports-ball AP | +0.002763 |
| baseball-bat AP | -0.005135 |

因此不能說 MASF 完全沒用；它對 AP_S 與 sports-ball 有訊號。但它現在的位置讓局部收益伴隨額外
尺度與類別擾動，最後沒有通過 `+0.001` material-gain gate。

另外，即使改對 seam，stride-8 P3 MASF 仍只能重混既有 context，不能重新創造 stride-4 P2 已丟失的
細節；這是另一個結構上限，但不應和位置修正同一輪改。

## 3. B：預計修改的 Detect-entry 架構

~~~text
layer16：p3_raw ─────────────┬─> layer17 → layer19：p4_raw → layer20 → layer22：p5_raw
                             │
                             └─> model.23 內 MASF → p3_det ───────────────────────────────┐
layer19：p4_raw ──────────────────────────────────────────────────────────────────────────┤
layer22：p5_raw ──────────────────────────────────────────────────────────────────────────┤
                                                                                         ↓
                                                              Detect([p3_det, p4_raw, p5_raw])
~~~

這裡 `p4_raw`／`p5_raw` 的意思是它們沿用未經 MASF 的原始 pyramid route；MASF owner 與新 state
path 是 `model.23.p3_masf.*`。它在程式上屬於 Detect adapter，不是新增一個會改動 layer index 的
獨立 YAML layer。

新的等價式：

~~~text
p3_det = M(z3)
p4     = N4(z3)
p5     = N5(p4)
L      = L3(H3(p3_det)) + L4(H4(p4)) + L5(H5(p5))
~~~

此時對 MASF 參數 `θ`：

~~~text
∂L/∂θ = ∂L3/∂p3_det · ∂M/∂θ

∂p4/∂θ = 0
∂p5/∂θ = 0
~~~

這才符合「MASF 只服務 P3 predictor」的原始目標。注意：`model.16` 仍不能在第一輪解凍；如果它的
權重改變，P4/P5 仍會因共同使用 `z3` 而改變，位置隔離實驗就不再乾淨。

## 4. 原本程式概念與建議程式概念

### 4.1 原本：升級共享 `model.16`

~~~python
class C3k2P3MASFPartial75(C3k2):
    def forward(self, x):
        return self.p3_masf(super().forward(x))

layer = model.model[graph.p3_index]      # model.16
layer.__class__ = C3k2P3MASFPartial75
layer.add_module("p3_masf", module)
~~~

### 4.2 建議：只升級 Detect entry

~~~python
class DetectP3MASFPartial75(Detect):
    def forward(self, features):
        detect_features = list(features)  # 不原地修改 caller list
        detect_features[0] = self.p3_masf(detect_features[0])
        return super().forward(detect_features)

detect = model.model[graph.detect_index]  # 現行圖為 model.23
detect.__class__ = DetectP3MASFPartial75
detect.add_module("p3_masf", module)
~~~

Detect 的 `f=[16, 19, 22]` 與 layer index 不需要重排。舊 `C3k2P3MASFFull35/Partial75` class 必須
保留，讓 legacy checkpoints 能唯讀反序列化；舊權重不能靜默搬到新 seam 後當作正式可比較結果。

## 5. A/B 架構差異總表

| 項目 | A：目前 shared-seam | B：預計 Detect-entry |
|---|---|---|
| MASF owner | `model.16` | `model.23` Detect |
| state path | `model.16.p3_masf.*` | `model.23.p3_masf.*` |
| P3 Detect input | 改變 | 改變 |
| P4 feature | 會改變 | gate-on/off bit-exact |
| P5 feature | 會改變 | gate-on/off bit-exact |
| MASF gradient | P3 + P4 + P5 loss paths | 只有 P3 loss path |
| 首輪公式 | Partial75 | 同一個 Partial75 |
| 可回答的問題 | 三尺度共同重塑是否有用 | 只改 seam 是否更符合 P3 目標 |
| 目前角色 | 歷史證據／legacy | `MASF-P3` 候選 |

CPU memory prototype 已在同 parent、同 MASF 權重、`alpha=0.2` 下量到：

| feature change | shared-seam | Detect-entry |
|---|---:|---:|
| P4 | 0.006724 | 0 |
| P5 | 0.004886 | 0 |

這證明 B 的隔離性質成立；尚未證明 B 的 AP 一定比無 MASF 的 matched `CTRL-P3` 高。

## 6. 最小訓練比較

為了選出部署模型，只新增兩個 training arms：

~~~text
同一個無 MASF BASE-FP
           │
           ├─> CTRL-P3：只做 P3 predictor recovery
           │
           └─> MASF-P3：Detect-entry Partial75 + 同一組 P3 predictor recovery
                                      │
                                      ├─ 未過 +0.001／類別 guardrail → 停止
                                      └─ 通過一組 paired seed → 再補到共 3 組
~~~

兩臂都做相同 P3 recovery，才能排除「只是多訓練幾個 epoch」的效果。現行 shared-seam 不重跑；
learned selector、dynamic gate、P2 lateral、RepConv 與 BinaryQK 都不放進本方向。

## 7. 修改後必須通過的工程檢查

1. gate=0：parent 與 grafted full-model output 必須 `torch.equal`。
2. gate-on/off：layer 16 raw P3、layer 19 P4、layer 22 P5 必須 `torch.equal`。
3. gate-on：送入 P3 predictor 的 feature 必須改變。
4. parent state tensors 全保留；新增 tensors 只能是 `model.23.p3_masf.*`。
5. Float save/reload、Bit-True materialization、deep-copy early-stop 保留新 owner。
6. legacy `model.16.p3_masf.*` checkpoints 可唯讀載入，但不和新 lineage 混排名。

## 8. 最終建議

~~~text
BASE-FP 主訓練完成後：
  CTRL-P3：無 MASF + P3 predictor recovery
  MASF-P3：Detect-entry Partial75 + 相同 P3 predictor recovery

升級條件：
  MASF-P3 對 CTRL-P3 至少 3 個 paired seeds 的 overall AP mean delta ≥ +0.001，
  同時 AP_S、sports-ball 不退化，baseball-bat 不被犧牲。
~~~

所以答案是：**不要停止融合；要停止把「P3-only enhancement」融合在三尺度共享 seam。**
先把 MASF 移到 Detect 的 P3 entry，與相同 recovery 的無 MASF control 做乾淨 A/B，判斷它是否
值得進入最後 FP winner。

返回[方向說明](<README.md>)、[執行計畫](<plan.md>)或[優化方向索引](<../README.md>)。
