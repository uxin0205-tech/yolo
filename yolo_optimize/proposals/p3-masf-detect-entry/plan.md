# P3 MASF Detect-entry：基礎訓練後最小優化計畫

> 2026-09-08 按總計畫S5執行，no-MASF FP-QK parent；新整合版固定5epoch recovery（待驗證提案），不同於舊未定長度Stage A；實際DualHead owner須解析。詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

本計畫是主模型 FP 訓練完成後的**小範圍 recovery optimization**，不是完全免訓練的 post-process。
方向證據見[README](<README.md>)，目前／希望資料流見[終端架構圖報告](<architecture-report.md>)。

## 一、放在整體流程的哪裡

~~~text
BASE-FP 主訓練完成（無本方向的 MASF）
              │
              ▼
凍結 baseline checkpoint、digest、完整 validation
              │
              ▼
本方向：Detect-entry MASF 兩臂 recovery
              │
              ▼
選出並凍結最終 FP winner
              │
              ▼
RepConv deployment fuse／BinaryQK／QAT／PTQ／calibration／export
~~~

因此「最後訓練完後」應理解為**主訓練完成後**；這個方向本身仍包含一次很短、很小範圍的 recovery。
只要後面還要做 BinaryQK、INT8 或 calibration，就必須先決定要不要保留 MASF，否則後續量化證據會失效。

若 RepConv 仍在做需要訓練的架構實驗，它不能和本輪一起加入；先固定 RepConv 狀態，再讓兩個 arm
完全相同。只有 deployment-time fusion 可放在選出 FP winner 之後。

## 二、唯一可接受的 parent

兩個 arm 必須從同一個、已完成主訓練的 `BASE-FP` checkpoint 複製：

- 不含本方向的 MASF。
- 尚未做 BinaryQK／PTQ／INT8 calibration。
- checkpoint digest、程式 commit、環境與完整 baseline metrics 已保存。

既有 `model.16.p3_masf.*` shared-seam checkpoint 不能當 parent；它的 P4/P5 與 head 已對舊 routing
共同適應。若手上只有該 checkpoint，應回到它加入 MASF 前的 FP parent，而不是直接搬動權重。

## 三、只做兩個必要的新訓練 arm

| Arm | 從相同 BASE-FP 開始 | Trainable | 用途 |
|---|---|---|---|
| `CTRL-P3` | 不加 MASF | P3 predictor | 排除「只是多 recovery 幾個 epoch」造成的增益 |
| `MASF-P3` | graft Detect-entry Partial75 | 同一組 P3 predictor + MASF | 測量 MASF 的真正額外收益 |

現行 shared-seam 只保留為歷史證據與 legacy loader，**不重新訓練第三個 arm**。只有論文需要嚴格回答
「共享位置和 Detect 位置誰更好」時，才另立研究實驗；部署選模不需要為此增加 GPU 成本。

## 四、兩臂必須完全相同的設定

- 相同 parent checkpoint、dataset、split、seed、image size、batch、augmentation、epoch、patience、
  optimizer、scheduler、AMP 與 evaluation。
- 沿用已正式使用的 Stage A recovery 長度，不做 epoch／LR sweep。
- P3 predictor LR 兩臂都使用 `1.9e-4`。
- `MASF-P3` 額外給 MASF LR `3.8e-4`；CTRL 不建立空的 MASF param group。
- P4/P5 predictors、`model.16/17/19/20/22`、attention、其餘 Neck 與 backbone 全部 frozen。

兩臂共同 trainable P3 predictor：

~~~text
model.23.cv2.0.*
model.23.cv3.0.*
model.23.one2one_cv2.0.*
model.23.one2one_cv3.0.*
~~~

`MASF-P3` 另外 trainable：

~~~text
model.23.p3_masf.*
~~~

使用 trainable-name assertion；白名單外只要出現一個 trainable tensor，就在 training 前直接失敗。

## 五、先過工程 gate，不浪費 GPU

Detect adapter 的外部 interface 保持 `[P3, P4, P5] → Detect outputs`，MASF routing 藏在
`model.23` implementation 內。只保留以下必要測試：

1. **Parent equivalence**：graft 後 gate=0，完整模型 output 必須和 BASE-FP `torch.equal`。
2. **影響隔離**：gate-on/off 時 layer16 raw P3、layer19 P4、layer22 P5 必須 `torch.equal`；只有
   P3 Detect input 改變。
3. **State locality**：parent state tensors 全保留；新增 tensors 只能是 `model.23.p3_masf.*`。
4. **Lifecycle**：Float save/reload、Bit-True materialization 與 early-stop deep-copy 保留同一 owner；
   legacy shared-seam checkpoint 只能唯讀載入。

四項任一失敗就不啟動訓練。

## 六、最省資源的執行順序

### Step 1：一組 paired seed 篩選

以同一 seed 各跑一次 `CTRL-P3` 與 `MASF-P3`，總共只新增兩個 training jobs。

### Step 2：早停判斷

`MASF-P3 - CTRL-P3` 必須同時符合：

- overall AP50-95 delta `≥ +0.001`。
- AP_S 與 sports-ball 不退化。
- baseball-bat 不被犧牲。
- latency、參數量與峰值記憶體在訓練前登錄的部署預算內。

任一失敗：停止此方向，保留 `CTRL-P3` 或原 BASE-FP，不調 alpha、不加入 learned selector、不加入
P2 lateral，也不做超參數 sweep。

### Step 3：只有通過才補 paired seeds

第一組通過後，再補兩組 paired seeds，讓 `CTRL-P3` 與 `MASF-P3` 各自總計 3 seeds。以 paired
mean 套用相同 gate；通過才把方向改成 `validated`，否則改成 `rejected`。

~~~text
初篩成本：2 jobs
通過後總成本：6 jobs
未通過時：不再追加 jobs
~~~

## 七、資料與結果契約

使用和 BASE-FP 相同的正式 validation。若使用 BBAT5 detection，只能使用不可變的
`/home/uxin/yolo/original/pose/derived/bbat5-v1/` 與正式 `configs/detect.yaml`；不得重切 split、
抽樣或修改影像／標註。

每個 arm 只需保存：

- config snapshot、seed、commit、parent digest 與 trainable names。
- best／last checkpoint digest。
- overall、AP50、AP75、AP_S/M/L、sports-ball、baseball-bat、precision、recall。
- 參數量、latency、峰值記憶體與 paired delta。

結果摘要寫入同目錄 `results.md`；大型 checkpoint 與原始 artifacts 留在正式實驗目錄。

## 八、最後決策

| 結果 | 動作 |
|---|---|
| 工程 gate 失敗 | 修正 adapter；不訓練 |
| 第一組 paired seed 未過 | `rejected`，停止 MASF |
| 三組 paired mean 未過 | `rejected`，保留 CTRL／BASE-FP |
| 三組 paired mean 全部過 gate | `validated`，凍結 MASF-P3 FP winner，再進量化／部署流程 |

本計畫不包含 learned selector、per-channel dynamic gate、P2→P3 lateral、RepConv 架構替換或
BinaryQK。它們若要做，必須各自成為另一個優化方向。

返回[方向說明](<README.md>)或[優化方向索引](<../README.md>)。
