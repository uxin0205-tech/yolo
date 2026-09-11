# OPT-P3-MASF-DETECT-ENTRY：P3 MASF 改到 Detect entry

> 2026-09-08 本輪唯一順序依[方向1 master plan](<../integrated-roadmap/direction1-master-plan.md>)：先比較候選選 PSEL，必要時做原生 loss 修復／小範圍 AdamW 適應；基準可靠後才取得 no-MASF recovery bridge，再評估 Detect-only MASF，不能直接搬模組。MASF、HOG、QK、optimizer 不同時首次開；`training_ready=false`，GPU 0。舊 S5 與 5epoch 內容保留為歷史／待驗證提案，詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

| 欄位 | 內容 |
|---|---|
| 狀態 | `proposed` |
| 全域序位 | A0；person task/head與joint BASE-FP定案後 |
| 執行時點 | 無 MASF 的 BASE-FP 主訓練完成後、量化／部署轉換前 |
| 必要新訓練 | `CTRL-P3`、`MASF-P3` 共兩個 arms |
| 主要問題 | 現行 MASF 放在共享 `model.16`，不是只增強 P3 Detect branch |
| 唯一首輪變因 | MASF 插入 seam：`model.16` → `model.23` Detect entry |
| 模組公式 | 沿用現行 Partial75，不在首輪修改 |
| 架構圖 | [目前、修改後與條件式後續架構](<architecture-report.md>) |
| 下一份文件 | [執行計畫](<plan.md>) |

## 方向結論

先不要繼續增加 MASF 複雜度。最值得優先驗證的改法，是把現行 Partial75 的 MASF 從共享的 P3
feature node 移到 Detect entry，讓它只修改 P3 predictor 收到的 feature，並保證 P4、P5 feature
bit-exact 不變。

它適合放在無 MASF `BASE-FP` 主訓練完成後，以短程 P3-only recovery 驗證，並且必須早於
BinaryQK／QAT／PTQ／calibration。這個方向已由 CPU memory prototype 證明「隔離 seam 成立」，但**尚未**由 matched GPU validation
證明能提高 AP，所以目前只能標為 `proposed`，不能標為有效優化。

## 為什麼現行位置不符合原始意圖

目前等價計算圖：

~~~text
p3_raw  = model.16(...)
p3_masf = MASF(p3_raw)
p4      = model.17(p3_masf) -> model.19(...)
p5      = model.20(p4)       -> model.22(...)
Detect([p3_masf, p4, p5])
~~~

因 `model.17.from = -1`，`model.16` 的 MASF output 不只給 P3 Detect 使用，也成為 P4、P5 的上游。
正式 checkpoint 的 alpha-on／alpha-zero CPU probe 已量到 Full35 的 P3／P4／P5 dependency 為
`27.1% / 17.4% / 8.5%`，Partial75 為 `21.1% / 23.4% / 9.8%`。Partial75 對 P4 的相對擾動甚至
大於 P3，因此它實際上是三尺度的共同擾動源。

建議計算圖：

~~~text
p3_raw = model.16(...)
p4     = model.17(p3_raw) -> model.19(...)
p5     = model.20(p4)     -> model.22(...)

p3_det = MASF(p3_raw)
Detect([p3_det, p4, p5])
~~~

production 實作以 Detect adapter 複製 feature list，只替換第 0 個元素：

~~~python
class DetectP3MASFPartial75(Detect):
    def forward(self, features: list[torch.Tensor]):
        if len(features) != 3:
            raise ValueError("expected [P3, P4, P5]")
        detect_features = list(features)
        detect_features[0] = self.p3_masf(detect_features[0])
        return super().forward(detect_features)
~~~

Detect 的 `f=[16, 19, 22]` 與 model index 不必改；新 state path 應是
`model.23.p3_masf.*`，不再是 `model.16.p3_masf.*`。

## 已有證據

同機 COCO2017 internal AP50-95：

| 模型 | AP50-95 | 相對 A0 |
|---|---:|---:|
| A0 | 0.506753642 | — |
| Full35 A2 | 0.506391251 | -0.000362391 |
| Partial75 A2 | 0.506754491 | +0.000000849 |

最佳 Partial75 的 `+0.000000849` 遠低於既定 `+0.001` material-gain gate。Canonical Partial75
相對 A0 雖有 AP_S `+0.001559`、sports-ball `+0.002763`，但 baseball-bat `-0.005135`，顯示局部收益
被類別 trade-off 抵銷。

CPU Detect-entry prototype 已通過：

- gate=0 時完整模型 output 與 parent bit-exact。
- gate 開啟後，送入 P3 predictor 的 feature 會改變。
- gate 開啟前後，raw layer 16、P4 layer 19、P5 layer 22 都 bit-exact。
- parent tensors 全部保留；新 tensors 只在 `model.23.p3_masf.*`。
- 同 parent、同 MASF 權重、`alpha=0.2` 時，shared-seam 的 P4/P5 change 為
  `0.006724 / 0.004886`，Detect-entry 為 `0 / 0`。

以上只證明 routing 被正確隔離，不證明 AP 一定上升。完整推導、probe 方法與限制見
[P3 MASF 專項診斷](<../../docs/research/2026-09-01-yolo26-p3-masf-no-gain-diagnosis.md>)。

## 本方向範圍

首輪納入：

- 保留 Partial75 現行公式，只移動插入 seam。
- 新增 Detect-entry graft、統一 accessor、checkpoint 與 lifecycle 支援。
- 從同一 BASE-FP 建立 `CTRL-P3` 與 `MASF-P3`，兩臂做相同 P3 predictor recovery。
- `MASF-P3` 額外訓練 MASF；P4/P5 predictor、Neck 與 backbone 在兩臂都 frozen。

首輪不納入：

- learned reducer、per-channel gate、dynamic DW3/DW5 weighting。
- P2→P3 lateral。
- RepConv、BinaryQK、量化或資料 split 變更。
- 把既有 shared-seam trained MASF 權重靜默搬到新 seam，並當作可比較正式結果。

本方向不自動延伸模組公式。`MASF-P3` 未過 gate 就停止；若日後仍要研究 learned selector 或
P2→P3 lateral，另建獨立方向。

## 相容性與風險

- 舊 `C3k2P3MASFFull35/Partial75` class 必須保留，供 legacy checkpoint 唯讀反序列化。
- 新舊 seam 是不同 lineage，不能混用 checkpoint 或混在同一排名表中。
- 新兩臂只能從無 MASF BASE-FP 開始；不能把 shared-seam trained checkpoint 當 parent。
- 第一輪不能解凍 `model.16`；其權重一旦改變，P4/P5 又會共同改變，破壞位置隔離實驗。
- stride-8 MASF 沒有引入新的 P2 細節，所以即使 seam 修正，仍可能受 tiny-ball 空間資訊上限限制。

終端可讀的完整資料流與梯度推導見[架構圖報告](<architecture-report.md>)；執行順序、測試、實驗矩陣
與停止條件見[計畫](<plan.md>)。本次整理紀錄見
[工作紀錄](<../../docs/worklogs/2026-09-01-p3-masf-optimization-direction-organization.md>)。

返回[優化方向索引](<../README.md>)。
