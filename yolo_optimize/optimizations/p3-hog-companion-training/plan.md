# P3 HOG companion：最小實驗計畫

> 2026-09-08 本輪整合版為 F2-PRE-HOG9，在共同J0之後的J1/J2監督raw P3，MASF留主訓練後；本文post-MASF版本保留，不混用結果。詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

## 0. 目標與狀態

- 狀態：`proposed`；本文件只固定實作與實驗契約，尚未啟動訓練。
- 唯一因果問題：在不改 YOLO26M 正式輸入與推論圖的前提下，training-only 的 P3 HOG 方向監督，是否能改善目前 Full35 的 joint／person／ball-pose 指標？
- 完整方法與推導見[方向說明](<README.md>)，資料流見[架構報告](<architecture-report.md>)。

首輪不一起改 LR、optimizer、scheduler、資料抽樣、person head、MASF、RepConv、BinaryQK、KD 或梯度投影。

## 1. 執行前置條件

只有下列條件全部成立才可實作：

1. person task/head 與 P3 MASF 位置已定案；若上游 graph 後續再改，HOG 結果必須在新 lineage 重跑。
2. 使用 final-lineage Float full-resume checkpoint，不可從 inference-only 或 Bit-True checkpoint續訓。
3. 三臂共用一個在 J0 已註冊、但尚未啟用 `hog_aux` 的 stage-complete parent。
4. COCO80 若仍為正式 Detect，mask 使用所有訓練 annotations；不能暗中只保留 person。
5. BBAT5 只使用不可變的 `/home/uxin/yolo/original/pose/derived/bbat5-v1/` 與正式 `configs/pose.yaml`／`configs/detect.yaml`，不得建立新 split 或標註版本。

之所以要求「dormant-aux J0 parent」，是因為在既有 parent 後才插入新 optimizer group，會把 optimizer-state migration 混入處理差。F0、F1、F2 必須從相同參數結構與相同 optimizer state 開始。

## 2. 凍結的首輪規格

| 項目 | 固定值 |
|---|---|
| 主圖輸入 | 3-channel RGB，與 baseline 相同 |
| feature seam | prediction head 實際消費的 post-MASF P3；目前為 layer 16、256 channels、stride 8 |
| auxiliary head | `Conv1x1(256, 9)`；2,313 parameters，只存在於 training checkpoint |
| HOG target | luminance、9 個 unsigned orientation bins、cell 8、local normalization、`stop_gradient` |
| 空間大小 | imgsz 640 時 `9×80×80` |
| object weighting | train GT box soft mask × local gradient energy；不把 GT 帶入 validation/inference |
| loss reduction | 每張先正規化，再對 physical batch 求和；保持 native `raw_total` 的 batch-sum 語義 |
| optimizer | 現行 AdamW、role LR、warmup/cosine/plateau與 optimizer-state 延續；只新增明確的 `hog_aux` role |
| stage window | J0/J3 關閉；J1 warmup後漸開；J2 前8 epochs漸退到0 |
| 部署 | target generator、aux head與其 state keys全部移除 |

`mu0` 不用 validation 做 grid search。先在固定 batch probe 上選到：

\[
0.05 \le
\operatorname{median}
\frac{\lVert \mu_0 g_{hog}\rVert_2}
     {\lVert g_{native}\rVert_2+\epsilon}
\le 0.15.
\]

這只是避免 auxiliary gradient 太弱或壓過主任務的安全校準，不是精度調參。

## 3. Phase A：先完成工程不變量

### A1. 模組邊界

實作時拆成三個明確元件：

1. `HOGTarget`：從**已完成同一批 resize／flip／color preprocessing** 的 tensor產生停止梯度 target。
2. `HOGCompanionHead`：只接收本次 forward 明確回傳的 P3，不使用可殘留的全域 forward hook。
3. `HOGCompanionLoss`：建立 GT soft mask、逐圖正規化並回傳 batch-summed raw loss。

`hog_aux` 必須在 optimizer 建立前註冊成獨立 semantic role。正式 full-resume 保存其參數、optimizer moments、schedule state與 target config；inference materialization則反向要求完全剝除。

### A2. 必要單元／整合測試

- shape、dtype、device、finite與 local normalization。
- synthetic horizontal flip／resize 後，影像、boxes、HOG target三者對齊。
- 空 box、邊界 box、極小 box與最後不完整 batch不產生 NaN，也不錯算權重。
- microbatch切法不同時，batch-sum後的 raw auxiliary loss等價。
- AMP overflow/replay不重用上一個 task或上一個 step的 stale P3/target。
- `mu=0` 時 F0 的 native loss、gradients與 optimizer update在預先鎖定 tolerance內等同 baseline。
- J0/J3 `hog_aux` 不更新；J1/J2 ownership與 manifest一致。
- full-resume round trip逐值一致；inference state key exact-diff為零。

任一項失敗，只修工程，不進 AP 實驗。

## 4. Phase B：零／短訓 probe

用事前固定的少量 batches／macro steps量測，不選 AP winner：

1. 各類 GT mask中的有效 HOG cell比例，尤其 ball與bat。
2. `g_hog` 對 native shared gradient的 norm ratio與 cosine。
3. `hog_aux`、P3 producer、MASF與正式 heads的 update ratio。
4. clip率、AMP overflow、finite、wall time與 peak VRAM。
5. J1漸開及J2漸退邊界是否連續。

工程 go/no-go：

- ball/bat object mask內若大多數 target都因低能量而無效，停止 P3-HOG；首輪不臨時改成 P2。
- ratio無法在 `0.05–0.15` 內穩定且持續 clip/overflow，停止；只有 `hog_aux` update異常時，才對該 group做犧牲性 LR diagnostic。
- target generation使訓練成本超過事前預算，先停下回報，不用少跑 optimizer steps補償。

## 5. Phase C：最小正式矩陣

三臂固定相同 code revision、parent SHA、auxiliary initialization、seed、sampler順序、augmentation、optimizer state、macro exposure、task weights與 validation selector：

| Arm | Temporary head | Target／loss | 因果用途 |
|---|---|---|---|
| `F0-MATCH` | 已註冊但不做 forward | 無；`mu=0` | 排除新程式與 optimizer group本身 |
| `F1-LUMA9` | 相同 `256→9` | 9-bin soft local luminance | generic companion supervision負控制 |
| `F2-HOG9` | 相同 `256→9` | normalized HOG orientation | 測方向／形狀先驗 |

若算力只允許最簡版本，先跑 `F0/F2`。只有 `F2` 對 F0出現至少 `+0.001` 的 joint／person訊號才補 F1；但在 F2沒有勝過 F1以前，只能說「整個 HOG recipe有觀測增益」，不能說原因是 HOG orientation。

## 6. Phase D：判定與重複

seed 0 的 tight gate：

- joint score相對 F0至少 `+0.001`；
- 八項正式指標任一項不得低於 F0超過 `0.001`；
- COCO person或 ball pose至少一項改善 `0.002`；
- 若要宣稱 HOG-specific，`F2 > F1`；
- Float／Bit-True、finite、checkpoint、資料 digest及部署等價 gate全部通過。

流程：

```text
seed0：F0 / F1 / F2
  ├─ F2 未過 tight gate ──> rejected；停止本方向
  └─ F2 通過
       ├─ 只判斷產品收益：F0 / F2 補 paired seeds 1、2
       └─ 要主張 HOG 機理：F0 / F1 / F2 全補 paired seeds 1、2
```

三個 seeds完成前只可標 `provisional`。不能因 seed0失敗而接著掃 P2、Canny、WST、bins、cell size、位置或多個 `mu`。

## 7. Phase E：部署等價與交付物

winner在進 BinaryQK 前必須 materialize 成沒有 auxiliary 的正式 Float checkpoint，並驗證：

1. input仍為 `[B,3,H,W]`；
2. 正式 layer 0–23 parameter names/shapes與 matched control一致；
3. state dict、ONNX／engine graph沒有 `hog_aux` 或 target generator；
4. Float／Bit-True output schema不變；
5. 正式 parameters、FLOPs與 batch-1 latency沒有因本方向增加；
6. 保存 source full-resume SHA、winner inference SHA、code/data/target config digest與完整八項 metrics。

若 auxiliary 無法被 exact strip，這個方法就不符合「training-only」契約，不能交給 Q0。

## 8. 執行順序摘要

```text
固定 person head與最終 FP graph
          ↓
建立同一 dormant-aux J0 parent
          ↓
工程不變量 → 短 probe
          ↓
F0/F1/F2 seed0
          ↓
tight gate通過才補 seeds
          ↓
strip auxiliary + 部署等價
          ↓
以 winner重開 BinaryQK Q0
```

返回[方向說明](<README.md>)或[優化方向索引](<../README.md>)。
