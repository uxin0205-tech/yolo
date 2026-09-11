# Head 對照／P3 bridge：名稱、架構與運算量

本說明只針對 optimize 新方向的 Detect 模型。比較中的「一開始」指使用者指定的 **Full35-B100、layer16 shared P3 MASF**，不是尚未加入 BinaryQK／attention 的原生 YOLO26M。數值為 640×640、batch 1；運算量為依模組公式計算，不是硬體實測。

## 原正式 Pose checkpoint 的 MASF 推論消融

這是舊專案 `yolo_combine/variants/full35/artifacts/pose/p0-full35-p3-b32a4-e100max-seed0/weights/best.pt`，不是新方向 P2／P3 checkpoint。保持權重與資料不變，僅在記憶體將 `model.16.p3_masf.alpha` 從 0.1927490234375 改成 0，再量測 BBAT5 的 box／keypoint AP，觀察目前已訓練模型對該模組的依賴。

MASF 形式為 `y = x + alpha * context(x)`。alpha=0 時輸出等於 x，但原 forward 仍會計算 context，因此**這個開／關消融不節省卷積計算**；只有真正移除或旁路模組的實作才省掉運算。它也不等於無 MASF 重新訓練的 control。這段僅保留歷史旁證，不用於新方向採用決策。

## Head 對照 E8

權重：`studies/pre-fusion-full35-b100/artifacts/masf-head-control-v1/epoch-08-resume.pt` 的 EMA。

它是無 MASF 的完整 Detect 模型，不是只有一個 head 的權重，也不是新增了一個 head。Head 指本階段只更新原有 Detect 中接 P3 的 box／class 分支（含 one-to-many 與 one-to-one），其他參數與 BN 統計固定。沒有新增 Pose、HOG 或 RepConv 模組。

E8 是本輪 MASF 實驗累計第 8 個 epoch：先完成 control E1–E5，再續訓至 E8；不是整個模型只訓練過 8 epoch。起點已經過更早的預訓練與方向 1 恢復訓練。續訓時 P3 head 基礎 LR 從 2e-6 調成 1e-5，沿用 optimizer／EMA，沒有重新 warmup。

```text
layer16：p3_raw ────────┬─> layer17 → P4 → layer20 → P5
                       │
                       └─────────────────────────────┐
layer19：p4_raw ──────────────────────────────────────┤
layer22：p5_raw ──────────────────────────────────────┤
                                                     ↓
                                  Detect([p3_raw, p4_raw, p5_raw])
```

## 一開始的 Full35-B100：shared P3 MASF

```text
layer16：p3_raw → MASF → p3_shared ─┬─> layer17 → P4 → layer20 → P5
                                  │
                                  └─────────────────────────────┐
layer19：p4_after_masf ───────────────────────────────────────────┤
layer22：p5_after_masf ───────────────────────────────────────────┤
                                                                ↓
                              Detect([p3_shared, p4_after_masf, p5_after_masf])
```

MASF 位於 P3 的共用節點，除了供 P3 預測，也會影響下游 P4／P5 特徵。這不會自動增加三倍 MASF 計算：模組只執行一次，之後是下游使用它的輸出。

## P3 bridge MASF E8：推論架構

權重：`studies/pre-fusion-full35-b100/artifacts/masf-task-bridge-v1/epoch-08-resume.pt` 的 EMA。

```text
layer16：p3_raw ────────┬─> layer17 → P4 → layer20 → P5
                       │
                       └─> MASF → p3_det ────────────┐
layer19：p4_raw ──────────────────────────────────────┤
layer22：p5_raw ──────────────────────────────────────┤
                                                     ↓
                                  Detect([p3_det, p4_raw, p5_raw])
```

MASF 移到 `model.23.p3_masf`，只作用於 Detect 的 P3 輸入，不回灌產生 P4／P5 的路徑。仍是 P3／P4／P5 三個尺度，沒有新增 P2 Detect head，亦沒有 Pose head。BinaryQK、attention 與固定 PWL [-10,0] 沿用本研究設定。

這是從相同大起點分支重訓後的候選，不是僅把原 B100 的已訓練 MASF 原封不動搬過去就宣稱等效。該分支最初重用 B100 context 初始化、alpha 歸零；E5 之後續訓加入 bridge。E8 EMA alpha 為 0.017174629494547844。

## Bridge 改的是訓練梯度

原生 Detect 的 one-to-one 路徑先 detach 已增強特徵，導致該分支 loss 無法直接更新 MASF；MASF 仍可從 one-to-many 分支取得梯度，並非完全沒梯度。

bridge 實作改成先 detach 原始 P3，再執行同一個 MASF，讓 one-to-one loss 可以回傳至 MASF，但不經這條新路徑回傳 Backbone。

```text
僅訓練：
p3_raw ─────────────> MASF ──────────────────> one-to-many loss
       └─> detach ─> 同一個 MASF ─> 梯度縮放 ─> one-to-one loss

推論：
p3_raw ─────────────> 單次 MASF ──────────────> 原生 Detect 推論
```

梯度縮放 forward 為恆等，backward 才乘固定 λ=0.012076444778011642。這不是 BinaryQK 的 scale，也不是 PWL 係數；推論不需要計算 λ。兩次訓練呼叫共用參數，不增加第二套 MASF 權重。BN 統計固定，避免重算造成額外 BN 更新。

## 運算量差多少

本實作 P3 是 80×80×256；MASF 的 context 包含 DW3×3、DW5×5、1×1 channel mixing，另有 BN／SiLU、相加及 alpha residual。

```text
Conv MAC = H×W×(9C + 25C + C²)
         = 80×80×(9×256 + 25×256 + 256²)
         = 475,136,000 MAC／image
         = 0.475136 GMAC／image

若 1 MAC 計為 2 FLOPs：0.950272 GFLOPs／image。

未融合 BN 的參數 = 9C + 25C + C² + 3×2C + 1
                = 75,777。
```

1×1 卷積佔 419,430,400 MAC，約為 MASF 卷積 MAC 的 88.28%；主要成本不是兩個 depthwise 分支。

| 推論方案 | 相對無 MASF 增加參數 | 相對無 MASF 增加 Conv MAC | 相對原 shared P3 MASF 的 Conv MAC 變化 |
| --- | ---: | ---: | ---: |
| Head 對照 E8 | 0 | 0 | -0.475136 G |
| 原 Full35-B100 shared P3 | 75,777 | +0.475136 G | 0 |
| P3 Detect-only（不加 bridge） | 75,777 | +0.475136 G | 0 |
| P3 bridge E8 | 75,777 | +0.475136 G | 0 |
| P2 MASF，160×160×256 | 75,777 | +1.900544 G | +1.425408 G |

因此，bridge 相對 Head 對照是多一個 MASF；相對原 shared P3 MASF，**推論卷積運算量不因換位或 bridge 增加**。P2 因空間面積為 P3 的 4 倍，MASF 本身的卷積 MAC 為 4 倍，不是整個模型 4 倍。

以上不含 BN／SiLU、elementwise、memory traffic、XNOR／popcount／PWL 的異質硬體成本，也不是整網總 FLOPs。P3 額外 elementwise 包含 3,276,800 次相加與 1,638,400 次 alpha 乘法；若做 BN folding 或其他部署優化，需另依實作計算。**不能將 MAC 相同說成硬體延遲必然相同。**

訓練時 bridge 每張圖片比原生 P3 MASF 多一次 MASF forward，即多 0.475136 G Conv MAC，另有對應 backward／梯度縮放成本；沒有多一次 Backbone forward。這不是訓練總成本或訓練時間的百分比。

## 比較口徑補充

若「原先」指最初 Full35-B100，同口徑 COCO overall／person 為 0.5035890014／0.6241112368；bridge E8 的 0.5082119552／0.6276641271 分別高 0.462295／0.355289 個百分點。但這包含先前恢復与微調，不能全歸因 bridge。原方案完整設定與三種基準差值見[基準核對紀錄](<../../docs/worklogs/2026-09-10-bridge-baseline-comparison.md>)。

Head 對照 vs P3 bridge 比較的是「無 MASF」對「P3 MASF＋bridge」整體方案，不能單獨證明 bridge 的作用。隔離 bridge 的直接對照應是 `masf-head-fork-v1`，兩者從同一 native fork E5 續訓，只有梯度橋接不同。

| E8 EMA | COCO overall | COCO person |
| --- | ---: | ---: |
| 無 MASF Head 對照 | 0.5082670924 | 0.6276994546 |
| 原生 P3 MASF，無 bridge | 0.5081607839 | 0.6275420268 |
| P3 MASF＋bridge | 0.5082119552 | 0.6276641271 |

bridge 相較原生 P3 MASF 有很小改善，但這兩項仍低於無 MASF Head 對照；不能因為 bridge 技術上成立就稱已取得足夠增準。

依據：`scripts/masf_p3.py`、`masf_task_bridge.py`、`continue_masf_head.py`、`train_masf_task_bridge.py` 與各 run 的 summary；相对路徑以 `studies/pre-fusion-full35-b100/` 為根。MASF 模組原始碼為 final/code/achitechure_1/masf.py。本次只核對程式、既有數據與 CPU 算術，未載入不安全 checkpoint、未執行 GPU 或更改訓練狀態。
