# Pose MASF 專項重訓：完整架構、梯度推導與最小實驗

狀態：**B 組 5 epoch、獨立驗證、alpha-off 與 CPU 稽核均已完成；A 組取消。** 最新結論見 [RESULTS.md](<RESULTS.md>)，本輪後續見 [0914 更新](<../../reports/update-0914/README.md>)。提案參數與已執行設定分開保留。

## 1. 結論與範圍

建議先固定共享 backbone／neck、兩處 Attention、Detect head 與 Detect MASF，只用 BBAT5 v1 訓練 Pose head 及 Pose 自己的 P3 MASF。第一輪不改 P2、不增加 head、不換 activation、不恢復原 Attention 續訓。

原先提出 A/B 配對，使用者隨後取消 A 組加訓。目前只核准 B 組 5 epoch、warmup 1：

| 組別 | 可訓練部分 | 想回答的問題 |
| --- | --- | --- |
| A：已取消，不執行 | 原 Pose head 加訓 | 缺少此對照，無法分離單純加訓收益 |
| B：5 epoch 與完整驗證已完成 | Pose head＋獨立 Pose P3 MASF | 相較固定 E2，實際框／關鍵點 AP 是否改善？ |

A 不是已取消的 BinaryQK 對照，也不是重跑整個 COCO 訓練。原提案的兩組起點均是同一份 **native QK E2 融合模型**：原生 QK＋PWL [-10,0] 20 段、qSiLU、既有 Detect P3 bridge MASF。固定來源 SHA：`4257ca9c471736aa97d0077097526b7863a9268b773913119b9969991d00b0f3`。

[先前結果](<../pose_masf_priority_v1/RESULTS.md>)只是 Detect MASF 直接移接到 Pose：overall Pose AP 幾乎不變，ball 小升、bat 小降，尚未對新分支共同訓練。因此這個重訓計畫針對的是「讓 Pose 學會使用 MASF」，不是重複那次推論開關。

## 2. 完整架構圖

[可放大 SVG](<figures/full-architecture.svg>)／[PNG](<figures/full-architecture.png>)／[可重建 DOT](<figures/full-architecture.dot>)。所有尺度取自本次 CPU forward；完整表在 [architecture.json](<architecture.json>)。

![完整架構](<figures/full-architecture.png>)

藍色全部固定；橘色是 B 組可訓練部分。虛線表示原 E2 的直通接法，只作參考；不執行 A 組加訓。圖中的 `2×(x,y,v)` 是權重契約 `[2,3]` 的兩個關鍵點槽，不是所有標註都必須有兩個可見點。

### 2.1 Backbone 與 Neck 的實際路徑

以下尺寸省略 batch，採 `C×H×W`：

```text
Input：3×640×640
  ↓ layer 0  Conv       →  64×320×320
  ↓ layer 1  Conv       → 128×160×160
  ↓ layer 2  C3k2      → 256×160×160  (P2，不接 head)
  ↓ layer 3  Conv       → 256×80×80
  ↓ layer 4  C3k2      → 512×80×80   ──────────────┐ skip 到 layer 15
  ↓ layer 5  Conv       → 512×40×40               │
  ↓ layer 6  C3k2      → 512×40×40   ─────────┐   │ skip 到 layer 12
  ↓ layer 7  Conv       → 512×20×20           │   │
  ↓ layer 8  C3k2      → 512×20×20           │   │
  ↓ layer 9  SPPF       → 512×20×20           │   │
  ↓ layer10  C2PSA      → 512×20×20           │   │  Attention ①
           │                                 │   │
           ├─ layer11 Upsample ×2             │   │
           │   ↓ layer12 Concat(11,6) <───────┘   │ → 1024×40×40
           │   ↓ layer13 C3k2                     │ → 512×40×40
           │   │   ↓ layer14 Upsample ×2          │
           │   │   ↓ layer15 Concat(14,4) <────────┘ → 1024×80×80
           │   │   ↓ layer16 C3k2 → p3_raw = 256×80×80
           │   │   │   ↓ layer17 Conv /2
           │   │   │   ↓ layer18 Concat(17,13) <─ layer13 → 768×40×40
           │   │   │   ↓ layer19 C3k2 → p4_raw = 512×40×40
           │   │   │   │   ↓ layer20 Conv /2
           │   │   │   │   ↓ layer21 Concat(20,10) <─ layer10 → 1024×20×20
           │   │   │   │   ↓ layer22 C3k2 → p5_raw = 512×20×20  (Attention ②)
           │   │   │   │
           └── layer10 同時提供 layer11 與 layer21；不因 Pose MASF 改動。
```

兩處 Attention 實際位置是 `graph.model.10.m.0.attn` 與 `graph.model.22.m.0.1.attn`。這裡沒有 BinaryQK；qSiLU、原生 QK 與 PWL 保持固定。

### 2.2 layer 23 的分流：最重要的差異

```text
layer16：p3_raw ──┬─> layer17 → layer19 p4_raw → layer20 → layer22 p5_raw
                 │
                 ├─> Detect MASF（固定）→ p3_det ──────────┐
layer19：p4_raw ───────────────────────────────────────────┤
layer22：p5_raw ───────────────────────────────────────────┤
                                                          ↓
                              Detect([p3_det, p4_raw, p5_raw])  固定

layer16：p3_raw ──┬─> 原 E2：直通 ──────────────────────────┐
                 └─> B 組：Pose MASF（可訓練）→ p3_pose ──┤  接法比較
layer19：p4_raw ───────────────────────────────────────────┤
layer22：p5_raw ───────────────────────────────────────────┤
                                                          ↓
                  Pose([p3_raw 或 p3_pose, p4_raw, p5_raw])  可訓練
                            └─ ball／bat 框＋類別＋關鍵點
```

MASF 的位置是 **Neck 的 layer16 輸出之後、Pose head 消費 P3 之前**，不是 backbone 的 layer4，也不是 layer17 下採樣之前的共享節點。Pose 不會因此改到 P4／P5 的特徵生成路徑。

## 3. 固定共享特徵後，為何比較更乾淨？

令共享網路為 `Bψ`，產生 `(x3,x4,x5)`；Detect 使用 `Dω` 與原 MASF `FD`；Pose 使用 `Pφ` 與新 MASF `Fθ`。

```text
(x3,x4,x5) = stop_gradient(Bψ(image))
detect_out = Dω(x3 + αD FD(x3), x4, x5)              固定
pose_out   = Pφ(x3 + α Fθ(x3), x4, x5)              B 組
```

訓練只更新 `(φ,θ,α)`。共享參數 ψ、Detect 參數 ω、Detect MASF 都不放入 optimizer；共享 BN 的 affine 與 running statistics 一起固定，Detect 同樣固定。**只設 requires_grad=False 不夠**，若 BN 仍在 train mode 更新 running mean／variance，COCO 仍可能漂移。

因此在相同 checkpoint、推論設定與輸入下，Detect 的數值應保持不變。每 epoch 仍做 COCO 全量驗證與凍結 state guard；一旦差異超過數值重現容差，先查 BN／EMA／參數共用，而不是把它當作容許的精度犧牲。

這一輪只能測試「既有共享特徵上，Pose head 與 MASF 的適應能力」。若真正瓶頸在共享 backbone，本輪可能碰到上限；那需要另一個解凍研究，不能混入這次 A/B 對照。

## 4. MASF 的前向與參數梯度

目前模組是：

```text
Fθ(x) = Project1×1(DW3×3(x) + DW5×5(x))
y     = x + α Fθ(x)
```

每個 Conv 包含目前模型的 BN／qSiLU；不是只有線性卷積。α 是一個所有影像共用的可訓練純量，不是逐張圖片重算的 scale。學完可保存為固定常數；但不能未經驗證就把 α 折進 qSiLU 前的卷積，因為非線性一般不滿足 `α·act(z)=act(αz)`。β 則只在訓練反向使用，不增加部署運算。

令 `g = ∂L/∂y`，則鏈式法則給出：

```text
∂L/∂α = Fθ(x)ᵀ g
∇θ L  = α · JF,θᵀ g
```

第一式讓 α 學習「這份上下文應該加多少，甚至是否反向補償」。第二式讓 context 卷積學習「Pose 真正需要什麼補充特徵」。不能只訓 α、永遠凍結 context，然後宣稱測完完整 MASF 的訓練潛力。

### 4.1 為什麼提案改成 α=0 起步？

B 組複製已有 Detect MASF 的 context，但把新 Pose α 設為 0；Pose head 沿用固定 E2 參數。初始時 `y=x`，所以 B 與原 E2 的初始輸出相同，避免把前一次直接移接的初始偏移混入訓練差異。

但有一個必要注意事項：α=0 時，context 的**任務梯度**為零，α 的梯度則不一定為零。先確認真實 loss 能更新 α，再確認後續 context 梯度出現。若使用 AdamW，weight decay 仍可能影響零梯度參數，不能把「任務梯度為零」說成「參數必然完全不動」。α 本身不做 weight decay。

為避免 gate 在 AMP 下完全不啟動，正式 smoke 必須核對 α 的 FP32 master 參數、unscaled 梯度與實際更新；不能只看整體 loss 下降。此檢查已在 B 組真實 GPU smoke 完成。

## 5. one2many／one2one：為什麼 bridge 需要重新考慮？

[梯度 SVG](<figures/training-gradient.svg>)／[PNG](<figures/training-gradient.png>)：

![訓練梯度圖](<figures/training-gradient.png>)

### 5.1 直接外掛的斷點

如果先算 `y=MASF(x)`，再用原生 `head_one2one(y.detach())`，那麼 one2one 的 loss 不能回到 MASF；MASF 只由 one2many 分支監督。這不等於整個 MASF 完全沒有梯度，但部署分支沒有直接教它。

### 5.2 現有 bridge 的做法

```text
ym = x + α Fθ(x)
yo = GSβ(stop_gradient(x) + α Fθ(stop_gradient(x)))

Lm = PoseLoss(head_many(ym,x4,x5), targets)
Lo = PoseLoss(head_one(yo,stop_gradient(x4),stop_gradient(x5)), targets)
L  = λm Lm + λo Lo
```

`GSβ` 前向是 identity，反向才乘 β。兩次 MASF forward 使用**同一組 θ、α**，不是建立兩套新模組；MASF BN running statistics 固定。推論只執行一次 MASF，沒有這個雙分支訓練成本。

令 `gm=∂Lm/∂ym`、`go=∂Lo/∂yo`，則回到 MASF 的有效梯度是：

```text
geff = λm gm + β λo go
∂L/∂α = Fθ(x)ᵀ geff
∇θ L  = α · JF,θᵀ geff
```

但 Pose one2one head 自己的梯度仍是 `λo ∇φo Lo`，**不乘 β**。把 β 乘到整個 Lo 上是另一種訓練方法，不能當作等價實作。

### 5.3 舊 β≈0.0121 為什麼可能太保守？

原候選沿用 Detect bridge 的 `β=0.012076444778011642`。現有原生 `E2ELoss` 的 one2many 權重從 0.8 降到 0.1，one2one 從 0.2 升到 0.9。若重建新的 5 epoch criterion，並在每回合結束 update 一次：

| 回合 | λm | λo | 舊 β×λo：傳回 MASF 的 one2one 係數 | 提案 β=1 時 |
| --- | ---: | ---: | ---: | ---: |
| E1 | 0.800 | 0.200 | 0.002415 | 0.200 |
| E2 | 0.625 | 0.375 | 0.004529 | 0.375 |
| E3 | 0.450 | 0.550 | 0.006642 | 0.550 |
| E4 | 0.275 | 0.725 | 0.008755 | 0.725 |
| E5 | 0.100 | 0.900 | 0.010869 | 0.900 |

即使 E5 的 one2one loss 權重已有 0.9，它傳到 MASF 時只剩約 0.0109；同時 one2many 仍是 0.1。不能因此就說哪一個實際梯度一定較大，因為 gm、go 的範數與方向也重要，但能確定 one2one 通道被額外壓縮了。

### 5.4 新提案：固定 trunk 的前提下，β 先設 1

共享網路已凍結，one2one 的 x 也已 detach，再用很小 β 保護共享特徵已不是必要條件。因此提案先用 β=1，讓 Pose 損失本身的 λm／λo 決定兩條通道；穩定性用小 LR、warmup、梯度 clipping 與真實 loss smoke 驗證。

這是**有理由的初始設定，不是已證明最佳的超參數**。若真實 unscaled 梯度或更新不穩定，先降低新分支 LR；需要重新校準 β 時，只根據 train 梯度，不在 val 上挑 β。β=1 是舊 one2one 梯度係數的約 82.8 倍，但不代表總梯度、AdamW 更新或精度增加 82.8 倍。

**實作邊界：目前封存的 `PoseP3BridgeMASF` 仍 assert β≤0.25。** B 組已在獨立 training_b.py 實作 β=1 並完成 CPU／GPU 驗證；原移接類別限制不变，不可只改 JSON 就宣稱生效，也不能悄悄修改已完成移接實驗的原程式。舊 β 不影響已完成的推論結果。

## 6. Pose 專項 loss 不只是關鍵點 loss

沿用目前 `NativeTaskLossRouter → E2ELoss(PoseLoss26)`，不先新增 HOG、KD 或額外正則：

```text
Lbranch = 7.5 Lbox + 0.5 Lcls + 1.5 Lreg
        + 12.0 Lkeypoint + 1.0 Lvisibility + 1.0 LRLE
Lpose   = λm Lbranch,many + λo Lbranch,one
```

這些是現有原生 gain 的表示；程式內部已套用的權重不可再重複乘一次。RLE 在 flow／sigma 路徑有效時才有值，不能假設每個 batch 都非零。B 組訓練 Pose head 部分，包括有梯度的關鍵點、框／類別、sigma／flow 與 BN affine。

權重契約 `reg_max=1`。本機的 `BboxLoss` 在無 DFL bins 時，沿用同一個 `loss_dfl` 欄位計算正規化 LTRB 的 L1 loss；因此上式寫 `Lreg`，不把它誤說成離散 DFL，也不把這一項當成零。

### 6.1 有效 batch 128 與 loss 正規化

先採 physical batch 16、累積 8 次；每個 epoch 完整走過 5,964 張：46 個滿載 macro＋1 個 76 張的尾端 macro，合計 47 次 optimizer 更新，不丟掉尾批。5 epoch 是每組 235 次更新、29,820 個訓練影像呈現。

令第 i 個 microbatch 的原生 `raw_total_i` 已包含實際 batch size，macro 總圖數為 B。沿用 reference batch 64：

```text
Lbackward = (64/B) × Σi raw_total_i
```

尾端要用 B=76，不能仍除以 128。這是現有 loss 正規化尺度的延續，不是把梯度又平均兩次。Pose-only 時 task weight 會與 weight_sum 抵消，不能以為保留舊 pose_weight=0.25 就仍有 0.25 倍監督。

**新 [training_b.py](<training_b.py>) 已實作 `batches_of(loader, 8)`，明確把 8 個 microbatch 送給 MacroStepEngine。** CPU 已核對 373 個 microbatch → 47 次更新、尾端 76 張；已以真實 BBAT GPU epoch 完成 5 回合，每回合 47 次更新。沒有修改舊 PoseEpochRunner。

「有效 batch 128」不等於一次將 128 張送進 GPU，也不保證與原生 physical batch 128 逐位等價；原生 target-score 的 batch 內正規化、實際 augmentation 等仍有差異。本次不跑 A 組，不能宣稱已有配對公平性證據。此專項每回合只訓 5,964 張而非 COCO 的 118,287 張，若回合變快是範圍不同，不是省略標註。

## 7. B 組已執行設定：GPU 5 epoch 完成

| 項目 | 第一輪提案 |
| --- | --- |
| 初始化 | 相同 E2 權重；B 的 context 複製、α=0 |
| 可訓練 | Pose head；B 再加 Pose MASF |
| 固定 | 整個共享網路／Attention、Detect／Detect MASF、全部 BN running statistics |
| Pose／MASF BN affine | 可訓練；不做 weight decay |
| 回合 | 僅 B 組 5，warmup 1；不啟用 patience，B5 對固定 E2 |
| Optimizer | AdamW，betas=(0.948,0.999)，WD=0.00027 |
| Peak LR | Pose head 5e-6；MASF context 1e-5；α 1e-4 |
| 不做 WD | α、bias、BN affine |
| LR schedule | warmup 起始 0.1 倍；cosine 最終 0.5 倍 |
| batch | physical 16 × accumulation 8；尾端依實際張數 |
| AMP／clipping | AMP；先 unscale 再 clip norm 10；含必要 FP32 數值路徑 |
| Pose bridge β | 新類別已實作 1.0，CPU 通過；真實 GPU smoke 待執行 |
| E2E criterion | fresh 5-epoch criterion，原生 0.8/0.2 → 0.1/0.9；不是原 E3 resume |
| 輸入／seed | 640、seed 1；沿用 canonical Pose augmentation |
| Augmentation | 沿用目前 canonical Pose runner，mosaic=0、fliplr=0；不另加搜尋 |

先用 AdamW 是為了少增加一個新變因，不是已證明它勝過 MuSGD。這 5 回合不安排 optimizer 切換；若有收益，再另設同口徑長訓或第二 seed。

## 8. 驗證流程與決策

1. **CPU 已通過**：固定 E2／B α=0 的 Detect 與 Pose 輸出精確相同；Float／BitTrue 重建、β=1、真實 Pose head 的合成 one2one 梯度、optimizer 分組、BN／EMA 固定及完整續訓快照 roundtrip 通過。
2. **等待 GPU 後先 smoke**：使用 canonical train loader 的正常批次做兩次有效 batch 128 更新。檢查真實 loss／AMP、α→context 啟動與固定 state；smoke 不混入正式 epoch。此步已完成。
3. **只跑 B 組**：完整 5 epoch、每回合 47 macro；每回合核對固定參數、全部 BN 與 EMA。共享網路及 Detect 不訓練，A 不執行。
4. **每回合 BitTrue 驗證**：完整 COCO val 5,000 與 BBAT5 v1 val 683；分列 COCO overall／person、BBAT overall／ball／bat 的框與關鍵點 AP。
5. **B5 對固定 E2**：回答新方案是否改善。沒有 A5，不能把差值全部歸因於 MASF；best Pose 回合另列，不取代事前固定 E5 比較。
6. **B5 再關 α、另測 Float**：α=0 只測已訓模型的分支依賴，不等於無 MASF 重訓。Float 與 BitTrue 數值分開保存。
7. **保存**：每回合保存不可覆寫的完整 optimizer／scheduler／scaler／criterion／RNG／EMA 續訓檔及推論 state。驗證失敗可從已保存邊界補驗，不覆寫原 E2 或移接候選。推論成本本次不另排新 benchmark，先前同結構量測僅作參考。

事前工程參考門檻改以固定 E2 為基準：COCO 指標差異 ≤1e-8；B5 overall Pose AP 增加至少 0.002，其餘 BBAT AP 不下降超過 0.001。它不是統計顯著性，B5 已通過該工程門檻，但不是統計顯著性證明；不自動升版、不安排 A 加訓或延長 B 回合。

## 9. 推導已驗證到哪裡？

[check_derivation.py](<check_derivation.py>) 用 CPU float64 小型線性 context＋兩個 head，驗證 5 項：context chain rule、α chain rule、β 不縮放 head 自身梯度、α=0 的 context 任務梯度為零，以及 α 更新後 context 任務梯度能出現。[結果 JSON](<derivation-check.json>) 全部通過。

另完成 [preflight_b.py](<preflight_b.py>) 的實際 YOLO head CPU 檢查，結果見 [CPU 摘要](<artifacts/cpu-preflight-v1/summary.json>)：首步 α 梯度非零、context 任務梯度為 0，第二步 context 最大梯度 0.00666454；固定 state 與完整快照還原通過。總參數 26,604,466，B 組可訓練參數 4,632,441。這是合成目標，不是 BBAT 真實 loss 或精度結果；AMP、吞吐量、GPU 記憶體與收斂尚待 smoke／正式訓練。

## 10. 原始碼依據與圖檔重建

- [已完成移接候選的接法](<../pose_masf_priority_v1/pose_candidate.py>)：目前 β 與範圍限制。
- [PoseLoss26／E2ELoss／BboxLoss](<../../../yolo_p2/ultralytics/utils/loss.py>)：實際 loss gain、DFL-free L1 與兩分支權重。
- [任務 loss 正規化](<../../../yolo_combine/src/yolo_combine/joint_loss.py>)：reference batch 與 Pose-only 權重。
- [epoch runner](<../../../yolo_combine/src/yolo_combine/_joint_trainer_impl.py>)：每回合結束才 advance criterion。

```bash
dot -Tsvg figures/full-architecture.dot -o figures/full-architecture.svg
dot -Tsvg figures/training-gradient.dot -o figures/training-gradient.svg
```

0913 發布時先有圖、分析與 CPU 檢查；後續 GPU 訓練與驗證已完成，本次 0914 發布補齊結果。


## 11. 程式與佇列入口

- [training_b.py](<training_b.py>)：獨立 beta=1 Pose 類別、累積梯度、Pose-only EMA、5 epoch 快照與全量驗證／alpha-off 分析。
- [preflight_b.py](<preflight_b.py>)：已完成的 CPU 契約檢查；測試快照只留本機，不作訓練起點。
- [queue_b.py](<queue_b.py>)：GPU 0 沒有其他 compute 程序且取得共享 lock 後，依序 smoke → B5 → 分析。背景程式每 600 秒檢查，正常時不讀 log／不输出進度；不終止外部工作。ERROR／STALLED 會記錄事件，需要主代理介入，不宣稱背景程式本身具有模型診斷能力。

佇列已完成，不因讀取本文而重啟。歷史操作入口：`/home/uxin/yolo/.venv/bin/python -B queue_b.py --execute`。原 Attention pause-request 保留，沒有自動解除。Git 只發布程式、圖、設定與數值證據；權重、CPU 測試快照、cache、資料集及日誌保留本機。


## 最新完整 B 組排程

[queue-plan.json](<queue-plan.json>) 定義四階段：`smoke → train B5 → analyze → finalize CPU`，不恢復 Attention 佇列。已補強驗證失敗的恢復與 RNG 隔離，CPU 測試見 [queue-preflight-v1.json](<artifacts/queue-preflight-v1.json>)。GPU 完成後自動產出 RESULTS.md、comparison.csv、training-curves.csv／圖與五回合 final-audit.json；不自動採用新模型。

前述「尚未啟動佇列」是 Git 發布當下狀態，現由這次明確排程授權更新；即時 JOB_STARTED／JOB_DONE／ERROR／ALL_DONE 以本機 events.jsonl 為準。工作紀錄見[完整流程與驗收限制](<../../docs/worklogs/2026-09-13-pose-masf-b-queue.md>)。

## B 組佇列完成

5 epoch／獨立重驗／alpha-off／CPU 稽核已完成，現況以 [最終分析](<RESULTS.md>)為準；原發布狀態詳見工作紀錄。A 未跑，沒有自動升版。
