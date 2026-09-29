# 先讀這份：目前到底用哪個模型？

> 0914：下列原選用權重未被新候選取代。[最新完成的 MASF → BinaryQK → Rep 三組報告](<../update-0914/README.md>)請由此進入；新候選與本頁舊正式模型不同。

這份是使用者閱讀入口，不是歷史實驗目錄。以下「目前使用」指研究選定版本，不代表已在目標硬體部署，也不代表每個指標都最好。

## 1. 唯一的目前選用版本

**YOLO26M，共用特徵的 Detect＋Pose，保留 BinaryQK＋PWL＋P3 MASF，activation 使用 qSiLU。**

實際推論權重（本機保存：`../../experiments/activation/bridge_v1/artifacts/runs/qsilu_pq-short-e10-seed1-v1/inference/best_joint.pt`；本次未上傳）。SHA-256：`1bbbbe0f87cb853827af1bb625e7066178e3fb2c7e14c410d997efb24e94834a`。

`qSiLU E2` 意思是 qSiLU 訓練實驗中，第 2 回合被選中；這個實驗實際跑了 10 回合，不是模型從零只訓練兩回合。`best_joint` 只是該次實驗依多任務評分選出的檔案，不是歷史上所有模型、所有指標的冠軍。

| 技術 | 目前權重到底有沒有？ | 白話說明 |
| --- | --- | --- |
| BinaryQK | 有，兩個 Attention 位置 | Q、K 的相似度使用二值表示，不是整個模型都二值化 |
| Hadamard 第二條分支 | 有 | 除了原座標的二值比較，也在轉換座標後做二值比較 |
| 固定尺度與相對位置偏置 | 有 | 配合二值分數的尺度與位置資訊；不是每張圖選一個 scale |
| PWL Softmax | 有，範圍 [-10,0]、20 段 | 用分段近似處理 Attention 分數，不是原生 Softmax |
| P3 MASF | 有，只在 Detect P3 輸入前 | 不在共用 P3 節點，也不直接進 Pose |
| bridge | 訓練方法有保留 | 讓 one2one loss 可以直接訓練 MASF；不是額外推論 head |
| qSiLU | 有 | 以固定分段二次式近似 SiLU；不是可訓練 activation 係數 |
| HOG | 沒有採用 | 試過的訓練輔助監督；不是目前推論的一個 head |
| RepConv | 沒有採用 | 試過 layer17 的對照，但目前 layer17 仍是 Conv |
| P2 MASF | 沒有採用 | 歷史候選，不在目前權重裡 |
| KD 權重 | 沒有採用 | 後續候選未取代目前 qSiLU E2 |
| one2many＋外部 NMS | 沒有改成預設 | 目前預設是 one2one；該候選有取捨 |

上述架構已重新載入目前權重做 CPU 稽核，不只引用舊標題。原始證據見[前置檢查](<../../experiments/inference/component_switch_v1/artifacts/preflight.json>)。

## 2. 現在的架構圖

![目前實際使用的 Detect／Pose 分支](<figures/current-architecture.svg>)

```text
影像 → 共用 Backbone／Neck
          │
          ├─ layer10：Attention A（BinaryQK＋PWL）
          │
          └─ layer16：p3_raw ─┬─> layer17 → layer19：p4_raw
                             │                       │
                             │                       └─> layer20 → layer22：p5_raw
                             │                                      └─ 內含 Attention B
                             │                                         （BinaryQK＋PWL）
                             ├─> MASF → p3_det
                             └─> p3_raw 保留給 Pose

Detect 輸入：[p3_det, p4_raw, p5_raw] → COCO 80 類框／分類
Pose   輸入：[p3_raw, p4_raw, p5_raw] → BBAT ball/bat 框／分類／關鍵點

qSiLU：原本使用 SiLU 的位置已替換，包括 head。
圖中省略上採樣與 Concat 的旁接，精確 layer.from 見前置檢查 JSON。
```

**BBAT 的 box 分數來自 Pose 模型自己的框分支，不是把 COCO Detect head 當成二類 head。** COCO 與 BBAT 是兩個固定資料集，用同一套共用特徵、不同任務 head 評估。

## 3. MASF 的開關是什麼？

```text
開：p3_det = p3_raw + α × context(p3_raw)
關：α = 0 → p3_det = p3_raw
```

目前 α 為 `0.017398657277226448`。這不是「精度增加 1.74%」，而是特徵殘差的乘數。context 的大小也會影響實際貢獻。

bridge 的訓練梯度係數為 `0.012076444778011642`，與 α 是兩個不同的數；前者控制訓練梯度，後者影響推論特徵。

CPU 已驗證：同權重 α 設為 0，Detect 輸出與把 MASF 換成 Identity 旁路一致；Pose 輸出完全相同。原因是 Pose 根本沒有經過這個 MASF。這不代表 MASF 在別的位置或重新訓練後都無效。

α 設為 0 的原始程式仍計算 context，**不會自動省下該分支運算**。要省運算，還需明確使用旁路或移除模組；目前没有因此更換選定權重。

## 4. BinaryQK 裡面有哪些部分？

```text
Q、K ─┬─ sign(Q)、sign(K) ────────────────> 二值相似度 ─┐
      │                                                 ├─ 固定尺度加權
      └─ Hadamard 轉換 → sign → 二值相似度 ──────────────┘
                                                        ↓
                                                加相對位置偏置
                                                        ↓
                                             PWL Softmax [-10,0]
                                                        ↓
                                              權重與 V 加權混合
```

兩個 Attention 各有 4 個 head，每個 head 有兩條分支尺度，共 16 個固定 slots。讀取目前權重：第一處 8 個值均為 0.25；第二處第一個 head 為 [0.125,0.25]，其餘為 [0.25,0.25]。不是 16 選 1，也沒有每圖動態選尺度。FP-QK 分支不使用這些二值尺度。

先前修正的 STE／surrogate 是讓二值分數可以回傳訓練梯度。它修復了「學不到」的問題，但不能保證二值化遺失的幅度資訊都能補回來。

## 5. 新要求：不是關開關，而是移除後重訓

![已採用的 BinaryQK 與待重訓的 QK＋PWL](<figures/attention-options.svg>)

使用者於 2026-09-13 更正目標：移除 BinaryQK 與為它加入的補償設計，恢復原本 Attention 後重新訓練。這與只換成 FP-QK 再驗證不同。

```text
現在：Q/K → 原座標二值＋Hadamard 二值 → 固定尺度＋相對位置偏置 → PWL

預計：Q/K → 原生浮點 QᵀK / √d → PWL → 與 V 加權混合
                                  ↑
                      已確認保留 PWL [-10,0]、20 段

共同保留：Detect＋Pose 雙 head、qSiLU、目前的 P3 MASF 位置。
重訓後：再以同權重比較 MASF 開／關，決定是否需要無 MASF 配對重訓。
```

移除範圍是二值 sign/XNOR 路徑、Hadamard 第二分支、固定二值尺度及額外相對位置偏置。保留原本 Attention 的 V 路徑、pe 位置卷積與 proj；pe 與額外 relative bias 不是同一個模組。

Q/K/V 的三個分開投影可以依每個 head 的原通道順序回填為原生合併投影，不能直接把三個 weight 沿第一維串接。必須驗證 Conv、BN 與輸出對應後才能訓練。

訓練初始建議使用目前已訓練的權重做恢復訓練，重新建立 optimizer，不沿用舊動量；這不稱「全部隨機初始化從零訓練」。暫不退回原生 YOLO 而丟掉已適應的 Pose head。具體訓練回合與 learning rate 必須根據重建後 E0 完整基準及梯度檢查設定，不把舊 Attention 的極小 LR 直接套用。

原先同權重 FP-QK 測試保留相對位置偏置與 PWL，而且沒有重訓，因此不能拿其降分否定這項新實驗。

## 6. 目前實測精度怎麼讀？

四組完整的開關診斷直接列在下方第 8 節；其中兩組是這次新增驗證，沒有重新訓練。

以下是目前選用版本，不是尚未開始的新重訓模型。AP50–95 越高越好，數值乘 100 可讀為 AP 分數；不是單張圖顯示的 confidence。

| 評估項目 | AP50–95 |
| --- | ---: |
| COCO 80 類框 | 0.503885 |
| COCO person 框 | 0.625887 |
| BBAT 平均框 | 0.618008 |
| BBAT 平均關鍵點 | 0.891329 |
| ball 框 | 0.505192 |
| ball 關鍵點 | 0.859649 |
| bat 框 | 0.730823 |
| bat 關鍵點 | 0.923008 |

不能用 Pose AP 較高來推論框就一定準；框和關鍵點是不同評分。正式測試使用完整 COCO val5000 與 BBAT5 v1 val683，不改 split。

## 7. 名稱與資料夾怎麼看？

| 名稱 | 白話意思 |
| --- | --- |
| checkpoint／權重 | 模型學到的數值；訓練快照可能另含 optimizer |
| inference/best_joint.pt | 目前選中、供重建推論的 state dict；不是獨立通用 YOLO 檔 |
| Float／BitTrue | 這裡常指 PWL 的計算後端，不代表 QK 已改回浮點 |
| gate | 接受或淘汰候選的指標條件；不是 MASF 的 α |
| E8／E2 | 該次實驗第幾個 epoch，不是全研究總訓練回合 |
| control／對照 | 同預算、盡量只差一項技術的比較組 |
| artifacts | 每次實驗的原始產物，內含成功和失敗，不都是最終成果 |
| archives | 封存副本，避免遺失；不是要拿來日常推論的新模型 |
| proposals | 尚未落實或保留的想法，不代表已採用 |

舊報告是完整研究帳本，這份才是目前選用狀態的白話入口。需要所有成本與歷史數據，再看[完整報告](<../final/README.md>)及[九項效能比較](<../performance/README.md>)。

目前狀態：PWL 契約已確認。依使用者最新順序先完成報告與發布；新的 scale／bias 調整及 Attention 重訓尚未啟動，不宣稱已取得新版精度。

新增[已完成的同權重開關診斷](<switch-diagnostic.md>)與[移除後重新訓練計畫](<../../experiments/attention_recovery_v1/README.md>)。兩者不是同一個實驗。

## 8. 實際把 MASF／BinaryQK 開關後，結果如何？

為了方便閱讀，以下把 AP 乘 100。差 0.01 AP 相當於 1 個 AP 分數，不是「相對提升 1%」。下表四組都來自同一個目前選用權重；浮點 QK 只換 score，仍保留 bias／PWL，**不是移除補償分支並重訓**。

| AP 分數 | 目前：Binary＋MASF | Binary＋MASF 關 | 浮點 QK＋MASF | 浮點 QK＋MASF 關 |
| --- | ---: | ---: | ---: | ---: |
| COCO 框 | 50.3885 | 50.3860 | 49.1979 | 49.2094 |
| person 框 | 62.5887 | 62.5904 | 61.8481 | 61.8481 |
| BBAT 平均框 | 61.8008 | 61.8008 | 60.1333 | 60.1333 |
| BBAT 平均關鍵點 | 89.1329 | 89.1329 | 87.4567 | 87.4567 |
| ball 框 | 50.5192 | 50.5192 | 48.0950 | 48.0950 |
| ball 關鍵點 | 85.9649 | 85.9649 | 82.8336 | 82.8336 |
| bat 框 | 73.0823 | 73.0823 | 72.1715 | 72.1715 |
| bat 關鍵點 | 92.3008 | 92.3008 | 92.0798 | 92.0798 |

**直接答案：目前 MASF 開／關幾乎沒有差；直接切成浮點 QK 則下降。但移除後重新訓練尚未做，不能據此說 BinaryQK 一定比較好。** BBAT 六項不受 MASF 開關影響，因為目前 Pose 不走這個 MASF。

完整小數、來源與比較邊界見[診斷附件](<switch-diagnostic.md>)。這次兩個 MASF off 組皆跑完整 COCO 5,000 張與 BBAT 683 張；兩個 MASF on 組重用先前相同權重及評估方式的完整結果。

## 9. 其他每個階段到底做了什麼？

### HOG：訓練時教模型看邊緣，沒有採用到最後模型

HOG 把局部邊緣分成方向直方圖；實驗曾讓 P3 特徵多預測這個輔助目標，期望改善形狀資訊。所謂「原生 control」是只使用原來 detection loss 的對照，不是一個叫「HOG 原生訓練」的新演算法。HOG 沒有通過當時的精度要求，因此最後權重沒有該輔助 head，也沒有把它造成的訓練結果選為主線。

### RepConv：讓訓練多分支，部署折成一個卷積；本次沒選它

只針對 layer17 試過，不是全面更換。初始化與折疊等價有驗證，但精度未勝過配對要求，因此目前 layer17 是 Conv。不能把「理論可折疊」写成「最後已使用 RepConv 並加速」。

### MASF：只保留 Detect P3 bridge，不是所有 MASF 都保留

舊 shared MASF 放在共用節點，会改變後續 P4／P5；P2 MASF 是另一個更高解析度候選。兩者都不是現在的接線。最後依使用者選定 P3 bridge 繼續融合研究；這是研究路線選擇，不是已證明 MASF 有獨立增準。bridge 改的是訓練監督通路，推論仍只做一次 MASF。

### Combine：不是把兩個模型的框混在一起

這裡是共享 Backbone／Neck，一個 COCO Detect head、一個 BBAT Pose head。Pose 在融合前先做適應，融合後也有 Pose-head 恢復。兩個資料集分別計算自己的 loss 和 AP，不把 BBAT 二類 labels 塞進 COCO80 head。新線比較保護 COCO，但相較舊 combine 的 bat 仍有缺口，所以不是完全解決融合問題。

### qSiLU：最後有使用，但不代表 RTX 5090 跑得更快

qSiLU 用固定分段二次式近似 SiLU。兩個候選各訓練 10 回合，SiLU 選 E9，qSiLU 選 E2。qSiLU 的 joint 評分較高、ball 有收益，但 person／bat 有小幅取捨；其 Python／PyTorch 實作在目前 GPU 上更慢。這也是「硬體結構簡單」與「現有軟體速度快」不能混為一談的例子。

### KD：有做，不代表有採用其結果

KD 是用 teacher 的輸出或特徵監督 student。双任務／Pose-head 候選未全面通過接受條件；後來重組關鍵點分支也沒有帶來可接受替代，因此目前仍是 KD 前的 qSiLU E2。不能把資料夾中日期較新的 best_pose.pt 當成新版正式 best_joint.pt。

## 10. BinaryQK 到底做好了嗎？scale／bias 還能改嗎？

結論是：**部分實作有驗證，但沒有證明精度已優化到最好。** 本次 CPU 檢查确认固定 scale 真正生效，gamma 在固定模式改成 7 倍仍不影響分數。修復過的 surrogate 在合成資料上能給 Q/K 有限、非零梯度；這不是說後續每個訓練階段都開放了 Q/K 更新。

目前有 16 個固定分支尺度，會隨投影特徵改變而自動重新適應的動態校準沒有啟用。下一步值得先查「最後 checkpoint 的尺度是否仍適合」，再查尺度與既有相對位置 bias 的比例。兩項都可以設計成推論只讀固定值，不必每張圖重新估算。

可替代 BinaryQK 的第一優先是「原生 QK＋PWL 恢復訓練，再評估固定尺度 INT8 QK」；局部 window、K/V 降採樣與 linear attention 也可簡化不同成本，但會改變注意力連接或正規化語意，不先全面替換。詳見[scale／bias 稽核、公式、論文依據與實驗順序](<../../docs/research/2026-09-13-attention-scale-bias-alternatives.md>)。文中明確分開已查證與尚未實驗的部分。

## 11. 目前這一個模型的大小、速度與成本

只列 qSiLU E2，避免把不同候選的數字拼成一個不存在的最佳模型。B1、640×640、FP32、兩個 head 的模型核心 forward，CPU 為 4 threads；不包含讀圖、resize、H2D、外部 NMS。

| 指標 | 現行模型結果 | 白話與限制 |
| --- | ---: | --- |
| Params | 26,529,701 | 模型登錄參數數量，含保留但該推論路徑未執行的權重 |
| 推論權重檔大小 | 106.826 MB | 十進位；原檔 106,825,797 bytes，不是整個封存包大小 |
| MAC subtotal | 47.883 G | 已計入算子的乘加小計，不是所有算子成本 |
| 浮點 FLOPs subtotal | 95.766 G | 此表採 2 FLOPs/MAC；不把 bit operations 當免費浮點運算 |
| GPU peak allocated | 485.35 MiB | PyTorch tensor allocator，含常駐模型及輸入；不是整卡所有顯存 |
| CPU latency 中位數 | 657.25 ms/frame | Intel Core Ultra 9 285K、固定 4 threads |
| GPU latency 中位數 | 39.66 ms/frame | RTX 5090、CUDA events；不是完整應用端到端延遲 |
| target hardware latency | 未量測 | 目標板尚未指定／連接，不能用 5090 代填 |
| GPU energy/frame | 8.221 J/frame | NVML 整張 GPU 遙測，含 idle／其他活動，非整機能耗 |
| target energy/frame | 未量測 | 沒有板端能耗數據 |

原始浮點 SiLU 比較組 GPU 約 30.18 ms；qSiLU 約 39.66 ms，在這個環境沒有加速。這是不同訓練候選在同口徑的實測，不是板端結果。完整樣本、峰值 reserved、CPU RSS 與限制見[效能報告](<../performance/README.md>)及[CSV](<../performance/comparison.csv>)。沒有為本次報告重跑已成功的量測。

## 12. 「包體」、標題與實驗代號到底代表什麼？

如果你說的是包體：archives 的約 113.66 GB 是許多階段結果、訓練快照與原始記錄的封存副本，不是部署需要載入 113.66 GB。日常使用的是上面約 106.826 MB 的選定推論檔，加上相符的客製程式／設定。GitHub 是報告與研究程式發行，不含這次的權重與資料集；只有 GitHub 文字不能憑空重建已訓練數值。

如果你說的是標題：以下把研究代號翻成實際意義。

| 舊名稱 | 正確讀法 |
| --- | --- |
| full35-B100 | 本研究融合前來源模型的歷史代號；不是新的精度分數 |
| P3 bridge MASF E8 | 融合前 MASF 訓練支線第 8 回合選定的 Detect 權重，不是目前整個雙 head 模型 |
| 原正式 Pose checkpoint 的 MASF 推論消融 | 對舊獨立 Pose 權重開關其 MASF；不是對新 P3 bridge Pose 的重訓 |
| Head 對照 E8 | 該配對的 head 訓練對照組；需看同 parent／預算，不是通用模型名稱 |
| J0／J1／J2／J3 | combine 不同訓練阶段代號；要搭配當次設定，不能只按號碼推定變好 |
| activation-relative best | 這個 activation 實驗裡選出的候選，不表示最初獨立模型的全部指標都通過 |
| strict gate 未全過 | 預先要求的一部分精度條件尚未達標，不是執行報錯 |
| FP-QK teacher candidate | 曾把學生 QK 改浮點的未重訓候選；不是已訓練好的浮點 teacher |
| smoke／preflight | 小型執行或結構檢查，不是完整訓練成果 |

請依序看本頁第 1 節（用什麼）、第 2–5 節（怎麼接）、第 8 節（實測差異）、第 10 節（接著查什麼）；只有需要追來源時才進 experiments／artifacts。標題不再代表「有做過就有採用」。
