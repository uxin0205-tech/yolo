# OPT-P3-HOG-COMPANION-TRAINING：YOLO26M 訓練期結構先驗

> 2026-09-08 本文 `F2-PRE-HOG9` 是歷史／另案方案；本輪依[方向1 master plan](<../integrated-roadmap/direction1-master-plan.md>)使用 `W-HOG10`／原生 raw P3，僅在 PSEL 基準可靠且單因子 gate 通過後評估，不列基準修復前必跑，不與歷史名稱／結果混用。`training_ready=false`，GPU 0。詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

| 欄位 | 內容 |
|---|---|
| 狀態 | `proposed`；尚未訓練 |
| 與論文的關係 | 保留第 3.1 節的 filter-assisted training 思想，不照搬 SAR 的輸入拼接 |
| 首選方法 | P3 `Box-Aware HOG9 Companion Supervision` |
| 推論圖 | 不變；正式輸入仍為 3-channel RGB，auxiliary head 必須在 export 前移除 |
| 最小完整消融 | `F0-MATCH / F1-LUMA9 / F2-HOG9` |
| 全域位置 | 最終 FP head／架構固定後、BinaryQK QAT／PTQ 前；與 conflict-safe 為獨立候選，不在首輪疊加 |
| 完整研究 | [論文第 3.1 節對目前 YOLO26M 的適配研究](<../../docs/research/2026-09-04-paper31-to-yolo26m-training-adaptation.md>) |
| 執行計畫 | [最小實驗計畫](<plan.md>) |
| 終端架構圖 | [目前、論文與建議資料流](<architecture-report.md>) |

## 直接答案

這篇論文第 3.1 節的訓練方法**有一部分可以用在目前 YOLO26M 上**，但最合理的移植不是：

```text
COCO → DOTA → YOLO26M
RGB → gray+HOG 10 channels → 改造 stem
全模型統一使用論文的 5e-5
```

而是：

```text
RGB ───────────────────────────────→ 現有 YOLO26M ─→ 原生 task loss
 │                                        │
 └→ luminance → fixed HOG9 target         └→ P3 → temporary 1x1 head
                    │                                  │
                    └──── object-aware companion loss ┘

訓練後：移除 target generator 與 temporary head；推論仍只剩原本 RGB YOLO26M。
```

它保留論文「讓固定 filter 幫助 representation learning」的核心，但把 filter 從永久輸入改成暫時監督，
因此不改 pretrained RGB stem、不要求硬體每張圖重算 HOG，也不增加正式模型參數與 FLOPs。

## 為什麼不是直接照搬論文

論文的問題是 natural RGB／optical remote sensing 到 SAR 的大 domain gap。官方 MSFA filter 路徑會：

1. 把輸入取 channel mean 成 gray；
2. 在 forward 內計算固定 HOG／Canny／WST；
3. 把 gray 與 filter channels concatenate；
4. 改寫 backbone 的 `in_channels`。

這對目前模型有四個直接問題：

- YOLO26M 的 RGB pretrained stem 權重 shape／通道語義被改變；
- 制服、皮膚、草地與球棒背景的色彩線索會被削弱；
- filter 每張 inference image 都要重算，不符合既有硬體路徑；
- HOG/WST 可能和 early convolution 已學到的邊緣高度冗餘。

所以 HOG 本身不是不能用，而是**不應作永久輸入**。

## 為什麼 DOTA bridge 不適用

目前 COCO80 與未來 COCO person-only Runtime View 使用同一批 RGB COCO 影像；差的是 labels／head scope，
不是影像 domain。中間插入 DOTA 會同時改成 aerial imagery、rotated-box annotation、額外資料與額外 steps，
沒有和 person-only 目標相符的機理。

BBAT5 canonical labels又只有 ball／bat Detect/Pose，沒有 person GT，因此不能把 BBAT5 寫成
`COCO person → BBAT person` 的 fine-tune target。若日後研究 relevant-image curriculum，必須另案以
matched random sampler 作負控制，不能混進本方向。

## 現在的 YOLO26M 已做過哪些訓練

目前 authoritative Full35 不是未調過的 baseline：

- `J0`：Pose head only；
- `J1`：Neck + Detect/Pose heads；
- `J2`：layer 9+ backbone、Neck、MASF、heads；
- `J3`：full low-LR refinement，才開 attention 可微部分；
- AdamW、role-specific LR、warmup、cosine、plateau、AMP recovery、Bit-True selection都已存在。

Attention 主線也已做 `5e-6 / 1e-5 / 2e-5` sweep，較高 LR 沒有更好；best-observed 只比 parent高約
`0.000194`，正式仍保留 zero-train winner。因此本方向首輪不重做論文的五組 LR，也不改 optimizer／scheduler。

## 精確作用位置

以目前 Full35 graph：

```text
layer16：C3k2 → P3 MASF → F3 (256 channels, stride 8)
layers17…19              → F4 (512 channels, stride 16)
layers20…22              → F5 (512 channels, stride 32)
                                  ↓
DualHeadPredictionModule([F3,F4,F5])
```

`DualHeadPredictionModule.forward(features)` 已直接收到 `[F3,F4,F5]`；現在的第一個 feature就是
post-MASF P3。未來若 MASF 改成 Detect-only fork，位置規則不綁死 layer index，而改為：

> 取「該 task prediction head 真正消費的 P3 feature」，不改 P4/P5 forward，也不把 auxiliary activation
> 餵回主圖。

## 方法定義

先由 model 實際看到的、完成 resize／flip／color augmentation 的 RGB tensor產生亮度：

\[
Y=0.299R+0.587G+0.114B.
\]

以固定、停止梯度的 dense HOG 產生 `9×80×80` target：

\[
T=\operatorname{sg}\!\left(\operatorname{HOG}_{9,\,cell=8}(Y)\right).
\]

由 P3 預測方向分布：

\[
P=\operatorname{softmax}_{bin}(h_\phi(F_3)),
\qquad h_\phi=\operatorname{Conv}_{1\times1}(256,9).
\]

令 `M` 是 train GT boxes 投影到 P3 的 soft filled mask，`A` 是停止梯度的局部 gradient energy：

\[
\ell_{hog}^{(b)}=
-\frac{\sum_{u,v,k}M_{buv}A_{buv}T_{bkuv}\log(P_{bkuv}+\epsilon)}
{\sum_{u,v}M_{buv}A_{buv}+\epsilon}.
\]

每張先正規化，再對 physical batch 求和：

\[
L_{raw}=L_{native,raw}+\mu(t)\sum_b\ell_{hog}^{(b)}.
\]

這一點不可寫成 batch mean 後直接相加，因目前 `MacroStepEngine` 的 native loss是 batch-summed，之後才按
Detect/Pose真實圖片數、task weight與 reference batch size 縮放。

## 訓練時段

```text
J0                         ：mu=0
J1 warmup                  ：mu=0
J1 warmup後                ：0 → mu0，再保持
J2 最前8 epochs            ：mu0 → 0
J2其餘與J3                 ：mu=0
```

`mu0` 不做大 grid；先在固定 batch probe 令 auxiliary/native shared-gradient norm ratio 的中位數落在
`0.05–0.15`。這是安全校準區間，不是精度結論。正式 validation前鎖定，不可看結果再調。

## 為什麼先用 P3／HOG9

- P3 是 head 的最高解析度輸入，640輸入時為 `80×80`；HOG cell 8可一一對齊。
- HOG保留局部方向分布，比 hard Canny threshold更適合做平滑 target。
- WST 的 channel與計算明顯較重，第一輪沒有必要。
- side head只有 `256×9+9=2,313` 個 training-only parameters；成功 strip後 inference增加為0。

它仍可能失敗：P3對極小 ball/bat可能已太粗、球場直線可能主導 HOG、或 auxiliary gradient與 native task
方向衝突。因此首輪必須記錄有效 HOG cells、gradient cosine、clip/overflow、wall time與VRAM，而不能只看
auxiliary loss有沒有下降。

## 最小實驗與判定

三臂共用同一個註冊了 dormant auxiliary role 的 J0 exact-resume parent、相同 optimizer state、資料順序、
macro exposure、role LR、augmentation、BinaryQK與MASF：

| Arm | Target | 回答的問題 |
|---|---|---|
| `F0-MATCH` | `mu=0`，不做 aux forward | 新程式／optimizer group 的 matched control |
| `F1-LUMA9` | 相同 head/mask/loss 的 9-bin local luminance | generic companion supervision 是否已足夠 |
| `F2-HOG9` | 9-bin normalized HOG orientation | 方向先驗是否有額外價值 |

若算力只允許最簡版本，先跑 `F0/F2`；只有出現至少 `+0.001` joint/person訊號，才補 `F1`。
此時兩臂只能回答「HOG recipe有沒有觀測增益」，不能先宣稱增益來自 orientation。

完整 HOG-specific 結論要求：

- `F2` joint score相對 F0至少 `+0.001`；
- 八項正式指標任一項不得低於 F0超過 `0.001`；
- COCO person或 ball pose至少一項改善 `0.002`；
- `F2 > F1`；
- seed0通過後才補 paired seeds 1/2；
- inference graph/input/params/FLOPs/latency與 F0一致。

## 與其他方向的邊界

- BinaryQK 的 FP-teacher／attention ranking KD 已屬 Q0，本方向不重複加入 teacher。
- Detect/Pose gradient projection 是獨立候選；首輪不與 HOG 疊加。
- MASF relocation、RepConv或 person head改動後，舊 HOG結果不再是 final-lineage證據；須從新 frozen parent重開。
- HOG方向若失敗就停止，不依序掃 Canny、WST、P2、bins、cell與多個 `mu` 掩蓋失敗。

返回[優化方向索引](<../README.md>)。
