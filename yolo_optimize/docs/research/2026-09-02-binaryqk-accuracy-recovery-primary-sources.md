# BinaryQK 精度恢復：第一手證據、誤差推導與最小實驗建議

- 日期：2026-09-02
- 問題：YOLO26 的 BinaryQK 相對 FP 掉點過多，哪些方法最可能補回，應先測什麼
- 本地範圍：`yolo_attention` 的兩個 YOLO26 BinaryQK sites，以及舊 `yolo_binaryqk` 的 YOLO11 單-site 結果
- 證據規則：外部只採原始論文、作者 repository／原始碼及框架官方文件
- 非目標：不啟動 GPU 訓練、不修改 production model、不把跨架構論文結果宣稱為本專案保證

> **硬體優先補充：**使用者確認 production 不希望每張 image 重算 scale。既有 YOLO26
> V1-DYN/SHEAD/P2 已完成 global dynamic、fixed-head、fixed-PoT evaluation-only消融，dynamic只比
> PoT 高 `0.000506`。因此 deployment-first計畫改為重用此結果、先做 fixed-PoT site isolation；
> per-token dynamic保留為條件式 accuracy ceiling，而不是首輪 production候選。

## 1. 結論先行

1. **BinaryQK 不是沒有機會，而是目前 YOLO26 還沒有完成足夠乾淨的 site 歸因與 recovery。** 現行兩個 binary sites 是 `model.10.m.0.attn` 與 `model.22.m.0.1.attn`。本地 B26-FP 為 `0.517998`，目前較好的 W-DIR 為 `0.507457`，仍差 `0.010540` mAP；A-FINAL 為 `0.506357`，差 `0.011641`。這些值來自同一份 [YOLO26 attention 報告](<../../../yolo_attention/reports/REPORT.md>)，但仍是 Ultralytics internal metric，不是 canonical COCO API。
2. **主要嫌疑不是 Softmax approximation。** 本地 Exact／PWL／SHIFT 的 zero-train 差異都遠小於 `0.001`，而 W-DIR 明顯優於 W-PROG；因此第一輪不應再搜尋 normalization 或 progressive blend。真正需要量的是 Q/K sign 後的 magnitude loss、score ranking disorder、STE saturation，以及兩個 sites 的誤差傳播。
3. **硬體優先的三件事依序是：**
   1. 保留 fixed PoT，做 `site-10-only`、`site-22-only`；敏感 site維持 FP，只二值化較安全的一處。
   2. 對 site winner 做 direct QAT；ranking/KL仍差才加一種 FP-teacher loss，saturation高才加
      training-only STE-window。
   3. 前兩步失敗且接受 dynamic硬體成本時，才成對測 `DYN-GLOBAL` 與 **per-token/head dynamic**。
4. **per-head learnable gamma／threshold 是後續條件式方法。** 它只在 per-token scale 後仍顯示 head 間 sign balance 或 STE saturation 差異很大時值得測；它比 dual-basis 便宜，但仍需新的 bit-true／export 契約。
5. **dual／multi-basis 可提高表示上限，但不是第一輪。** 舊 YOLO11 的 full residual dual-basis 確實把差距縮到約千分位；然而目前 YOLO26 Hadamard two-basis 未勝 single-path W-DIR，而且 full residual dual 需要最多四次 binary QK products。若單-site、recovery、KD／threshold 都失敗，才用它探 accuracy ceiling。
6. **INT8 calibration 不能補回 1-bit QK 的表示誤差。** INT8 P/V、projection 或 deployment calibration 是另一個精度邊界；本地 fake quant 也不等於已產生 XNOR-popcount kernel。沒有 custom binary kernel、convert/export artifact 與 target-device latency，就不能宣稱 BinaryQK 已帶來端到端加速。

## 2. 證據標籤與外推邊界

- **【論文證據】**：原始論文直接支持的機制或該論文設定下的結果。
- **【作者實作】**：作者 repository／source code 直接顯示的公式、訓練入口或 kernel 契約。
- **【本地證據】**：本專案 source、manifest 或實驗報告直接支持。
- **【本專案推論】**：由第一手資料與本地結構導出的可否證假說；尚未由新的 YOLO26 matched experiment 驗證。

BinaryAttention 的 DeiT、detection 或 A100 kernel 結果，Bi-ViT 的 fully-binarized ViT 結果，以及 ReActNet／ABC-Net 的 binary CNN 結果，都不能直接當成 YOLO26 QK-only detection 的預期增益。它們只能支持方法機制與候選順序。

## 3. 本地 BinaryQK 到底掉多少

### 3.1 現行 YOLO26：仍有約 1.05 AP 缺口

【本地證據】本地 [YOLO26 attention 完整報告](<../../../yolo_attention/reports/REPORT.md>) 的同 evaluator 結果如下：

| 模型／階段 | mAP50-95 | 相對 B26-FP | 解讀 |
|---|---:|---:|---|
| B26-FP | 0.517998 | 0 | 本地 FP parent |
| I-SCR | 0.492482 | -0.025515 | identity-basis binary screen |
| H-SCR | 0.495243 | -0.022755 | Hadamard two-basis screen |
| W-DIR | 0.507457 | -0.010540 | 目前較佳 direct recovery |
| W-PROG | 0.502866 | -0.015132 | progressive recovery 未勝 direct |
| A-FINAL | 0.506357 | -0.011641 | 演算法流程選出的 final，不是 accuracy 最高 parent |

W-DIR 規劃 40 epochs，但在 26 epochs early-stop，best epoch 是 21；W-PROG 在 11 epochs early-stop，best epoch 是 6。W-DIR 當時已是全模型解凍，不是 attention-only recovery；其 `best.pt` 也已在核准清理中刪除。這不能證明「只增加 epoch／解凍更多參數就一定改善」，也不能把不存在的 W-DIR checkpoint 當成 resume parent；它只支持不再優先重跑原樣 progressive blend。

### 3.2 normalization 已不是最大嫌疑

【本地證據】同一報告的 normalization zero-train screen：

| 方法 | mAP50-95 | 相對 Exact |
|---|---:|---:|
| Exact Softmax | 0.506658 | 0 |
| LUT | 0.506700 | +0.000043 |
| PWL | 0.506772 | +0.000114 |
| SHIFT | 0.506746 | +0.000089 |

【本專案推論】這四者的差都小於 `0.001`；因此目前約 `0.0105` 的 FP gap 不能主要歸因於 Exact Softmax 被近似。繼續微調 normalization 最多是在 binary parent 周圍做次要工程 trade-off，不是主要 accuracy recovery。

### 3.3 舊 YOLO11 證明「可以補回」，但不能直接外推

【本地證據】舊 [YOLO11 BinaryQK 研究](<../../../yolo_binaryqk/README.md>) 只改一個 attention site：

| 變體 | COCOeval mAP50-95 | 相對同階段 FP |
|---|---:|---:|
| E0 FP zero-train | 0.510850 | 0 |
| E1-S raw sign | 0.459297 | -0.051554 |
| E1 scaled sign | 0.480561 | -0.030289 |
| E2-DUAL zero-train | 0.495645 | -0.015205 |
| T0 FP 10-epoch control | 0.512671 | 0 |
| T4 full residual dual + QAT | 0.511631 | -0.001040 |
| T6-F/A feature + attention KD | 0.511841 | -0.000830 |

【本專案推論】這組結果直接支持「QAT、dual representation 與 KD 在本專案舊 lineage 曾補回大部分掉點」，但它只有一個 binary site、10 epochs、single seed，模型是 YOLO11；不能推定 YOLO26 的兩-site W-DIR 也會得到相同幅度。

## 4. 目前公式為何會丟精度

### 4.1 本地實作保留了什麼、丟掉了什麼

【本地證據】[`BinaryScore`](<../../../yolo_attention/src/yolo_attention/binary_basis.py>) 支援 dynamic、
fixed-head與 power-of-two 三種 modes。正式 V1-BR／A-FINAL 是 `power_of_two`：calibration 時先對每個
sample/head 以 channel與 token 的全域平均幅度估計 dynamic coefficient：

```text
a_q = mean_{channel,token} |Q|
a_k = mean_{channel,token} |K|

c_dyn = gamma · a_q · a_k / sqrt(d)
c_fixed[h,basis] = PoT_round(mean_calibration c_dyn[:,h,basis])
S_bin(i,j) = c_fixed[h,basis] · <sign(q_i), sign(k_j)>
```

`gamma` 對 identity basis 只有一個可學 scalar；Hadamard／T5 各 basis 一個 scalar。calibration 的
`a_q/a_k` 是 per-sample、per-head，但正式 inference 的 `c_fixed` shape 是 `[1,H,1,1]`；同一
head/basis 的所有 images與 token pairs共用一個 PoT coefficient。A-FINAL 的 site 10 四個 heads 在
identity/Hadamard 都是 `0.25/0.25`；site 22 的 head 0 是 `0.125/0.25`，其餘是 `0.25/0.25`。
forward sign 在 eval 將零映為 `+1`；train 使用 clipped STE，只有 `|x| <= 1` 傳梯度。

【本地程式碼限制】`BinaryScore._coefficient()` 目前在判斷 `ScaleMode.DYNAMIC`／calibration之前，
無條件呼叫 `_dynamic_coefficient()`；因此 eager fixed-mode reference仍會先算每-image `abs().mean()`，
再丟棄該值並回傳 `_fixed()`。Hadamard forward對 identity與 transformed basis各呼叫一次
`_coefficient()`，等於每 site白算 Q/K/Q_h/K_h四次 reduction、兩 site共八次。這是可用
early-return移除的實作冗餘，不是 fixed-PoT演算法或硬體的必要運算；production latency不得在保留
此冗餘的 reference上宣稱最終結論。

【作者實作】BinaryAttention 作者的 [`models.py`](https://github.com/EdwardChasel/BinaryAttention/blob/main/models.py#L97-L136) 同樣先以 mean-absolute scale 乘 sign，再做 QK score；P 用靜態 unsigned 8-bit，V 用 8-bit fake quant。這支持本地 scaled-sign 的基本方向，但不代表本地兩-site recipe 已完整重現作者訓練。

### 4.2 誤差可分解成三個被丟掉的 residual 項

對單一 head、query token `i`、key token `j`，令：

```text
q_i = a_q b_qi + r_qi,   b_qi = sign(q_i)
k_j = a_k b_kj + r_kj,   b_kj = sign(k_j)
```

則 full-precision dot product 為：

```text
q_i^T k_j
= a_q a_k b_qi^T b_kj
  + a_q b_qi^T r_kj
  + a_k r_qi^T b_kj
  + r_qi^T r_kj
```

對每一個 identity／Hadamard basis，sign-dot approximation只保留相對應的第一項；兩個 bases相加
仍不會自動恢復下列 token-pair residual。其概念性 logit error 為：

```text
Delta S_ij
= -[a_q b_qi^T r_kj
    + a_k r_qi^T b_kj
    + r_qi^T r_kj] / sqrt(d)
```

【本專案推論】若所有 token 的 magnitude pattern 相近，一個 head-level scale 可能已足夠；若 ball、bat 或背景 token 的 channel magnitude 差異很大，三個 residual 項會依 token pair 改變，global scale 無法把它們補回。這正是 per-head／per-token scale、bias、KD 或 residual basis 各自要處理的誤差來源。

### 4.3 Per-token/head scale 已有正向 CPU 機理訊號

作者實作的 Q/K layout 是 `[B,H,N,D]`，其 `_quantize` 對 `N,D` 取平均；本地 dynamic path 的
layout 是 `[B,H,D,N]`，對 `D,N` 取平均，兩者語義都是每個 sample/head只保留一個 global
magnitude。本地正式 PoT path又將這些 calibration observations壓成 fixed per-head/basis coefficient。
候選公式改成：

```text
a_q[b,h,i] = mean_d |q[b,h,d,i]|
a_k[b,h,j] = mean_d |k[b,h,d,j]|

S_token(i,j)
= gamma · a_q[b,h,i] · a_k[b,h,j]
  · <sign(q_i), sign(k_j)> / sqrt(d)
```

【本地證據】本次父任務以兩張 COCO image 做 CPU probe（尚未形成正式 artifact），比較
`global dynamic` 與 `per-token dynamic`：只把 scale 軸由 `D,N` 全域平均改成只對 `D` 平均，
其他 score path 不變。它不是對正式 fixed-PoT checkpoint 的直接比較：

| site | global-scale top-10 overlap | per-token top-10 overlap | global-scale KL | per-token KL |
|---|---:|---:|---:|---:|
| `model.10.m.0.attn` | 0.4206 | **0.5119** | 0.5431 | **0.4651** |
| `model.22.m.0.1.attn` | 0.4962 | **0.5952** | 0.8810 | **0.6188** |

兩個 sites 都同時提高 top-10 overlap 並降低 KL，支持「global magnitude 粒度可能是表示瓶頸」這個
候選；但樣本只有兩張、沒有正式 artifact、沒有完整 validation AP，因此不能宣稱已補回 mAP。
既有 scale ablation 的 dynamic `0.507457` 與 PoT `0.506952` 只差 `0.000506`，顯示 fixed PoT 本身
不是完整 `0.011` gap 的主因，也不值得為此優先恢復每-image reduction。若 hardware-friendly recovery
仍失敗並重新開啟 per-token研究，才須將 `DYN-GLOBAL` 與 `TOKEN-BOTH` 成對執行，以乾淨歸因收益。

【本專案推論】per-token scale 保留每個 query／key token 的幅度，直接縮小上一節因共用 `a_q/a_k` 造成的 token-pair residual error；它仍無法恢復 channel-wise magnitude，也不保證完整 ranking。

硬體代價不可忽略。若 `B=1,H=4,N=400`，每個 site 的 Q+K 有 `2×B×H×N=3,200` 個 scales，
兩站共 `6,400`；score matrix 的 outer-product scale為每站 `B×H×N²=640,000` 個 pair factors，
兩站共 `1,280,000` 個 score epilogue scale operations／image。這是共用單一 scale組的下限；若
identity與 Hadamard各自保留 scale，兩者分別加倍為 `12,800`、`2,560,000`。scale可嘗試在 tiled
kernel epilogue即時計算而不 materialize完整矩陣，但仍可能侵蝕 binary matmul gain，所以它必須
同時通過 accuracy與 target-kernel latency gate。

### 4.4 修正整體 scale 不等於修正 ranking

令 `p = softmax(s)`；小擾動的一階近似為：

```text
Delta p ~= [diag(p) - p p^T] Delta s
```

同一列所有 logits 加相同常數不改變 softmax，但 token-pair-specific 的 `Delta s` 會改變 pairwise differences、top-k 以及概率質量。單一 `gamma` 或 temperature 只能縮放整列，不能任意恢復原本排序。

【論文證據】[Bi-ViT](https://ojs.aaai.org/index.php/AAAI/article/view/28109) 把 fully-binarized ViT 的主要問題分析為 attention distortion、gradient vanishing 與 ranking disorder，並用 learnable head-wise scale 及 ranking-aware teacher distillation處理。這是較激進的 fully-binarized 架構，對本專案只構成機制證據，不是預期 AP 保證。

【論文證據】BinaryAttention 論文指出，丟失 magnitude 會使 binary attention 過度平坦，因而加入 scale、bias、QAT 與 self-distillation；其 DeiT Table 5 中，scale、distillation、bias 在作者 recipe 下逐步改善結果。[論文公式與消融](https://arxiv.org/html/2603.09582#S4) 與 [作者 repository](https://github.com/EdwardChasel/BinaryAttention) 均支持這個完整組合，但各元件在 YOLO26 上的收益必須重測。

### 4.5 clipped STE 會形成飽和區

【本地證據】本地 clipped STE：

```text
forward:  b = sign(x)
backward: db/dx ~= 1{|x| <= 1}
```

因此 `|Q|>1`、`|K|>1` 的元素對 sign boundary 沒有直接 STE gradient。若 head 間分布不同，同一固定 window 會讓某些 heads 大量飽和，另一些 heads 又聚集在零附近而對微小擾動敏感。

【論文證據】Bi-ViT 明確推導 clipped STE 的 gradient mismatch，並用 head-wise learnable scale 改變 effective clip range，使部分消失梯度恢復；其 scale 與 ranking-aware distillation 分別在該 fully-binarized ImageNet 設定帶來改善。[AAAI 原論文第 4 節](https://ojs.aaai.org/index.php/AAAI/article/download/28109/28222) 是直接來源。

### 4.6 兩個 sites 可能把誤差傳到不同尺度

現行資料流可簡化為：

```text
backbone P5
  └─ model.10.m.0.attn：BinaryQK-10
       └─ top-down / bottom-up Neck
            ├─ P3 Detect feature
            ├─ P4 Detect feature
            └─ model.22.m.0.1.attn：BinaryQK-22
                 └─ P5 Detect feature
```

【本地證據】兩個 module path 由 [`HardwareFriendlyAttention`](<../../../yolo_attention/src/yolo_attention/attention.py>) graft 與現有 experiment manifest 確認；舊 YOLO11 只有一處 binary attention。

【本專案推論】site 10 的 feature drift 可向後進入整個 Neck，site 22 更靠近 P5 output。因此 two-site gap 可能來自單一敏感 site，也可能來自兩處誤差交互。沒有 `site-10-only`／`site-22-only`，就無法知道該擴大模型 capacity、改善 quantizer，還是只要保留一處 FP。

## 5. 各恢復方法的證據、適用條件與成本

### 5.1 Per-token/head dynamic scale：條件式 accuracy ceiling

【本地證據】第 4.3 節兩張 COCO CPU probe 在兩個 sites 都提高 top-10 overlap並降低 KL；相較於新增 basis，它沒有增加 binary QK products，直接針對現行 global magnitude 粒度不足。

【本專案推論】只有 fixed-PoT site/QAT/KD路徑失敗且接受 runtime scale成本時，才在同一 V1-BR
checkpoint 增加 `DYN-GLOBAL`，將 fixed PoT 改回 global dynamic；再以
`TOKEN-BOTH - DYN-GLOBAL` 歸因 token granularity。兩者保持 sign、basis、bias、normalization、sites
不變。完整 validation後才決定是否投入 QAT；同時必須 profile單一 scale組下限的 `6,400` 個 token
scales與 `1,280,000` 個 pair-scale epilogue operations／image，以及 two-basis獨立 scale上界。
accuracy或 kernel latency任一不過 gate就停止。

### 5.2 Site isolation／hybrid precision：deployment 第一優先

【本專案推論】最小因果矩陣是：

| ID | site 10 | site 22 | 回答的問題 |
|---|---|---|---|
| S0 | FP | FP | matched FP control |
| S1 | Binary | FP | 上游 site 10 的單獨代價 |
| S2 | FP | Binary | P5-output site 22 的單獨代價 |
| S3 | Binary | Binary | two-site interaction／現況重現 |

如果 S1 已承擔大部分掉點，最有效率的 accuracy recovery 可能是 `site10=FP, site22=Binary`；反之亦然。如果 S1、S2 都接近 S0，只有 S3 下降，才支持 staged conversion／joint adaptation，而不是把任一 site 判定為天生不適合二值化。

這個方法的優點是不用先發明新 quantizer；缺點是 hybrid precision 減少 BinaryQK 覆蓋率。是否仍有產品價值，必須用 target backend 的 end-to-end latency，而不是只看 binary op proxy 決定。

### 5.3 Direct QAT：所有訓練候選的共同 protocol

【論文證據】BinaryAttention 從 FP pretrained model 初始化，以 STE 做 QAT 並用 FP counterpart 作 teacher；作者的 DeiT 無 bias 結果由 100 epochs 延長至 300 epochs後，Tiny／Small／Base 從 `71.98/79.44/81.80` 到 `72.44/79.97/81.99`。[論文 Appendix C](https://arxiv.org/html/2603.09582#S10) 支持「充分 adaptation 會影響極低 bit 結果」，但訓練長度不能直接照搬到 YOLO26。

【作者實作】作者 [`main.py`](https://github.com/EdwardChasel/BinaryAttention/blob/main/main.py#L143-L156) 提供 teacher distillation、FP checkpoint fine-tune 與 `--attn-only` 選項；這也說明 full recipe 與 attention-only 是不同 trainable-scope 契約。

【框架官方】PyTorch 官方將 QAT 定義為訓練期間以高精度 tensor 模擬 quantize/dequantize numerics，之後仍需要 convert 才得到實際量化圖；fake quant 本身不是低 bit runtime。[torchao QAT 文件](https://docs.pytorch.org/ao/stable/workflows/qat.html)

【本地證據】W-DIR `0.507457` 已高於 W-PROG `0.502866`，但 W-DIR 已全模型解凍且 checkpoint 已刪除；因此只能沿用 direct 方法，不可聲稱能從 W-DIR `best.pt` resume。

【本專案推論】新 arm 應只保留一個主要變因：

1. 若 retained `V1-BR` 的 resolved config 與目標 arm 完全相容，才可經 state-transfer audit 作 parent；否則從 FP checkpoint建立新的 direct-QAT lineage。
2. trainable scope 必須預先固定並與 control 相同；不把 staged unfreeze 與新 scale/KD 同時加入。
3. 同 schedule 跑 matched FP/no-KD control，分離正常 fine-tune gain與 BinaryQK recovery。
4. 將 early stopping patience 與 planned epochs 分開記錄；best epoch 仍用預先定義 metric 選擇。

QAT 是 accuracy adaptation；若仍使用 float fake-sign matmul，它不構成 1-bit runtime 證據。

### 5.4 FP teacher KD／ranking-aware KD：第三優先

【論文證據】BinaryAttention 使用 FP counterpart self-distillation；其 DeiT Table 5 在 scaled binary 上加入 distillation 後，Tiny／Small／Base 分別由 `72.42/79.81/81.33` 到 `72.44/79.97/81.99`。[原論文消融](https://arxiv.org/html/2603.09582#S7.SS6) 顯示增益依模型大小不同，不能外推為固定 AP。

【論文證據】Bi-ViT 定義 teacher／student attention 的 ranking-aware loss，對 attention score 的相對次序作一致性約束；其原始公式與實驗見 [AAAI 論文](https://ojs.aaai.org/index.php/AAAI/article/download/28109/28222)。它量化整個 ViT，遠比本專案的 QK-only 更激進，因此只支持 ranking loss 的候選性。

【論文證據】Kim et al. 對 sub-2-bit Transformer QAT 的分析顯示，直接對 attention scores 做 MSE 不足以保留 token 相對重要性；在其 BERT 設定中，對 softmax attention map 做 KL divergence 比 score MSE 更能降低 ranking distortion，而且 attention-map 與 attention-output loss 的偏好具有模型／任務相依性。[原論文與公式](https://arxiv.org/html/2211.11014#S3.SS2) 來自 NLP Transformer，故只支持 loss 設計，不保證 detection AP。

【本地證據】舊 YOLO11 中，T4 binary QAT 到 FP T0 的 COCOeval gap 是 `0.001040`，T6-F/A 後是 `0.000830`。這是 single-seed 小差異，只能視為本地可行性訊號。

【本專案推論】不要第一個 KD run 同時疊 output、feature、attention-map、ranking 四種 loss。先依診斷選一個：

- detection outputs 已偏移，但 attention ranking 尚可：先做 output／feature KD；
- FP-vs-binary score top-k overlap 或 rank correlation 很差：做 ranking-aware attention KD；
- probability entropy／KL 差，但 ranking 尚可：做 temperature-aware attention-map KL。

最小 teacher arm 應固定 teacher、augmentation、loss weight、site 與 trainable scope，並保留 no-KD matched arm。teacher 只增加訓練成本，不應進部署 graph。

### 5.5 Per-head learnable gamma／threshold／temperature：有診斷才做

目前可區分三個參數：

```text
threshold:    b_q = sign(q - tau_q,h), b_k = sign(k - tau_k,h)
scale:        q_hat = alpha_q,h b_q,    k_hat = alpha_k,h b_k
temperature:  S_hat_h = g_h S_bin,h
```

- `tau` 改 sign boundary，主要處理正負號極度不平衡。
- `alpha` 改近似 magnitude 與 STE effective range。
- `g` 改 softmax 前 logit sharpness，但不會恢復被改掉的 token ranking。

【論文證據】Bi-ViT 採 learnable head-wise scaling factor；這是 attention-specific 的間接證據。[Bi-ViT 原論文](https://ojs.aaai.org/index.php/AAAI/article/view/28109)

【論文證據】[ReActNet 論文](https://arxiv.org/abs/2003.03488) 與 [作者 repository](https://github.com/liuzechun/ReActNet) 以 learnable channel-wise bias／RSign 改變 binary activation threshold，支持「零閾值不一定適合所有 channel」。它是 binary CNN，不是 QK attention，不能證明 YOLO26 會增準。

【論文證據】[QKNorm](https://aclanthology.org/2020.findings-emnlp.379/) 在每個 head 的 channel 維度對 Q/K 做 L2 normalization，再用 learnable scale 取代固定 `sqrt(d)`；它支持 scale／temperature 可控制 softmax saturation，但證據來自低資源翻譯，不是 binary vision detection。

【論文證據】[LSQ](https://research.ibm.com/publications/learned-step-size-quantization) 顯示 quantizer step size 可與網路參數共同學習，且需處理 step-size gradient scale；它研究 2–4 bit weights／activations，不是 1-bit sign threshold，故只能支持「量化參數可學」的通用原理。

【論文證據】[Q-ViT](https://arxiv.org/abs/2201.07703) 觀察到 ViT attention heads 的 quantization robustness 不同，因而學習 head-wise bit-width 與 scale。這支持先量 head sensitivity、再考慮 selective precision；它是 3-bit ViT 研究，不支持直接指定本專案哪個 head 應保持 FP。

【本地證據】舊 YOLO11 dual-basis code 已含 per-head、per-channel threshold，但它與 residual dual basis 綁在一起，不能從 T4 結果單獨歸因 threshold。現行 YOLO26 dynamic magnitude 是 per-sample/per-head，`gamma` 卻只按 basis 共用；因此最小新 arm 應先做 `per-head gamma/temperature`，再按 sign-balance 診斷決定是否加 `tau_q/tau_k`。

部署注意：threshold 後仍可輸出 1 bit，但 compare、packing 與 threshold state 是否能融合進 custom kernel必須實測；per-head FP scale／temperature 也會留下 side information。不能只因 parameter 數很少，就宣稱 runtime cost 為零。

### 5.6 Dual／multi-basis：accuracy ceiling，不是最小方案

兩個 residual bases 可寫成：

```text
q ~= a_q1 b_q1 + a_q2 b_q2
k ~= a_k1 b_k1 + a_k2 b_k2

q^T k ~= a_q1 a_k1 b_q1^T b_k1
        + a_q1 a_k2 b_q1^T b_k2
        + a_q2 a_k1 b_q2^T b_k1
        + a_q2 a_k2 b_q2^T b_k2
```

full residual dual 恢復四個 cross terms，但 binary QK products 從一次增至四次；matched dual 只留 `(1,1)`、`(2,2)` 兩項，成本較低但丟掉 cross terms。

【論文證據】[ABC-Net](https://papers.neurips.cc/paper/6638-towards-accurate-binary-convolutional-neural-network.pdf) 用多個 binary weight bases／activations逼近 full precision，顯示增加 binary bases 可提高表示能力；它是 CNN，不是 attention-specific 證明。

【本地證據】舊 YOLO11 T4 full residual dual 在 attention-only QAT 後很接近 FP；但目前 YOLO26 H-SCR／Hadamard two-basis 在 recovery 前只比 identity screen 好，且正式 W-DIR 仍是更好的 accuracy parent。這表示「basis 數更多」不保證在不同 basis construction／training recipe 下更好。

【本專案推論】只有在以下條件同時成立時才測 full residual dual：

1. one-site／hybrid policy 已定；
2. direct recovery 與一個 KD／threshold arm 仍超出 accuracy gate；
3. binary attention 是 target latency 的顯著熱點；
4. 四次 binary products 後仍有部署淨收益。

### 5.7 Progressive quantization：目前不重跑

【作者實作】ReActNet 作者流程分成先 binarize activations、再 binarize weights的兩個 training stages，[官方 repository](https://github.com/liuzechun/ReActNet) 說明 staged conversion 在 binary CNN 有先例；這不是「FP/binary attention score 線性混合必勝」的證據。

【論文證據】Zhuang et al. 的 progressive quantization 分別研究「先量 weights、後量 activations」以及逐步降低 bit-width，以尋找較佳 low-bit basin；同一研究也提出 stochastic precision 與 joint full-/low-precision training。[原始論文](https://arxiv.org/abs/1908.04680) 是 CNN 跨架構證據，不能推定本地 score-level blend 應該有效。

【本地證據】本地 [`ProgressiveBlend`](<../../../yolo_attention/src/yolo_attention/schedule.py>) 在 10 epochs 內由 FP score 線性切到 binary score；W-PROG 為 `0.502866`，低於 W-DIR 的 `0.507457`，且更早 early-stop。

【本專案推論】目前沒有理由原樣重跑 W-PROG。若未來 single-site direct QAT 在 epoch 0 發散，才測「逐 site 切換」或「先 scale/threshold、後 sign」；不要再用同一個已失敗的 score blending schedule。

### 5.8 Normalization／temperature：只按診斷處理

【論文證據】QKNorm 說明 L2-normalized Q/K 與 learnable scale 可降低 arbitrary softmax saturation；BinaryAttention 也指出 binary score 可能過度平坦並用 scale／bias補償。[QKNorm](https://aclanthology.org/2020.findings-emnlp.379/)；[BinaryAttention](https://arxiv.org/html/2603.09582)

【本專案推論】本地 normalization approximation 已近似無損，所以只有在 per-head 診斷發現 entropy 明顯過高／過低時，才調 `g_h` 或 QK pre-normalization。若 top-k 已亂，temperature 不會單獨修復 ranking。

## 6. 最小必要實驗，不再擴成大矩陣

### 6.1 先做不改模型的診斷

用同一批 validation samples、同一 FP／binary checkpoint，對兩個 sites、每個 head 記錄：

- `cosine(S_fp, S_bin)`、RMSE、relative L1；
- top-k overlap、Spearman／pairwise ranking agreement；
- attention entropy、最大 probability、FP-vs-binary KL；
- Q/K 正號比例、`|Q|>1`／`|K|>1` 的 STE saturation 比例；
- threshold 附近密度，例如 `|Q|<epsilon`、`|K|<epsilon`；
- site output feature drift，以及 APs/APm/APl／ball／bat 分解。

【論文證據】BinaryAttention 本身以 attention-map cosine、relative L1、RMSE 與 top-100 precision評估 FP／binary fidelity，[論文 Table 6](https://arxiv.org/html/2603.09582#S7.SS6) 支持這類診斷，而非只看 final accuracy。

### 6.2 首輪只新增兩個 site validations

| 階段 | 實驗 | 保持不變 | 通過後做什麼 |
|---:|---|---|---|
| 首輪 | `ISO-10-BIN`、`ISO-22-BIN` | fixed PoT、parent、bias、normalization、evaluator | 決定 hybrid／both policy |
| recovery | winner + matched FP direct QAT | epochs、LR、trainable scope | 量 paired FP gap |
| 條件式 | 單一 FP-teacher attention-map／ranking KD | site、scale、schedule | 判斷 teacher能否修 ranking |
| accuracy ceiling | `DYN-GLOBAL` + `TOKEN-BOTH` | sign、basis、bias、normalization、sites | 只有接受 dynamic成本才開啟 |

S0 FP 與 S3 both-binary 若現有 checkpoint、evaluator、recipe 完全 matched，可重用；只要其中一項不同，就必須重跑 control，不能用舊數字省掉因果基準。

### 6.3 決策規則

- S1 或 S2 明顯支配掉點：保留該 site FP，先驗證另一 site 的實際 speed／accuracy價值。
- S1、S2 各自接近 FP，S3 才掉：優先 joint recovery／逐 site conversion，不加 basis。
- saturation 高：先 per-head scale／STE range；sign ratio 極偏：再加 threshold。
- entropy 異常但 ranking 尚可：per-head temperature／bias。
- ranking 異常：teacher ranking KD；temperature 不是主解。
- direct recovery + KD 後仍失敗：才考慮 full residual dual。
- 任一候選若沒有 target backend speedup：停止用額外 basis 換 accuracy，回到 FP/hybrid。

## 7. INT8 calibration 與部署邊界

### 7.1 QAT checkpoint 仍不是低 bit runtime

【框架官方】torchao 說明 QAT 的 fake quant 仍以高精度 tensor 模擬 quantization，須在訓練後 convert 成實際 quantize/dequantize graph；因此本地 STE／fake-quant checkpoint只能證明 accuracy simulation。[PyTorch torchao QAT](https://docs.pytorch.org/ao/stable/workflows/qat.html)

【論文證據】BinaryAttention 的實際速度來自 A100 上 `mma.s32.b1.b1.s32` binary QK 與 `mma.s32.u8.s8.s32` P/V kernel，加上 FlashAttention2 式 tiling；不是 PyTorch `sign()` 自動獲得的速度。[BinaryAttention hardware-aware implementation](https://arxiv.org/html/2603.09582#S4.SS3)

### 7.2 INT8 calibration 只負責 INT8 tensor scale

【框架官方】TensorRT 對 INT8 PTQ 需要 representative calibration data，從 activation histogram建立 tensor scales；官方也指出 calibration 需平衡 rounding/discretization 與 clipping error，batch、資料順序、device／fusion 狀態會影響 cache 可攜性。[TensorRT quantized types／calibration](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)

【框架官方】TensorRT 現行建議 explicit Q/DQ quantization；INT8、INT4、FP8 等內建 dtype 與 scale契約不等同於 BinaryAttention 的 1-bit QK custom kernel。[TensorRT-RTX quantized types](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/work-with-quantized-types.html)

【本專案推論】部署順序應是：

```text
選定 FP／hybrid／BinaryQK accuracy winner
  -> freeze quantizer state 與 threshold/scale
  -> export/convert graph
  -> 接入或實作 binary QK kernel
  -> 另對 INT8 P/V、Conv/Linear 做 representative calibration 或 QAT
  -> bit-true parity
  -> target-device p50/p95 latency、throughput、memory、accuracy gate
```

1-bit QK、INT8 P/V 與其他 Conv/Linear INT8 必須各自列出 quantization boundary；不能讓一般 INT8 exporter把 custom BinaryQK 靜默回退成 FP16／FP32，再宣稱整網量化成功。

## 8. 推薦的三個先測方法

### 第一名：fixed-PoT site isolation，必要時採 hybrid precision

理由：兩個 sites 尚未被獨立歸因；若只有一處敏感，保留該處 FP/INT8可能比增加 basis更省成本。

成功訊號：一個 binary-only site 接近 FP，而另一個明顯掉點；hybrid accuracy與實測 latency達標。

### 第二名：FP teacher attention-map／ranking KD + direct QAT

理由：BinaryAttention 有 QK-only 的 self-distillation 原始證據，Bi-ViT 有 ranking disorder 的原始分析，本地 YOLO11 也有小幅正訊號。KD 不增加 deployment graph，是 dual-basis 前較便宜的 recovery。

成功訊號：KD 相對 no-KD matched arm 改善預先選定的 output／feature 或 ranking diagnostic，並轉化為多 seed AP 改善。

### 第三名（條件式）：per-token/head dynamic scale

理由：兩張 COCO CPU probe 已在兩個 sites 同方向改善 top-10 overlap與 KL，但它要求每-image reduction
與 N² pair-scale epilogue；既有 T5/dual/N4也已有 per-token-like負向或混合訊號。

成功訊號：完整 validation AP 相對 `DYN-GLOBAL` matched control至少改善 `0.001`，且 fused/低 bit
scale後的 target latency仍有淨收益。

## 9. 暫不先做的項目

| 項目 | 暫緩理由 | 重新開啟條件 |
|---|---|---|
| 原樣 W-PROG | 本地已低於 W-DIR | direct 在 single-site 發散 |
| 再搜尋 PWL/SHIFT/BDCN | normalization 不是主要 gap | deployment normalization 成為實測熱點 |
| 一次疊四種 KD loss | 無法歸因 | 單一 KD arm 已證明方向有效 |
| full residual dual | 2–4 倍 QK products與 kernel 複雜度 | 前三方法仍未達 gate且有 latency budget |
| 全部 sites 強制 1-bit | 可能犧牲高敏感 site | both-binary 確有端到端產品收益 |
| 先做 INT8 calibration 補 QK | INT8 scale 不恢復 sign 丟失資訊 | BinaryQK winner 與 export boundary 已 freeze |

## 10. 最終判斷

【本專案推論】目前最合理的判斷不是「BinaryQK 本身不可用」，也不是「再加更多 normalization 就會回來」。現有證據顯示：

```text
主要待解問題
= 未知的 site sensitivity
  + binary sign-basis residual information loss
  + token ranking disorder
  + clipped-STE / trainable-scope adaptation 限制
  + two-site error propagation
```

硬體優先的最短路徑是先以現行 fixed PoT拆兩個 sites，再讓唯一 hybrid winner與 matched FP control
做 direct QAT；仍有 ranking gap才加單一 teacher loss／STE-window。只有這條零或低 inference-overhead
路徑仍無法達標，而且願意承擔 dynamic scale硬體成本時，才成對測 global dynamic與 per-token
dynamic。這能避免重跑既有 scale消融，也避免用 fidelity改善換來不可接受的端到端成本。

## 11. 第一手來源

1. Chaodong Xiao, Zhengqiang Zhang, Lei Zhang, **BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers**：[原始論文](https://arxiv.org/html/2603.09582)、[作者 repository](https://github.com/EdwardChasel/BinaryAttention)、[作者模型實作](https://github.com/EdwardChasel/BinaryAttention/blob/main/models.py)、[作者訓練入口](https://github.com/EdwardChasel/BinaryAttention/blob/main/main.py)。
2. Yanjing Li et al., **Bi-ViT: Pushing the Limit of Vision Transformer Quantization**：[AAAI 原始論文頁](https://ojs.aaai.org/index.php/AAAI/article/view/28109)、[PDF](https://ojs.aaai.org/index.php/AAAI/article/download/28109/28222)、[作者 repository](https://github.com/YanjingLi0202/Bi-ViT)。
3. Zechun Liu et al., **ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions**：[原始論文](https://arxiv.org/abs/2003.03488)、[作者 repository](https://github.com/liuzechun/ReActNet)。
4. Xiaofan Lin, Cong Zhao, Wei Pan, **Towards Accurate Binary Convolutional Neural Network (ABC-Net)**：[NeurIPS 原始論文](https://papers.neurips.cc/paper/6638-towards-accurate-binary-convolutional-neural-network.pdf)。
5. Alex Henry et al., **Query-Key Normalization for Transformers**：[ACL Anthology 原始論文頁](https://aclanthology.org/2020.findings-emnlp.379/)、[PDF](https://aclanthology.org/2020.findings-emnlp.379.pdf)。
6. Steven K. Esser et al., **Learned Step Size Quantization**：[IBM Research／ICLR publication](https://research.ibm.com/publications/learned-step-size-quantization)、[arXiv](https://arxiv.org/abs/1902.08153)。
7. Minsoo Kim et al., **Understanding and Improving Knowledge Distillation for Quantization-Aware Training of Large Transformer Encoders**：[原始論文](https://arxiv.org/html/2211.11014)。
8. Zhexin Li et al., **Q-ViT: Fully Differentiable Quantization for Vision Transformer**：[原始論文](https://arxiv.org/abs/2201.07703)。
9. Bohan Zhuang et al., **Effective Training of Convolutional Neural Networks with Low-bitwidth Weights and Activations**：[原始論文](https://arxiv.org/abs/1908.04680)。
10. PyTorch／torchao 官方文件：[Quantization-Aware Training](https://docs.pytorch.org/ao/stable/workflows/qat.html)。
11. NVIDIA TensorRT 官方文件：[Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)、[TensorRT-RTX Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/work-with-quantized-types.html)。
