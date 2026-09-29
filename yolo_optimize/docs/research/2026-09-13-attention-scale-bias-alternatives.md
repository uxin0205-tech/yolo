# BinaryQK 是否做好？固定 scale／bias 與替代路線

2026-09-13。本文區分實測、程式稽核與研究提案。**目前沒有新的 scale 校準後 AP、原生 QK 重訓 AP 或 INT8 部署結果。** 本次先交付清楚報告；既有正式權重不變。

## 1. 結論先說

BinaryQK 目前的固定尺度前向與修復過的訓練梯度有 CPU 證據，但沒有足夠證據說它已達到最好的精度。最值得再查的是固定尺度有沒有適應後期融合／activation 的特徵，以及它和相對位置偏置的比例；不是直接新增更多二值分支。

同時保留另一條路：移除二值與額外補償、恢復原生 QK 點積，但保留使用者已確認的 PWL [-10,0]／20 段，再恢復訓練。這才回答「二值化是否限制目前效果」。

## 2. 這次真正查到了什麼？

安全重建目前 qSiLU E2，SHA 固定為 `1bbbbe0f87cb853827af1bb625e7066178e3fb2c7e14c410d997efb24e94834a`。兩個位置皆 4 個 head、Q/K 每 head 32 維、V 每 head 64 維。

| 檢查 | 結果 | 能說明什麼／不能說明什麼 |
| --- | --- | --- |
| 二值 XNOR 結果對 signed-dot | 合成 CPU 測試精確一致 | 確認這組前向算式，不是 AP 勝出的證據 |
| 固定 scale | registered buffer、requires_grad=False | 普通 optimizer 不會學它；需明確校準或專用訓練設計 |
| gamma 改為 7 倍 | 固定模式輸出精確不變 | 只訓 gamma 無法調整當前正式 score |
| 修復版 surrogate | Q/K 梯度有限且非零，gamma 無梯度 | 合成測試確認梯度路徑，不代表真實資料訓練已最優 |
| 相對位置偏置 | 兩個 1D 表 table_y、table_x，每處合計 504 個參數 | 已有分解式結構；不是一個普通 scalar bias |
| 目前 bias 表 RMS | 約 2.25–2.60 | 只是參數統計，不能直接說 bias 已壓過內容分數 |
| 目前固定二值 scale | 共 16 個，15 個 0.25、1 個 0.125 | 不逐圖選取，不是尚未實施的 8-scale codebook |

完整原始證據：[CPU 稽核 JSON](<../../experiments/inference/component_switch_v1/artifacts/binary-audit-v1.json>)與[可重現程式](<../../experiments/inference/component_switch_v1/audit_binary.py>)。腳本輸出的 `native_scale_exact_power_of_two` 是浮點 hex 字串欄位，**不是「該常數可精確用單一 shift 表示」的布林判斷**；實際 1/√32 不是二的整數次方。

實作來源是本機 `yolo_attention/binary_basis.py`、`relative_bias.py`、`normalization.py`，以及本研究 `qk_challenger.py`。固定模式 `_coefficient` 最後回傳 fixed buffer，即使先計算過 dynamic coefficient 也不會使用它；固定 scale 早返回可省無用 reduction，但那是程式成本優化，不會憑空提高 AP。

後期融合的 `stage_policy.py` 明列 Q/K 投影與 score.gamma 為 hardware-frozen；所以「融合前修好 surrogate」不能等同「後期 qSiLU 訓練一直在微調 Q/K」。移除二值後重訓必須重新確認可訓練參數，不可沿用這項硬體凍結規則。這是政策程式稽核，仍須配合每次 resolved config／state 差異才可描述完整歷史更新。

## 3. scale 為什麼可能仍有改善空間？

每個 head 的分數可寫成：

```text
Z0 = sign(Q)ᵀ sign(K)
Z1 = sign(HQ)ᵀ sign(HK)
S  = c0 × Z0 + c1 × Z1 + B(位置)
P  = PWL_normalize(S − rowmax(S))
```

H 是 Hadamard 轉換。c0／c1 對整張影像的所有位置共用，不需要每張圖重算。尺度越大，內容分數差距通常越大；經正規化可能變得更集中，也可能更容易碰到 PWL 下界。太小則可能讓內容差異相對偏置變弱。這是機制推導，不是已量測的退化原因。

目前後續融合與 activation 訓練使用固定 buffer，並不會因 Q/K 分布改變自動重估。因此「有固定 scale」不等於「已校準到這個最後 checkpoint 的最佳 scale」。但也不能因 15 個係數相同就判定校準錯誤，PoT rounding 本來可能得到相同值。

必要的下一步：從兩個正式 train 資料流記錄每 head 的內容分數範圍、row-centered bias 大小、PWL 飽和比例、Q/K sign 飽和／梯度比例；不重新切分資料，不用 formal val 反覆搜索係數。

可比較的尺度模型，依成本由小到大：

| 選擇 | 硬體 | 精度上的假說 | 狀態 |
| --- | --- | --- | --- |
| 目前 PoT 係數 | 每分支固定 shift | 最簡單但係數格點較粗 | 已使用 |
| 同樣 16 個 PoT slots 重新校準 | 不增加分支或 selector | 對齊最後模型分布 | 待試 |
| 16 個固定 dyadic 係數 m/2^r | 固定整數乘法＋shift，可限制為少數 shift-add | 比單一 PoT 更細，例如 3/16 位於 1/8 與 1/4 之間 | 待試 |
| 每張圖動態 scale | 需線上 reduction／控制 | 更能追隨輸入，但違背目前簡化方向 | 不優先 |

若讓 scale 可訓練，必須真的接到 score 的前向與反向，不是更新一個不生效的 gamma。LSQ 提供「讓量化步長接受 task loss 梯度」的方法先例，但它不是本專案 BinaryQK 的現成修正，不能直接挪用論文精度作保證。[LSQ 原論文](https://arxiv.org/abs/1902.08153)

可用 train-only 的 row-centered 分數誤差檢查尺度初始化，也應加入 task loss 驗證；最小化相對 FP-QK 的誤差只是一種 proxy。最後學生 Q/K 本身已適應二值化，不能把其 FP-QK 分數當成必然更好的 teacher。

## 4. bias 怎麼判斷，而不是亂加？

目前 `B(i,j)=by(Δy)+bx(Δx)`。它可以幫助位置關係，但不能還原 Q/K 幅度的全部資訊。位置表的效果與内容 scale 相互作用，必須看真實資料的 row-centered 分布、PWL 截斷比例及 AP。

一個常被誤用的方式是加「每個 head 一個常數 bias」。對同一 row 加常數 b，`(S+b)−max(S+b)=S−max(S)`，在這份 PWL 實作也抵消，因此不能用它改變注意力分布。可研究的是既有相對位置表、它的限制或固定 head-wise gain，而不是這種會抵消的常數。

最小對照應先只改尺度，再判斷是否需要短訓 scale＋既有 bias。保留兩個位置、兩条分支，不增加 dense N×N 可學表。不一次把 scale、bias、activation 與 MASF 都改掉。

## 5. 移除 BinaryQK 後，原生點積也能硬體友善嗎？

可以設計成固定係數與低位元乘加，但要區分三層：

```text
精度參考：FP32 Q/K → QᵀK × (1/√32) → PWL
GPU 候選：FP16 Q/K → 矩陣乘法，累加策略依實際 kernel → PWL
整數候選：INT8 Q/K → INT32 累加 → 固定 requant → PWL
```

`1/√32 = 0.1767766952966369` 是編譯時常數，**無需逐圖 sqrt／除法器**。例如 `181/1024=0.1767578125` 的常數相對誤差約 -0.01068%，可以固定乘法加右移實作；這只是係數誤差，不是 AP 降幅，PWL 邊界與矩陣點積误差仍需實測。常數也可併入部署量化乘數，或在適用且驗證等價時併入投影／BN 折疊，不在訓練時任意縮放會改變優化動態的參數。

固定對稱量化的例子：`Q≈sQ×Qint`、`K≈sK×Kint`，則 `S≈(sQ×sK/√32)×sum(Qint×Kint)`。sQ／sK 可離線由 train 校準，推論只讀常數，不逐圖量測。對稱 INT8 限定 [-127,127] 時，32 項點積最大幅度為 `32×127²=516128`，有號 20 位可容納點積本身，工程上先用 INT32 累加；後續固定乘數、rowmax 相減與 PWL 格式轉換仍須另外做位寬分析，不能把 20 位當全部路徑足夠。

INT8 保留多個幅度等級，比 sign-only 保存更多資訊，但不保證 AP 一定高。NVIDIA 文件說明對稱量化與 scale 的用途；I-ViT 示範整數矩陣運算及 dyadic 路徑，結果僅適用其驗證模型。自訂 PWL 不一定可直接套用 TensorRT fused attention kernel，per-head activation scale 的支援也需依實際後端確認。[TensorRT 量化文件](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)、[I-ViT 原論文](https://arxiv.org/abs/2207.01405)

FPGA 不必為了省乘法而強迫使用 BinaryQK。具 DSP 的 FPGA 可以做固定點乘加；例如 DSP48E2 提供固定寬度乘法及累加資源。但能放多少並行路徑、實際時脈／能耗取決於板型與實作，不能從通用規格推定本模型速度。[AMD DSP48E2 文件](https://docs.amd.com/r/2021.2-English/ug1483-model-composer-sys-gen-user-guide/DSP48E2)

PWL 依使用者要求保持 [-10,0]、20 段，每段寬 0.5；目前 BitTrue exp 插值使用 Q8.8 分數／UQ1.15 端點，最後正規化仍是 exact software reciprocal reference，**不能宣稱全 Attention 已沒有除法或全整數**。

## 6. 還有什麼架構能簡化？

以下優先次序是依本專案限制的推論，不是論文保證本模型改善：

| 方向 | 簡化哪裡 | 保留原本全域 PWL 語意？ | 本專案優先度 |
| --- | --- | --- | --- |
| 固定尺度 INT8 QK | 降低每次乘加位寬、保留幅度 | 近似保留，須校準／可能 QAT | 高，浮點重訓參考建立後再試 |
| 少量 head 混合位寬 | 敏感 head 多給位元，其餘減少 | 近似保留，但排程更複雜 | 有逐 head 敏感度證據才做 |
| 局部 window attention | 減少互相比較的 token 數 | 可保留 PWL 算子，但改了連接範圍 | 第二輪，可能傷害遠距情境 |
| K/V 降採樣、Q 保持原尺寸 | 縮短 K/V 序列 | 可保留 PWL，但注意力目標數改變 | 第二輪，小物件資訊需驗證 |
| Linear attention | 改寫聚合順序，避開完整 N×N attention | 通常不等同原 Softmax／PWL | 目前不優先，違背保留既有語意的最小更動 |
| 新增更多二值基底 | 多分支近似補資訊 | 仍可接 PWL，但運算／存取也增加 | 不优先，先檢查已有兩分支 |

Swin 的窗口與移位設計是減少全域配對的原始先例，移植不是直接換一個算子。[Swin 原論文](https://arxiv.org/abs/2103.14030)

此處 Linear attention 指 MIT Han Lab／Cai 等人的 ICCV 2023 EfficientViT 路線，不是同名 CVPR 2023 cascaded group attention；它使用輕量線性 attention 處理高解析度 dense prediction，並非保留現有 PWL 的等價改寫。[EfficientViT 作者論文](https://arxiv.org/abs/2205.14756)

固定 N 個 tokens、head 維度 d 下，完整 QK 配對約 O(N²d)，窗口每窗 w 個 tokens 約 O(Nwd)，K/V 剩 M 個 tokens 約 O(NMd)。這是配對算量關係，不是整網加速倍數；QKV 投影、V 聚合、卷積和記憶體仍存在。

## 7. 報告交付後的最小實驗順序

1. **已完成**：目前模型身份核對、scale／gamma CPU 稽核、四組同權重 MASF／QK 切換診斷。沒有偷偷換掉預設模型。
2. **先做低成本診斷**：train-only 統計與固定 scale 候選；先比較現有 PoT、重估 PoT、少量 dyadic；保持 bias／MASF 不變。鎖定方法後完整 val，不能用 val 不斷挑係數。
3. **只在有證據時調 bias**：若 scale-only 有進展或偏置／內容比例確有異常，新增一組 scale＋既有 bias 的短恢復，與同預算不改方法的 control 配對。
4. **必要重訓對照**：目前 BinaryQK 同預算 control，對上移除 BinaryQK／Hadamard／額外 bias 的浮點 QK＋PWL；兩者相同父權重、資料、回合、optimizer、MASF、qSiLU。warmup 1，暫定 20 epoch 上限，LR 及停止規則須實作與 smoke 後鎖定。
5. **獨立確認與成本**：各自選定 checkpoint 重新驗證 COCO overall／person、BBAT box／pose／ball／bat，再做 MASF α 開／關；只有恢復路線有效才進 INT8 QK 校準／QAT。精度與成本都列，沒有板端測試就不寫板端改善。

此順序避免把「多訓練了一些」誤歸因為新 scale 或新 Attention。新的 CPU 稽核證明哪些參數實際生效，但沒有證明可以回復先前約 0.01 的 AP 差；需要上述真正的配對重訓。
