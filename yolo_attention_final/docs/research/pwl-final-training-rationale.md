# PWL Attention 最終訓練方法依據

本文件所有外部來源的查閱日期皆為 **2026-08-17**。

## 論述範圍

本文件記錄 YOLO26m PWL Attention 訓練配方的依據，並明確區分：

- **來源事實**：原始論文、PyTorch／Ultralytics 官方文件或原始碼直接陳述的內容。
- **專案推論**：針對固定 checkpoint、固定硬體 table、有限 recovery 預算與稽核要求所做的工程選擇。

這些來源不能證明本配方一定提升 COCO mAP；是否提升只由完整 COCO2017 validation 與預先宣告的 gates 判定。

## Float-PWL surrogate 與 Bit-True reference

### 為何需要可微分 surrogate

正式部署路徑包含 Q8.8 rounding、saturation、integer conversion 與離散 segment lookup。PyTorch autograd 只支援 floating-point 與 complex tensors 的梯度，不支援 integer tensors（[PyTorch `torch.autograd`](https://docs.pytorch.org/docs/stable/autograd)）。PyTorch 的 autograd definitions 對 `floor` 指定 zero derivative，而 fake quantization 則另有明確的 custom backward rule（[PyTorch `derivatives.yaml`](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml)）。因此，直接執行 Bit-True integer path，無法向上游 Q/K 提供有用的一般梯度。

Straight-through estimator（STE）不是離散 forward operation 的數學導數。Bengio、Léonard 與 Courville 將其描述為讓 gradient 穿過 hard stochastic／non-smooth unit 的 heuristic（[原始 STE 論文](https://arxiv.org/abs/1308.3432)）。PyTorch 也說明 QAT 會在 floating point 中模擬 quantization，並常因 rounding 不可微而採 STE（[PyTorch 官方 QAT 文章](https://pytorch.org/blog/quantization-aware-training/)）。所以 STE 必須被視為明確的 optimization approximation，不能假裝是 bit-accurate inference 的自然性質。

**專案推論：**訓練使用直接的 **Float-PWL surrogate**。固定 range `[-10, 0]`、20 segments、`delta=0.5` 與固定 endpoint values 都不變，但 gradient path 不做 Q8.8 rounding、integer cast 或 integer indexing。線性插值讓 relative-bias tables 與 Q/K 在 segment 內取得有限梯度；exact denominator 仍維持專案指定的 float reference。

這個 surrogate 不宣稱反向傳播能重現所有 integer rounding effect。訓練時以 finite／non-zero gradient tests 檢查，所有模型選擇再以實際 Bit-True checkpoint 驗證。

### 為何 Bit-True 路徑不暗藏 STE

STE 會刻意讓 backward semantics 不同於 hard operation 的真實 derivative（[Bengio 等人](https://arxiv.org/abs/1308.3432)）。若把 STE 藏在 Bit-True implementation，會混合兩項責任：

1. 驗證 Q8.8 rounding、saturation、table lookup 與 row normalization 的 reference oracle。
2. 訓練使用的 biased gradient estimator。

**專案推論：**Bit-True implementation 保持 reference-only，不使用 detach trick 或 custom STE。如此才能獨立測試 save/reload equivalence、endpoint packing、rounding boundaries 與 Float-to-Bit-True reconfiguration。模型選擇只認完整 5,000 張 COCO2017 val 的 Bit-True 結果；Float-PWL metrics 與 live weights 只作 diagnostics。

## EMA 與 non-floating state 同步

Ultralytics `ModelEMA.update()` 雖迭代 EMA `state_dict`，但只有 `v.dtype.is_floating_point` 才納入 interpolation，實作上透過 `lerp`／multiply-add 更新（[Ultralytics 官方 `ModelEMA` reference 與 source](https://docs.ultralytics.com/reference/utils/torch_utils/#ultralytics.utils.torch_utils.ModelEMA.update)）。因此 `current_epoch` 這類 integer buffer 只在 EMA 建立時複製，不會隨一般 EMA update 更新。

**專案推論：**只要 discrete control state 會影響 forward，就必須在 lifecycle boundary 明確由 live model 複製到 EMA，並以 regression test 驗證兩者一致。最終配方不採 progressive normalization，但仍保留此同步修正，以免 EMA `best.pt` 錯誤停留在 epoch-0 behavior。

EMA 同步與 freezing 是兩件事。`requires_grad=False` 只阻止 parameter gradient；BatchNorm 在 training mode 預設仍會維護 running estimates（[PyTorch `BatchNorm2d` 官方文件](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)）。因此所有非允許 buffer 都必須鎖定，且訓練前後 snapshot 要證明 running mean、variance 與 batch counter 未被意外改變。

## 多 seeds 與結果報告

Reimers 與 Gurevych 顯示，只改 random seed 就可能明顯改變 neural-network score，並主張報告多次執行的 distribution，而不是只報單一分數（[原始 score-distributions 研究](https://arxiv.org/abs/1707.09861)）。NeurIPS reproducibility program 也把可靠 workflow 與透明報告視為可重現 ML 研究的一部分（[Pineau 等人，JMLR](https://jmlr.org/papers/v22/20-303.html)）。PyTorch 則提醒，即使 seeds 相同，也不能保證跨 release、commit、platform 或 CPU/GPU 完全重現（[官方 reproducibility note](https://docs.pytorch.org/docs/stable/notes/randomness.html)）。

**專案推論：**原始正式規則要求 seeds 0、1、2，並報告 mean、sample standard deviation、minimum 與 maximum。只有三-seed mean 比 zero-train parent 至少高 `0.001` mAP50-95，訓練權重才能取代正式 fallback；這是預先宣告的工程 tolerance，不是 p-value。

後續依使用者明確指示停止 seed 1/2，只用已完成的 seed 0 結果。因此本輪不能宣稱多-seed 穩健改善，單一最高 checkpoint 只能標記為 best-observed。

## 分層解凍與兩個 sites 同步

Howard 與 Ruder 提出 gradual unfreezing，以減少 fine-tuning 對 pretrained knowledge 的破壞，並比較 last-layer、full 與 gradual-unfreezing strategies（[ULMFiT 原始論文](https://proceedings.mlr.press/v80/howard18a.html)）。該證據來自 language modeling，只支持設計模式，不保證 YOLO 的 effect size。

**專案最初配方：**先同步調整兩個 sites 的 decomposed bias，再加入 Q/K convolution 與 Q/K BN affine，最後解凍完整 `HardwareFriendlyAttention`。兩個 sites 在每階段始終一起解凍，避免產生不對稱部署狀態。每階段使用新 optimizer，只有通過 Bit-True gate 的 checkpoint 才能成為下一個 parent。

**後續 recovery：**在 Attention-only 結果沒有顯著改善後，依使用者授權擴大範圍，先解凍 Attention 所在 block，再加入 Neck/Detect、Backbone 最後 stage，最後測試 full model。越遠離 Attention 的層使用越低 LR，BatchNorm buffers 仍鎖定，每階段仍受 `0.001` rollback gate 約束。因此擴大 scope 並未放棄風險控制。

## 為何沒有採用其他方法

### 不使用 progressive Exact-to-PWL blend

PyTorch QAT 文章提到某個 LLM setup 延後 fake quantization 曾有幫助，但不能推導所有任務都需要 progressive schedule（[PyTorch 官方 QAT 文章](https://pytorch.org/blog/quantization-aware-training/)）。本專案 target approximation 與 parent 已固定，早期 integer epoch buffer 又曾造成 EMA/live mismatch。

**專案推論：**Float-PWL 從第一個 batch 就啟用，避免增加 schedule hyperparameter 與 discrete control state。Progressive state synchronization 只保留作 regression safeguard。

### 不學習 PWL endpoints

PACT 顯示 learnable clipping／quantization parameters 可以有效（[PACT 原始論文](https://arxiv.org/abs/1805.06085)），但不代表它符合本專案的固定 table contract。

**專案推論：**21 個 UQ1.15 endpoints、336-bit table、range 與 segment width 都是硬體規格，必須固定。若學習 endpoints，會把問題改成硬體 table search，並混淆改善究竟來自 Attention weights 還是 approximation redesign。

### 不使用 SWA

原始 SWA 會平均 cyclical／constant LR SGD trajectory 上的多個點（[Izmailov 等人](https://arxiv.org/abs/1803.05407)）。PyTorch implementation 也提醒 buffer handling 與平均後更新 BatchNorm statistics 的需求（[PyTorch `AveragedModel` 官方文件](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html)）。

**專案推論：**本專案是多個短而獨立、使用 AdamW 的 gated phases，已採 Ultralytics EMA。再加入 SWA 會新增 model averaging 與 BatchNorm policy，削弱 state-scope audit 的可解釋性。

### 不使用 Knowledge Distillation

Knowledge distillation 透過 teacher 或 ensemble 的 softened outputs 將行為轉移給 student（[Hinton、Vinyals 與 Dean](https://arxiv.org/abs/1503.02531)），因此會新增 teacher、temperature／loss weighting 與第二個 training target。

**專案推論：**本實驗要測量固定 PWL surrogate 與受控權重 recovery 的效果。KD 會改變研究問題，使改善來源難以歸因，所以本輪不使用。

## 最終方法摘要

- Training 只走固定 Float-PWL interpolation；Bit-True reference 不藏 STE。
- 所有 gate 與正式選擇只使用完整 COCO2017 val 的 Bit-True EMA checkpoints。
- 明確同步 EMA non-floating control state，並鎖定所有不允許變動的 BatchNorm buffers。
- 先測 block LR x1/x2/x4，再分層擴大到 Neck/Detect、Backbone-last 與 full model；各層採遞減 LR。
- 每階段 early stop，child 相對直接 parent 退化超過 `0.001` 就 rollback。
- 固定 endpoints、PoT coefficients 與 denominator reference；不使用 KD、SWA 或 learnable endpoints。
- 因只完成 seed 0，最高 trained checkpoint 只能列為 best-observed，正式 winner 保留 zero-train fallback。

## 主要來源索引

以下來源皆於 2026-08-17 查閱：

- Bengio、Léonard、Courville，*Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation*：<https://arxiv.org/abs/1308.3432>
- Choi 等人，*PACT: Parameterized Clipping Activation for Quantized Neural Networks*：<https://arxiv.org/abs/1805.06085>
- Hinton、Vinyals、Dean，*Distilling the Knowledge in a Neural Network*：<https://arxiv.org/abs/1503.02531>
- Howard 與 Ruder，*Universal Language Model Fine-tuning for Text Classification*：<https://proceedings.mlr.press/v80/howard18a.html>
- Izmailov 等人，*Averaging Weights Leads to Wider Optima and Better Generalization*：<https://arxiv.org/abs/1803.05407>
- Pineau 等人，*Improving Reproducibility in Machine Learning Research*：<https://jmlr.org/papers/v22/20-303.html>
- Reimers 與 Gurevych，*Reporting Score Distributions Makes a Difference*：<https://arxiv.org/abs/1707.09861>
- PyTorch autograd 官方文件：<https://docs.pytorch.org/docs/stable/autograd>
- PyTorch autograd derivative source：<https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml>
- PyTorch `BatchNorm2d` 官方文件：<https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>
- PyTorch QAT 官方文章：<https://pytorch.org/blog/quantization-aware-training/>
- PyTorch reproducibility 官方說明：<https://docs.pytorch.org/docs/stable/notes/randomness.html>
- PyTorch `AveragedModel` 官方文件：<https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html>
- Ultralytics `ModelEMA` 官方 reference／source：<https://docs.ultralytics.com/reference/utils/torch_utils/#ultralytics.utils.torch_utils.ModelEMA.update>
