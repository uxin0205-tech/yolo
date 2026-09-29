# YOLO26 Neck RepConv 與 BinaryQK 精度恢復研究

- 日期：2026-08-31
- 範圍：`~/yolo` 現行 YOLO26、既有 YOLO11m BinaryQK、MASF 結果，以及 RepConv／BinaryAttention 第一手資料
- 研究問題：是否先只替換 P3／P4 Neck 的少數 `3×3 Conv`；BinaryQK 精度為何下降、應如何補回
- 證據規則：外部只採論文、作者官方 repository／原始碼與框架官方原始碼；本地結果不與文獻結果混為一談

> **2026-09-02 更正：**後續 cleanup 已刪除 W-DIR `best.pt`，且 training audit 確認 W-DIR 本來就是
> 全模型解凍的 direct recovery，不能再從它 resume 或把「擴成 full-model」當新實驗。新增的本地
> per-token probe 與 hardware-first 約束也改變了候選順序。既有 V1-DYN／V1-SHEAD／V1-P2
> 已完成 scale 消融，因此不重跑 global dynamic；首輪只新增兩個 fixed-PoT site-isolation validation。
> 可執行的最新結論以
> [BinaryQK 精度恢復方向](<../../proposals/binaryqk-accuracy-recovery/README.md>)與
> [最小實驗計畫](<../../proposals/binaryqk-accuracy-recovery/plan.md>)為準。

## 結論先行

1. **RepConv 值得做，但第一輪只做兩個 outer stride-2 `3×3 Conv`。** 現行 YOLO26 最乾淨的候選是 head layer 17（P3→P4 下採樣）與 layer 20（P4→P5 下採樣），先跑 `R0=原 Conv`、`R1=17`、`R2=20`、`R3=17+20`。這兩層不在 C3k2 的 residual／concat 內，且因 `stride=2` 自然沒有 identity branch，較接近 YOLOv7 所稱的 RepConvN。這是**工程推論與待驗證假說**，不是已有論文證明「P3/P4 一定最佳」。
2. **不要把 RepConv 全面替換。** RepVGG 證明的是「訓練多分支可在部署前精確融合為單一 `3×3`」，不是任意網路全面替換都會增準；YOLOv7 更直接報告，將帶 identity 的 RepConv 直接套到 residual／concat layer 會破壞預期的 gradient diversity，因而提出無 identity 的 RepConvN。先碰 layer 17/20，可把初始化、拓撲與 pretrained-weight 擾動控制在最小範圍。
3. **局部 RepConv 的主要價值是訓練時的結構過參數化，不是部署 FLOPs 下降。** 原本單一 `3×3 Conv+BN` 與 RepConv 融合後仍是單一 `3×3 Conv`；若 backend 都能完成 Conv-BN fusion，理論運算量大致相同。是否有準確率收益，以及實際 latency 是否持平，均須實測。
4. **「BinaryQK 掉很多」需分世代與訓練階段。** 舊 YOLO11m 的 COCOeval zero-train sign 由 `0.510850` 降至 `0.459297`（`-0.051554`），但 10-epoch attention-only QAT 的最佳 binary 結果已到 `0.511841`，相對 FP T0 `0.512671` 只差 `0.000830`。現行 YOLO26 則由 B26-FP `0.517998` 降到最佳 W-DIR `0.507457`，約差 `0.010540`，問題仍未完全解決。兩套 lineage、attention sites、evaluator 與 recipe 不同，不能互相代替。
5. **現行 YOLO26 BinaryQK 最值得先做的是 fixed-PoT site isolation。** 正式 A-FINAL 使用 fixed per-head/basis PoT coefficient；既有 V1-DYN／V1-SHEAD／V1-P2 已完成 scale 消融，global dynamic 只比 PoT 高 `0.000506`，不足以優先承擔每張 image 重算 scale 的硬體成本。首輪只做 site-10-only、site-22-only 兩個免訓練 validation；per-token dynamic 降為條件式 accuracy ceiling。
6. **MASF 不應與 RepConv 同時移除。** 既有 COCO 與 BBAT5 結果未證明 MASF 帶來 material accuracy gain，但目前正式 Full35 J3 checkpoint 的 `graph.model.16.p3_masf.alpha=0.110659`，表示現有權重已適應該分支。RepConv 第一輪應從目前 Full35 J3 parent 出發並固定 MASF／BinaryQK；另以 `alpha=0` 的 zero-train full validation 判斷 MASF 依賴，再決定是否做移除後 recovery。
7. **RepConv 與既有量化工作有交互風險。** RepConv 必須先 fuse，再重跑 deployment catalog、calibration 與 matched PTQ gate；不能用 FP32 融合等價直接推定 INT8 也等價。QARepVGG 的第一手結果顯示，結構重參數化後的 weight／activation 分布可能放大量化誤差，因此量化驗收是首輪 RepConv 的必要停止條件。

---

## 1. 證據類型與解讀邊界

本文每項關鍵判斷使用以下標籤：

- **【文獻事實】**：論文、作者官方 repository 或框架官方原始碼直接支持。
- **【本地證據】**：`~/yolo` 內的設定、程式、manifest 或實驗報告直接支持。
- **【工程推論】**：由文獻與本地結構推導出的設計判斷，尚未由本 repo 的配對實驗驗證。
- **【待驗證假說】**：明確列入消融、可被結果否證的主張。

尤其要避免三個誤讀：

- RepVGG 的等價融合不等於「全面 RepConv 一定比局部 RepConv 準」。
- BinaryAttention 在 DeiT/ImageNet 的收益不等於能直接複製到 YOLO26 detection。
- 單 seed 的千分位差異不等於穩定增益。

## 2. 本地架構與既有結果

### 2.0 `~/yolo` 現行工作線與正式整合模型

**【本地證據】** 目前各資料夾不是同一層級的「候選模型」，而是不同階段與 lineage：

| 路徑 | 角色 | 目前判定 |
|---|---|---|
| `yolo_attention/` | YOLO26m 兩個 BinaryQK sites 與 normalization/bias 實驗 | A-FINAL 仍比本地 FP 低 `0.011641`；W-DIR 是較佳 accuracy parent |
| `yolo_achitechure/achitechure_1/` | YOLO26m P3 MASF Full35／Partial75 | 未證明 material accuracy gain；Partial75 工程上較省，但不是穩定 accuracy winner |
| `yolo_achitechure/achitechure_2/` | Full35 C3k2 簡化 C1–C3 | Float20 掉點過大，已封存，不應列為現行候選 |
| `yolo_combine/final/full35/` | 現行正式 shared-trunk Detect＋Pose 整合 | 目前應作 deployment／後續優化的 authoritative parent |
| `yolo_activation/` | SiLU 與非 SiLU activation | accepted SiLU 為主 parent；完成 10-epoch 的 `qsilu_pq` 可作量化 fallback |
| `yolo_quantize/` | Full35 weight/activation quantization | V0–V3 只有 CPU 靜態分析；尚無量化後 GPU mAP、QAT 或正式 winner |

Full35 的 runtime 關係是：

```text
input
  └─ shared YOLO26m layers 0–22（只執行一次）
       ├─ BinaryQK attention：model.10.m.0.attn
       ├─ P3 MASF：model.16.p3_masf
       ├─ BinaryQK attention：model.22.m.0.1.attn
       └─ P3／P4／P5 features
            ├─ COCO80 Detect head（應用層取 person）
            └─ BBAT5 Pose26 head（ball、bat；kpt_shape=[2,3]）
```

它共享 `26,529,701` parameters；兩個獨立模型合計為 `45,580,762`，少 `41.796%`。正式 J3 `best_joint` 的 Bit-True joint score 是 `0.711174738975`；COCO overall box、BBAT box、BBAT pose mAP50-95 分別為 `0.498022`、`0.630036`、`0.903717`，八項 gate 全部通過。這些值來自 `/home/uxin/yolo/yolo_combine/final/full35/analysis/FINAL_ANALYSIS.md`，不可和單任務 `yolo_attention` 的 COCO baseline 直接作 parent/child 差值。

**【工程推論】** 因此「優化目前模型」有兩種合法邊界：若目標是改善 BinaryQK 演算法，沿用 `yolo_attention` 的單任務 matched FP/Binary lineage；若目標是改善現行部署模型，則從 Full35 J3 `best_joint` 出發，每輪只改一個因子，並保留 Detect＋Pose 八項 gate。本文的 RepConv 首輪採後者。

### 2.1 現行 YOLO26 Neck：第一輪應鎖定 layer 17、20

**【本地證據】** 現行 YAML 為：

`/home/uxin/yolo/yolo_p2/ultralytics/cfg/models/26/yolo26.yaml`

對應的官方 Ultralytics YAML 為 [Ultralytics `yolo26.yaml`](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/26/yolo26.yaml)。與本題直接相關的 head 路徑是：

| layer | 本地語意 | 現行模組 | 第一輪處理 |
|---:|---|---|---|
| 16 | P3/8-small feature | fusion/refinement block | 不動 |
| 17 | P3→P4 outer downsample | `Conv[256, 3, 2]` | R1：換 RepConv |
| 19 | P4/16-medium feature | fusion/refinement block | 不動 |
| 20 | P4→P5 outer downsample | `Conv[512, 3, 2]` | R2：換 RepConv |
| 22 | P5/32-large feature | fusion/refinement block | 不動 |

**【工程推論】** 口語上的「換 P3/P4 Neck」容易有歧義。本文把首輪候選明確定義成「以 P3、P4 feature 為輸入的兩個 outer stride-2 `3×3` transition」，即 layer 17 與 20；不是把 layer 16/19 的整個 C3k2 換掉，也不是把 C3k2 內所有 `3×3` 全換掉。實驗 manifest 必須記錄完整 `named_modules()` path、輸入／輸出 channel、stride、groups 與來源 checkpoint，不能只寫「P3/P4」。

### 2.2 本機已具備 RepConv 與 fuse 邏輯，但不能只改 YAML 名稱

**【本地證據】** 本機 Ultralytics 實作位於：

- `/home/uxin/yolo/yolo_p2/ultralytics/nn/modules/conv.py`
- `/home/uxin/yolo/yolo_p2/ultralytics/nn/tasks.py`

`RepConv` 的 training graph 是 `3×3 ConvBN + 1×1 ConvBN + optional identity BN`，`get_equivalent_kernel_bias()`／`fuse_convs()` 可轉成單一 `3×3`。identity 僅在 `c1 == c2 && stride == 1` 時存在，因此 layer 17、20 的 `stride=2` 會自然成為無 identity 的型態。`model.fuse()` 已處理 RepConv。這與 [Ultralytics 官方 `conv.py`](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/nn/modules/conv.py) 及 [Ultralytics 官方 `tasks.py`](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/nn/tasks.py) 一致。

**【本地證據】** 現行 `parse_model()` 的 `base_modules` 未列入 RepConv。因此，不能假設把 YAML 中的 `Conv` 字串直接改成 `RepConv` 就會正確解析 channel；實作須採 graph-aware graft、受控 custom wrapper，或先補上有測試的 parser 支援。

**【工程推論】** 若從 pretrained Conv parent 開始，graft 必須在 epoch 0 保持函數：

1. 使用未 fuse 的 source checkpoint；把原 `3×3 conv`、BN running statistics、BN affine 與 activation 完整複製到 RepConv 的 `3×3` branch。
2. 新增 `1×1` branch 的輸出初始化為零，例如 BN `gamma=0`、`beta=0`；不能保留隨機分支後就宣稱是同一 parent。
3. 保留原 activation 的種類與位置；不能把 YOLO26 的 activation 暗中換成另一個 RepVGG recipe 的 ReLU。
4. 訓練前比較 graft 前後 eval-mode 輸出；部署前再比較 branch graph 與 fused graph。

推薦門檻是 FP32 `max_abs_error <= 1e-5`，若未達標就停止該 run。此門檻是工程驗收值，不是論文定律。

**【本地驗證】** `scripts/audit_repconv_seams.py` 已直接使用目前 vendored Ultralytics 實作完成 CPU 檢查：

- YAML 的 layer 17／20 分別確實是 `Conv[256,3,2]`／`Conv[512,3,2]`。
- 把原 Conv／BN 完整搬到 RepConv `3×3` branch，並將新增 `1×1` branch 的 BN `gamma/beta` 歸零後，初始輸出 `max_abs=0`。
- 再執行 `fuse_convs()`，相對原 Conv 的 `max_abs=1.073e-6`，通過 `1e-5` gate。
- layer 17／20 訓練期相對原 Conv 各增加 `66,048`／`263,168` parameters，合計 `329,216`；兩者因 stride 2 都沒有 identity branch。
- 這些額外參數在部署融合後消失，deploy graph 仍是原 shape 的單一 `3×3 Conv+bias`。因此它沒有降低相對原 Conv 的部署 MACs，只提供 training-time over-parameterization。

正式 Full35 checkpoint 的 layer 17／20 tensor shape 也已用 `torch.load(..., weights_only=True)` 核實為 `(256,256,3,3)` 與 `(512,512,3,3)`，可按上述方式承接，不需要重設原分支。

### 2.3 MASF：目前沒有 accuracy gain 證據

**【本地證據一：YOLO11m BBAT5 clean study】** `/home/uxin/yolo/yolo_masf/clean_bbt5_study/results/REPORT.md` 的兩 seed strict-fair validation 顯示：

| 變體 | mAP50-95 mean | 相對 B0 |
|---|---:|---:|
| B0-Clean | 0.460696 | 0 |
| P2-PaperFormula-Clean（最佳 MASF） | 0.452507 | -0.008189 |
| P3-Partial25-Clean | 0.449535 | -0.011161 |

B0 與最佳 MASF 的差小於各自 seed standard deviation，因此可說「B0 目前較穩健」，不能聲稱已證明 MASF 普遍有害。P2-Control-Full 是 `0.492881`，但它承接不同 checkpoint／兩階段 schedule；這反而顯示最佳化契約會強烈影響 MASF，不能拿它宣稱架構公平勝出。

**【本地證據二：YOLO26m COCO architecture_1】** 正確結果位於 `/home/uxin/yolo/yolo_achitechure/achitechure_1/results/rtx5060ti-half-0822.md` 與 `results/results-template.csv`：

| 變體 | COCO mAP50-95 | 相對 A0 |
|---|---:|---:|
| A0 immutable | 0.506737 | 0 |
| Full35 A2 | 0.506391 | -0.000346 |
| Partial75 A2 | 0.506754 | +0.000018 |

這三者在 `0.001` tie band 內；完整資料 Phase B 又分別比各自 A2 低 `0.002889`／`0.002745`。可支持的結論是「沒有 material accuracy gain」，不是「Partial75 已穩定勝出」。

**【本地證據三：目前正式 Full35】** MASF 實際 graft 在 `model.16` 的 C3k2 輸出後：

`x + alpha * Conv1x1(DW3(x) + DW5(x))`

`torch.load(..., weights_only=True)` 對正式 J3 `best_joint` 檢查得到 `graph.model.16.p3_masf.alpha=0.1106591076`，並非接近初始化的 `0.01` 或零。它不能證明 MASF 有益，卻證明目前 checkpoint 沒有完全忽略該分支。

**【工程推論】** 安全順序應是：`M0=目前 checkpoint`、`M1=只將 alpha 設為 0 的完整 zero-train validation`、若 M1 仍在八項 gate 內，再做 `M2=移除 MASF 後的 matched low-LR recovery`。RepConv 第一輪固定 MASF／BinaryQK 不動；直接改用另一個 A0/no-MASF parent 會同時更換 task lineage 與既有 shared-model 適應狀態，反而失去歸因能力。

### 2.4 舊 YOLO11m BinaryQK：zero-train 大跌，但 QAT 已補回大部分

**【本地證據】** 舊 lineage 的摘要在 `/home/uxin/yolo/yolo_binaryqk/README.md`：

| 變體 | 設定 | COCOeval mAP50-95 | 對同階段 FP 差 |
|---|---|---:|---:|
| E0 | FP zero-train | 0.510850 | — |
| E1-S | raw sign Q/K | 0.459297 | -0.051554 |
| E1 | scaled sign Q/K | 0.480561 | -0.030289 |
| E2-DUAL | dual basis zero-train | 0.495645 | -0.015205 |
| T0 | FP，10-epoch control | 0.512671 | — |
| T4 | full residual dual，attention-only QAT | 0.511631 | -0.001040 |
| T6-F/A | feature + attention KD | 0.511841 | -0.000830 |

**【本地證據】** 該模型只置換 `model.10.m.0.attn` 一個 attention module；formal training 凍結約 19.91M 參數，只訓練約 0.20M attention 參數，10 epochs、single seed。其 quantizer 是 fake quantization，不是實際 bitwise kernel。

本地原始 Ultralytics metric 的正式 selector 是 T6-O/F（T6），相對 T0 差 `0.00048`；上表則固定使用 COCOeval，最佳 binary 是 T6-F/A，相對 T0 差 `0.000830`。兩個 evaluator 的排名細微不同，報告時不可混成同一欄。

**【工程推論】** 若「下降很多」指 E1-S/E1，答案是成立的；若指 T4/T6 family，則既有證據顯示 QAT 已補回絕大部分，但單 seed 的千分位差距不能視為已完全解決，也不能外推到 YOLO26。

### 2.5 現行 YOLO26 BinaryQK：主要缺口仍約 0.0105 mAP

**【本地證據】** 現行結果在 `/home/uxin/yolo/yolo_attention/reports/REPORT.md`：

| 實驗 | mAP | 本地可支持的解讀 |
|---|---:|---|
| B26-FP | 0.517998 | 現行 FP parent |
| I-SCR identity basis | 0.492482 | 已二值化的 identity-basis screen，不是 FP control |
| Hadamard | 0.495243 | 兩 basis 不是自動解法 |
| W-DIR | 0.507457 | 目前較佳，40 epoch 中完成 26、best epoch 21 |
| W-PROG | 0.502866 | progressive 未勝 direct |
| dynamic scale | 0.507457 | 優於 PoT 0.506952 |
| no bias | 0.505416 | bias 有小幅幫助 |
| dense bias | 0.506877 | 優於 no bias |
| decomposed bias | 0.506658 | 接近 dense |
| N0 exact | 0.506658 | normalization 不是主要剩餘瓶頸 |
| SHIFT | 0.506746 | 與 exact 幾乎相同 |
| final N1-SHIFT | 0.506357 | 未超過 W-DIR |

另外，獨立 PWL 驗證的 Exact／Float PWL／Q8.8 Bit-True PWL 分別為 `0.506658`／`0.506730`／`0.506737`，彼此遠小於 `0.001`。因此目前約 `0.0116` 的 A-FINAL 對 FP 缺口不是 PWL 或 SHIFT normalization 造成；主要問題仍在 Q/K 表示、score ranking 與適應訓練。

**【本地證據】** 實作位於：

- `/home/uxin/yolo/yolo_attention/src/yolo_attention/binary_basis.py`
- `/home/uxin/yolo/yolo_attention/src/yolo_attention/attention.py`

現行 clipped STE 只在 `|x| <= 1` 有梯度；Hadamard 使用兩個 bases。程式的 dynamic path 以
per-sample/head global magnitude計算 coefficient，但正式 V1-BR／A-FINAL 在 calibration 後改用
fixed per-head/basis PoT coefficient `[1,H,1,1]`。量化只改 Q/K score 與 normalization，V、
`P@V + PE` 及 projection 保留。正式兩個 sites 是：

- `model.10.m.0.attn`
- `model.22.m.0.1.attn`

目前沒有 one-site ablation。兩者都處理 P5/32：site 10 位於 top-down fusion 上游，site 22 更靠近 P5 output；後一句是依 graph 位置做的**工程推論**。

**【工程推論】** 歷史 scale ablation 的 accuracy 最高點是 V1-DYN `0.507457`，但其 W-DIR parent
checkpoint 已刪除；retained V1-BR 是 power-of-two scale、decomposed bias、exact normalization。
V1-P2 `0.506952` 只低 `0.000506`，在既定 `0.001` tie band 內，因此 deployment-first 應保留 fixed PoT。
不同單因子 run 的最好數字不能拼成未驗證的聯合 parent。

## 3. RepConv：第一手資料支持什麼、不支持什麼

### 3.1 結構重參數化的確切能力

**【文獻事實】** [RepVGG 論文（CVPR 2021）](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html) 將 training-time 的 `3×3 + 1×1 + identity` branches 透過 BN fusion 與 kernel padding，相加成 deployment-time 的單一 `3×3` convolution。[作者官方 RepVGG 實作](https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py) 提供 `get_equivalent_kernel_bias()` 與 `switch_to_deploy()`，並示範轉換前後輸出等價檢查。

**【文獻事實】** identity branch 不是任意 layer 都有；作者實作只有在 channel 相同且 stride=1 時才建立 identity BN。這與本機 Ultralytics RepConv 的條件一致。

**【工程推論】** layer 17、20 都是 stride=2，因此其 RepConv 沒有 identity；訓練增加一條 `1×1` branch，部署仍融合回單一 `3×3`。這是拓撲上最簡單、也最容易做 function-preserving graft 的候選。

### 3.2 為何不應全面替換

**【文獻事實】** [YOLOv7 論文](https://arxiv.org/pdf/2207.02696) 的 planned re-parameterized convolution 章節明確指出，將含 identity 的 RepConv 直接套入具有 residual 或 concatenation 的網路，會破壞原先設計的 gradient diversity；其策略是在 residual／concat layer 使用移除 identity 的 RepConvN。[YOLOv7 官方 training YAML](https://raw.githubusercontent.com/WongKinYiu/yolov7/main/cfg/training/yolov7.yaml) 也採取有位置規劃的 RepConv，而不是把每個 convolution 全部改名。

**【文獻事實】** [YOLOv6 官方 RepPAN 原始碼](https://github.com/meituan/YOLOv6/blob/main/yolov6/models/reppan.py) 在 P4、P3、N3、N4 fusion blocks 使用 `RepBlock`，同時保留多個 reduction 與 stride-2 downsample 為一般 Conv；其 [官方 common layers](https://github.com/meituan/YOLOv6/blob/main/yolov6/layers/common.py) 實作了對應的 RepVGG block。這提供「按角色選位」的第一手先例，但 YOLOv6 與本 repo 的 YOLO26/C3k2 並不是同一架構。

**【工程推論】** 全面替換會同時改變更多 pretrained paths、BN 統計與 optimizer 動態，且 C3k2 內部涉及 concat／shortcut，讓因果歸因與 function-preserving graft 更難。先換 layer 17/20 的理由是降低實驗風險與定位成本；不是因文獻已證明它們必勝。

### 3.3 「只換 P3/P4 少數 3×3」的可否證假說

**【待驗證假說 H-R1】** layer 17 的 training-time `1×1` auxiliary branch 能改善 P3→P4 的多尺度融合優化，fuse 後不增加 deployment graph depth。

**【待驗證假說 H-R2】** layer 20 可能提供相同效果，但因其更靠近低解析度 P5，對 small-object／ball AP 的幫助可能小於 layer 17。

**【待驗證假說 H-R3】** layer 17+20 的收益不一定相加；若 R3 不勝 R1/R2，表示額外重參數化帶來的優化擾動已超過收益。

**【待驗證假說 H-R4】** 在相同 parent、相同 seed 與相同 recipe 下，局部 RepConv 至少能維持 accuracy；若沒有增準，它對原本已是單一 `3×3` 的 baseline 也不會帶來可期待的理論 FLOPs 優勢，應停止擴大替換。

### 3.4 RepConv 必須通過 post-fuse quantization gate

**【文獻事實】** [QARepVGG（AAAI 2024）](https://ojs.aaai.org/index.php/AAAI/article/view/29045) 指出，標準 RepVGG 的多分支融合可能放大 activation variance 與 fused-weight outliers；其 RepVGG-A0 案例在標準 PTQ 下由 FP32 `72.4%` 降至 INT8 `52.2%`，改成量化友善 block 後為 `70.4%`。這是整網 ImageNet/RepVGG 的結果，不是證明本 repo 只換兩層就會崩潰；它提供的是不可略過 PTQ gate 的直接理由。

**【本地證據】** `yolo_quantize/` 目前 corrected catalog 有 `251` 個 Conv/Linear：`148` 個 deployment、`99` 個 training-only、`4` 個 BinaryQK protected。V0–V3 只有 CPU 靜態分析；V4/V5 尚未獲准執行，沒有 RepConv 後的量化 mAP 或 QAT 證據。

**【工程推論】** 正確順序是：

1. 以未 fuse graph 訓練 RepConv。
2. fuse 成單一 `3×3`，先過 FP32 output parity 與完整八指標 gate。
3. 重新產生 deployment catalog；observer/calibration 必須看 fused weights/activations。
4. 用 accepted SiLU 與 qSiLU parent 做 matched PTQ，比較原 Conv 與 RepConv winner 的 incremental quantization loss。
5. 若只有量化失敗，再評估 QARepVGG-style branch/BN 調整，或 [RepOptimizer](https://github.com/DingXiaoH/RepOptimizers)；這兩者會改訓練契約，不列入第一輪。

## 4. BinaryQK 為何掉精度

### 4.1 sign 會丟失 magnitude，score 分布因而改變

**【文獻事實】** [BinaryAttention 論文（CVPR 2026）](https://arxiv.org/html/2603.09582) 指出，直接將 Q/K 二值化會丟失 magnitude information，改變 attention score 的分布，造成 softmax attention 過度平坦；作者用 scaled binary representation、learnable relative positional bias、QAT 與 self-distillation 補償。[作者官方 `models.py`](https://github.com/EdwardChasel/BinaryAttention/blob/main/models.py) 的 `_quantize` 以 `mean(abs(x))` 估 scale，再套 clipped-STE sign。

**【文獻事實】** 論文 Table 5 的 DeiT 消融為：無 scale/bias/distill 時 Tiny/Small/Base 為 71.95/79.59/81.10；加 scale 為 72.42/79.81/81.33；再加 distill 為 72.44/79.97/81.99；scale+bias+distill 為 72.88/80.24/82.04。這證明各補償在該 ImageNet/DeiT recipe 有用，不證明 YOLO26 的收益幅度相同。

**【本地證據】** 舊 YOLO11m 的 raw sign、scaled sign、dual zero-train 結果依序改善，方向與 magnitude-loss 解釋一致；現行 YOLO26 的 dynamic scale 也略勝 PoT scale。

**【工程推論】** 單一 per-sample/per-head global magnitude 只能恢復整體尺度，無法恢復 token/channel 間的 magnitude 排序；對 small object 或細長球棒所依賴的少數強關聯，score 排名可能仍被改寫。這需要用 score rank/attention entropy 診斷，不可只看最終 mAP 猜測。

### 4.2 ranking disorder 是獨立於整體 scale 的問題

**【文獻事實，間接】** [Bi-ViT（AAAI 2024）](https://ojs.aaai.org/index.php/AAAI/article/view/28109) 將 fully-binarized ViT 的 attention distortion 分解為 gradient vanishing 與 ranking disorder，並使用 learnable per-head scale 與 ranking-aware teacher distillation 修正。它量化的範圍遠比本 repo 的 QK-only 更激進，因此只能支持機理與候選 loss，不能搬用其 accuracy 增益幅度。

**【工程推論】** `mean(abs())` 或 dynamic scale 可校正 score 的整體量級，卻不保證每列 token 排名接近 FP。若 FP-vs-binary top-k overlap／Spearman ranking 很差、但 entropy 與 scale 已合理，應優先測 ranking consistency/KD，而不是再換一個 normalization approximation。

### 4.3 clipped STE 造成梯度近似與飽和區

**【文獻事實】** BinaryAttention 官方實作與本機 quantizer 都採 clipped STE：forward 是 sign，backward 僅在有限區間近似有梯度。

**【本地證據】** 現行 `binary_basis.py` 在 `|x| > 1` 時不傳 STE gradient。

**【工程推論】** 若 Q/K 在某些 heads/sites 大量落在 `[-1,1]` 外，該處量化邊界很難靠現有 gradient 更新；反之若大量集中在 0 附近，sign 對小擾動敏感。應記錄每 site/head 的 `|q|>1`、`|k|>1` 比例、正負號比例與距離 threshold 的分布，再決定是否需要 learnable threshold 或尺度校準。

### 4.4 兩個 sites 同時二值化，可能累積上游誤差

**【本地證據】** YOLO26 同時改 `model.10.m.0.attn` 與 `model.22.m.0.1.attn`；舊 YOLO11m 只有一個 site，且尚無 YOLO26 one-site 結果。

**【工程推論】** site 10 位於 top-down fusion 上游，其 score 擾動可能傳到後續 P3/P4/P5；site 22 較接近 P5 output。若 only-site-10 明顯較差，主要問題是上游誤差傳播；若 only-site-22 較差，則可能是大感受野／P5 head 對 QK precision 敏感；若兩個單 site 都接近 FP、both 才掉，才支持誤差累積或交互作用。這三種情況需要不同解法。

### 4.5 現行適應 recipe 與論文 recipe 不同

**【文獻事實】** BinaryAttention 論文以 full-precision model 初始化，採 QAT、FP teacher self-distillation，完整設定為長 schedule；官方 [training entrypoint](https://github.com/EdwardChasel/BinaryAttention/blob/main/main.py) 預設可更新全模型。官方 `--attn-only` 是選項，而且即使啟用仍會保留 head/positional embedding 等參數，與本地舊 YOLO11m 嚴格只開 attention 並不等同。

**【文獻事實】** BinaryAttention Table 9 顯示相同方法由 100 epochs 延長至 300 epochs 時，Tiny/Small/Base 分別由 71.98/79.44/81.80 到 72.44/79.97/81.99，說明充分適應在作者的分類設定中有實質影響。

**【工程推論】** YOLO26 的 `.0105` 缺口可能同時包含 binary representation loss、兩-site 交互作用，以及 recovery schedule/解凍範圍不足。不能在沒有 matched FP fine-tune control 的情況下，把全部差距歸因給二值化數學。

### 4.6 已有本地結果排除了哪些優先方向

**【本地證據】** W-DIR 勝 W-PROG；exact 與 SHIFT 幾乎相同；dynamic scale 只比 PoT 高
`0.000506`；dense bias 小勝 no-bias；N4 magnitude 舊實驗未勝 parent；learned BDCN 有 seed 不穩現象。

**【工程推論】** 因此下一輪不應優先重跑 progressive、normalization approximation、N4 magnitude 或 BDCN。dense bias 可留在 accuracy parent，但它不是主要缺口的答案。

## 5. BinaryQK 精度恢復方法：優先順序

### 5.1 條件式 accuracy ceiling：per-token/head magnitude

現行 dynamic scale 對 Q/K 的 channel 與 token 軸一起平均，每 sample/head 只留一個 magnitude。本地
兩張圖 probe 只把 scale 改為每 token/head，site 10 的 top-10 overlap 由 `0.4206` 到 `0.5119`、
KL 由 `0.5431` 到 `0.4651`；site 22 的 top-10 overlap 由 `0.4962` 到 `0.5952`、KL 由
`0.8810` 到 `0.6188`。

**【工程推論】** 它不增加 binary basis，但每張 image 都要重算 O(BHN) scales，並在 N² score
epilogue 套用 pair scale；兩張 probe 不是 mAP 證據，也不能直接宣稱加速。由於既有 global dynamic
與 fixed PoT 已有完整 scale screen，per-token 不列首輪 production 候選。只有 fixed-PoT site winner
經 matched direct QAT、單一 KD／STE-window 仍未達標，而且產品明確接受 dynamic-scale kernel成本時，
才成對執行 `DYN-GLOBAL` 與 `TOKEN-BOTH`；只有兩者差值才能歸因 token granularity。

### 5.2 第一優先：fixed-PoT site isolation，再做 matched direct QAT

| ID | site 10 | site 22 | 目的 |
|---|---:|---:|---|
| Q0 | Binary | Binary | retained V1-BR baseline |
| Q1 | Binary | FP | 測上游 site 10 的 hybrid 可行性 |
| Q2 | FP | Binary | 測輸出側 site 22 的 hybrid 可行性 |

**【本地證據】** W-DIR 在 40 epoch 計畫只完成 26 epoch，best 在 epoch 21；它已是全模型解凍，
且 checkpoint 已刪除。現行 direct 明顯勝 progressive，但不能從 W-DIR resume。

**【工程推論】** 既有 V1-DYN／V1-SHEAD／V1-P2 scale screens 直接重用；首輪只新增 Q1、Q2 兩個
免訓練 fixed-PoT site screens，並選一個 winner。正式 recovery 從 final same-lineage FP checkpoint
建立 `FP-CTRL` 與 BinaryQK candidate，兩者使用相同 full-model scope與 W-DIR direct
protocol。這樣才能分辨普通 fine-tune gain 與 binary gap；不再把 staged unfreeze、延長 epoch或
progressive blend 一起加入。

### 5.3 第三優先（需符合實驗計畫授權）：FP teacher 與 ranking-aware distillation

**【文獻事實】** BinaryAttention 用 full-precision teacher 的 self-distillation，且 Table 5 顯示 scale 上再加 distillation 有小幅增益。舊本地 YOLO11m 的 positional+feature KD 也把 T4 與 T0 的 gap 由 `.00086` 縮至 `.00048`，但只有 single seed。

**【工程推論】** 若現行計畫原先明定 no-KD，必須先取得變更授權；不得把 KD 偷渡進主對照。獲准後先做一個最小 KD arm：凍結 FP teacher、student 使用同 augmentation，固定 loss weight，對齊 output logits／features，且保留 no-KD matched arm。若 score top-k/rank 診斷仍差，再獨立加入 Bi-ViT 類 ranking consistency；不要第一個 run 就同時加 output、feature、attention、ranking 四種 loss。KD 是 recovery 手段，不改變部署 graph。

### 5.4 第四優先：per-head learnable threshold／temperature

**【文獻事實，間接】** [ReActNet 論文](https://arxiv.org/abs/2003.03488) 與 [作者官方 repository](https://github.com/liuzechun/ReActNet) 以 learnable channel-wise threshold 改善 binary activation 的分布適配。它不是 attention-specific，也沒有直接證明可改善本 repo 的 Q/K。

**【本地證據】** T4 dual basis 已含 learnable thresholds，且舊 YOLO11m 表現最佳；現行 single-basis global scale 仍可能無法修正各 head 的 sign imbalance。

**【待驗證假說】** 先在 single-basis 上加入 per-head（必要時 per-channel）`tau_q/tau_k`，並可加入 per-head learnable score temperature；這比直接做四個 QK basis products 更便宜。初始化必須保持 `tau=0`、temperature 等於現行 scale，先驗證初始輸出等價。bit-balance regularizer 必須另設單因子 arm；若 sign-balance/entropy 診斷沒有改善，停止此方向。

### 5.5 最後才做 full residual dual-basis

**【本地證據】** 舊 YOLO11m T4 用兩個 residual bases 的四種 QK cross terms、一次 softmax/PV，能在 10-epoch attention-only QAT 把 gap 縮到 `.00086`；但現行 YOLO26 的 Hadamard two-basis 未勝 W-DIR。

**【工程推論】** dual-basis 的 accuracy 潛力較明確，但會增加 QK products、kernel 複雜度與硬體整合成本。只有在 site isolation、direct recovery 與單 basis threshold 後仍超過預先設定的 accuracy gap，且 profiling 顯示 attention 是值得優化的 latency 熱點時，再投入 full residual dual-basis。

## 6. 可直接執行的消融計畫

### 6.1 共通鎖定項目

**【本地規範】** BBAT5 的 detection、pose 與融合實驗只能使用不可變資料：

- root：`/home/uxin/yolo/original/pose/derived/bbat5-v1/`
- detect：`configs/detect.yaml`
- pose：`configs/pose.yaml`
- registry：`/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`

不得重新 split、抽樣、改 label 或另建資料版本。COCO 與 BBAT5 結果須分表，不得以不同資料集的 mAP 做前後比較。

每個 run manifest 至少固定：git commit、parent checkpoint hash、dataset registry/hash、model YAML、實際 module paths、trainable parameter names/count、optimizer、各 parameter-group LR、epochs、image size、batch/accumulation、AMP、augmentation、seed、early-stop 規則與 deploy/fuse 狀態。

### 6.2 RepConv 最小矩陣

| ID | layer 17 | layer 20 | 其他結構 | 問題 |
|---|---:|---:|---|---|
| R0 | Conv | Conv | Full35 J3；MASF/BinaryQK 固定 | same-recipe control |
| R1 | RepConvN | Conv | 同 R0 | P3→P4 transition 是否有益 |
| R2 | Conv | RepConvN | 同 R0 | P4→P5 transition 是否有益 |
| R3 | RepConvN | RepConvN | 同 R0 | 兩者是否可疊加 |

執行順序：

1. 用 runtime graph 解析 layer 17/20 的精確 module path 與 shape；寫入 manifest。
2. 由同一個未 fuse Full35 J3 `best_joint` checkpoint 做 function-preserving graft；本機類別使用 `RepConv(..., bn=False)`，在這兩個 stride-2 seams 即等價於表中的無 identity `RepConvN`。
3. 先跑 graft 等價測試，再進行訓練；R0 也用同一 loader/recipe 重跑。
4. 每組至少 3 seeds；validation 用於選擇，test 僅在方案 freeze 後一次性評估。
5. 每個 checkpoint 都產生 pre-fuse 與 post-fuse 成對驗證，並在實際 deployment backend 測 p50/p95 latency。

**停止規則（工程準則）**：若 R1/R2/R3 都沒有在 paired-seed CI 顯示 accuracy 改善或非劣，且 fused latency 相對 R0 沒有可重複的改善，就不擴大到 C3k2 內部／全 Neck。若只有 R1 有效，後續只保留 layer 17；不為「對稱」強留 layer 20。

### 6.3 MASF 最小獨立檢查

| ID | checkpoint／改動 | 訓練 | 目的 |
|---|---|---:|---|
| M0 | Full35 J3 `best_joint` 原樣 | 無 | 目前八項 baseline |
| M1 | 同一 checkpoint，只在 eval 將 `p3_masf.alpha=0` | 無 | 測目前模型對 MASF 的即時依賴 |
| M2 | 從 M1 移除 MASF graph，matched low-LR recovery | 僅 M1 通過或接近 gate 才做 | 判斷能否用 recovery 安全簡化 |

M1 不更新 checkpoint、不改資料、不建立新 split；它是最便宜且資訊量最高的 MASF 決策點。若 M1 已明顯跌破 gate，就不能把 architecture_1 的近似 tie 直接解讀成「目前 Full35 可無損移除」。

### 6.4 BinaryQK 最小矩陣

2026-09-02 hardware-first 修正後，Phase A 重用既有 scale screens，只做兩個新 validation：

| ID | site 10 | site 22 | scale | 訓練 | 目的 |
|---|---|---|---|---:|---|
| B0 | Binary | Binary | fixed per-head/basis PoT | 0 | 既有 V1-BR baseline |
| B1 | Binary | FP | fixed PoT | 0 | site 10 binary hybrid |
| B2 | FP | Binary | fixed PoT | 0 | site 22 binary hybrid |

`DYN-GLOBAL`／`TOKEN-BOTH` 不在首輪矩陣；若前述 hardware-friendly recovery 均失敗且願意接受
dynamic scale成本，兩者才作為一組 matched conditional experiment，不得只跑 per-token。

Phase B 只把唯一 winner 帶到 final same-lineage FP parent：

| ID | 改動 | 目的 |
|---|---|---|
| B5 | FP-CTRL，與候選相同 direct recovery | 排除普通 fine-tune gain |
| B6 | BinaryQK winner，direct QAT | 量 matched FP gap |
| B7 | B6 + 單一 attention-map 或 ranking KD | 只有診斷仍差才做 |

per-head STE window、threshold與 full residual dual-basis 都降為 B7 仍失敗後的條件式研究，不列入
首輪矩陣。完整 gate 以[新版計畫](<../../proposals/binaryqk-accuracy-recovery/plan.md>)為準。

Phase C：獨立 winner 決定後才做 interaction。

| | 現行 BinaryQK | recovered BinaryQK winner |
|---|---|---|
| RepConv off | I0 | I1 |
| RepConv winner | I2 | I3 |

這個 `2×2` 必須在同一 Full35 J3 lineage 上重建，可回答 RepConv 與 BinaryQK recovery 是否近似可加、是否互相抵銷；MASF 在四格中固定不動。單任務 FP/Binary control 仍留在 Phase A，不跨 evaluator 併表。

### 6.5 必記的 accuracy 與機理診斷

每組至少報：

- COCO/BBAT5 overall AP、AP50、AP75、APs/m/l；BBAT5 另報 ball/bat AP。
- 至少 3 seeds 的 mean、standard deviation、paired difference 與 95% CI；小於千分位的 single-seed 差異只列為觀察。
- 每 site/head 的 Q/K reconstruction error、FP-vs-binary score cosine/RMSE、top-k overlap。
- attention entropy、FP-vs-binary attention KL、最大 attention probability。
- Q/K 正號比例、`|q|>1`／`|k|>1` 的 STE saturation 比例、threshold 周邊密度。
- 各 site 的 feature drift，以及 small/medium/large object 與 ball/bat 的誤差分解。

**【工程推論】** 診斷對應的決策是：

- entropy 明顯升高、top-k overlap 降低：優先 scale/temperature/bias。
- sign 極度不平衡：優先 learnable threshold。
- saturation 高：調整 pre-quant scale、normalization 或 STE window，但必須另設消融。
- only-site-10 已造成大部分損失：先保留 site 10 為 FP，量化 site 22；反之亦然。
- 兩個單 site 都近 FP、both 才掉：做 staged recovery 或只量化一個 site，通常比加更多 basis 更省硬體成本。

### 6.6 部署與成本驗收

每個候選須同時報：

- train graph 與 fused/export graph 的 params、GFLOPs、checkpoint size。
- target backend 的 warm-up、batch、image size、precision、p50/p95 latency、throughput、peak memory。
- RepConv graft 前後、RepConv fuse 前後的 `max_abs`/`mean_abs` output error。
- BinaryQK 是否仍是 fake quant；若沒有 bitwise/custom kernel，不得宣稱 1-bit speedup。

**【文獻事實】** BinaryAttention 的速度結果來自作者的 binary kernel 路徑，而非只把 PyTorch tensor 經過 sign。[官方 repository](https://github.com/EdwardChasel/BinaryAttention) 提供其實作背景。

**【本地證據】** 舊 `yolo_binaryqk` 與現行 `yolo_attention` 的目前路徑主要是 fake quant／accuracy simulation。

**【工程推論】** 現行 YOLO26 只有兩個 attention sites；即使 QK kernel 很快，end-to-end speedup 仍受 attention 在總 latency 的占比限制。先 profile 才能決定值不值得用 dual-basis 換 accuracy。

## 7. 建議的立即決策

1. **RepConv：值得先做。** 只做 R0/R1/R2/R3，目標 layer 17/20，從目前 Full35 J3 `best_joint` 出發，固定 MASF/BinaryQK；function-preserving graft 與 fuse gate 的 CPU prototype 已通過。
2. **MASF：先做零成本依賴檢查。** 同一個正式 checkpoint 跑 M0 與 `alpha=0` 的 M1 完整 validation；未看到八項結果前，不直接刪除已適應的 MASF branch。
3. **BinaryQK：重用 scale 消融，先做 fixed-PoT hybrid。** 用 retained V1-BR 只跑 `ISO-10-BIN`、`ISO-22-BIN` 兩個免訓練 validation；不要重跑 V1-DYN／SHEAD／P2，也不要先開 per-token、progressive、SHIFT recovery、N4 或 BDCN 搜尋。
4. **恢復：從 final FP parent 做 matched direct QAT。** 唯一 winner 與 `FP-CTRL` 使用相同 full-model scope、schedule與 seed；不能 resume 已刪除的 W-DIR，也不先增加 staged-unfreeze 變因。
5. **KD/threshold 有條件地後測，dual-basis 最後。** KD 先單一 output/feature arm，再按 ranking 診斷測 ranking loss；threshold 針對 sign imbalance；dual-basis 必須連同 QK-op 次數與真實 latency 評估。
6. **RepConv 一定要過 post-fuse PTQ gate。** 先 fuse、重建 148-layer deployment catalog 與 calibration，再比較 accepted SiLU/qSiLU parents；FP32 等價不等於 INT8 等價。
7. **最後才做同 lineage interaction。** `{Rep off/winner} × {current/recovered BinaryQK}`；四格固定 MASF，MASF on/off 另作獨立矩陣。

## 8. 風險、未解事項與對策

| 風險／未解事項 | 影響 | 對策 |
|---|---|---|
| `yolo_optimize` 在本次稽核時尚未有 model scaffold | 無法直接指定新專案 module path | 先以 YOLO26 runtime graph 生成 manifest，再做 graph-aware graft |
| parser 未把 RepConv 列為 base module | YAML 直接替換可能解析錯 channel | custom wrapper/graft，補 parser unit test 後才允許 YAML 化 |
| pretrained graft 不等價 | accuracy 差混入初始化 shock | `1×1` zero branch；train 前 FP32 等價 gate |
| 目前 MASF alpha 非零 | 直接刪除可能造成 checkpoint shock | 先跑同 checkpoint `alpha=0` full validation，再決定 recovery |
| MASF、RepConv、BinaryQK 同改 | 無法歸因 | 三者先獨立，最後才 interaction |
| RepConv fuse 後 PTQ outlier | FP32 通過但 INT8 掉點 | fuse 後重建 catalog/calibration；matched PTQ gate，必要時才評估 QARepVGG/RepOptimizer |
| 舊 YOLO11 與現行 YOLO26 混讀 | 錯估 recovery 難度 | 分開報表與 lineage；不得跨世代宣稱增益 |
| single seed 千分位差 | 可能只是 noise | 至少 3 seeds、paired CI |
| fake quant 沒有真實加速 | accuracy trade-off 沒有產品價值 | target backend profiling；bit kernel 另設里程碑 |
| BBAT5 被重新切分 | 破壞可比較性與規範 | 只用 canonical bbat5-v1 configs/registry |

## 9. 第一手來源

### 結構重參數化／YOLO

1. Xiaohan Ding et al., **RepVGG: Making VGG-style ConvNets Great Again**, CVPR 2021：<https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html>
2. 作者官方 RepVGG repository／implementation：<https://github.com/DingXiaoH/RepVGG>、<https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py>
3. Chien-Yao Wang et al., **YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors**：<https://arxiv.org/pdf/2207.02696>
4. YOLOv7 作者官方 repository／YAML：<https://github.com/WongKinYiu/yolov7>、<https://raw.githubusercontent.com/WongKinYiu/yolov7/main/cfg/training/yolov7.yaml>
5. YOLOv6 作者官方 RepPAN／common layers：<https://github.com/meituan/YOLOv6/blob/main/yolov6/models/reppan.py>、<https://github.com/meituan/YOLOv6/blob/main/yolov6/layers/common.py>
6. Ultralytics 官方 YOLO26 YAML／RepConv／fuse parser：<https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/26/yolo26.yaml>、<https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/nn/modules/conv.py>、<https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/nn/tasks.py>
7. Xiangxiang Chu et al., **Make RepVGG Greater Again: A Quantization-Aware Approach**, AAAI 2024：<https://ojs.aaai.org/index.php/AAAI/article/view/29045>
8. RepOptimizer 作者官方 repository：<https://github.com/DingXiaoH/RepOptimizers>

### Binary attention／二值化

1. Chaodong Xiao, Zhengqiang Zhang, Lei Zhang, **BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers**, CVPR 2026：<https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_BinaryAttention_One-Bit_QK-Attention_for_Vision_and_Diffusion_Transformers_CVPR_2026_paper.html>
2. 作者官方 BinaryAttention repository／model／training entrypoint：<https://github.com/EdwardChasel/BinaryAttention>、<https://github.com/EdwardChasel/BinaryAttention/blob/main/models.py>、<https://github.com/EdwardChasel/BinaryAttention/blob/main/main.py>
3. Yanjing Li et al., **Bi-ViT: Pushing the Limit of Vision Transformer Quantization**, AAAI 2024：<https://ojs.aaai.org/index.php/AAAI/article/view/28109>
4. Zechun Liu et al., **ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions**, ECCV 2020：<https://arxiv.org/abs/2003.03488>
5. 作者官方 ReActNet repository：<https://github.com/liuzechun/ReActNet>

## 10. 最終判斷

**【工程推論】** RepConv 在本 repo 有用的最合理機會，不是「全面把 Conv 改名」，而是把 layer 17/20 當成兩個可控的 training-time over-parameterization probes：從目前 Full35 J3 保持 epoch-0 函數、分別驗證、部署前融合、融合後再做 PTQ。只要 R1/R2 沒有穩定 accuracy 收益，或 post-fuse quantization loss 不可接受，就沒有理由擴到 C3k2 內部。

**【工程推論】** BinaryQK 的精度恢復也不應再盲目堆技巧。第一個未知量是兩個 attention sites 各自造成多少損失；第二個是現行 direct 方案是否只缺 recovery time／適應範圍。只有這兩件事量清楚後，KD、learnable threshold 或 full residual dual-basis 才有可歸因的價值。現有證據支持「scaled/direct QAT 可以補回大量精度」，但尚不支持「YOLO26 已無損」或「假量化已帶來端到端加速」。

**【工程推論】** MASF 的合理結論是「兩條既有實驗線都沒有 material gain 證據」，不是「可直接從目前模型拔掉」。正式 checkpoint 的 alpha 已到 `0.110659`；先做 alpha=0 的 full validation，才知道移除是否值得花 recovery 成本。
