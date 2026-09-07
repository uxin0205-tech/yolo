# Full35 Activation-aware Quantization 實作計畫 v5

> 2026-09-07狀態補充：最新盤點與後續96小時執行順序以[四天整合計畫](docs/reports/2026-09-07-full-model-audit-four-day-plan.md)為準。V36 QAT已完成，epoch1/2搜尋通過；舊v5正文保留歷史契約，不代表其所有待辦仍未執行。parent統一、十大區域sensitivity與整數部署邊界仍未完成。

更新日期：2026-09-04。

這是2026-09-04的v5基礎契約；最新執行順序見上方四天整合計畫。它整合先前完整 grilling、V0–V3、使用者「納入 Hardswish、排除 `poly_quality`」的決策、Full35 Q3 CPU 證據，以及架構／實驗／文獻稽核。可供程式讀取的總契約是[`full35-quantization-plan-v5.yaml`](configs/experiments/full35-quantization-plan-v5.yaml)。現行硬門檻是：**activation替換與weight量化合併後，八項mAP50相對accepted Full35的每項下降不得超過`0.015`，八項mAP50–95每項下降不得超過`0.04`；兩族都通過才是green**。`poly_quality`產物凍結保留，不回寫成active結論；舊報告的mAP50–95仍是歷史證據，新結果一律從hash-pinned raw metrics重判。

## 一、目前進度

| 階段 | 內容 | 現況 | GPU |
|---|---|---|---|
| V0 | 遠端 activation publication 與本機 PTQ v1 血緣整合 | 已完成；公開報告逐位元相同 | 未使用 |
| V1 | corrected 148-layer catalog、unfused／BN-folded dual view | 已實作並通過真實 Full35 CPU parity | 未使用 |
| V2 historical | W8–W4、Fixed SD4、Paper-TWN 靜態分析 | 8,880 筆 active-parent 舊結果保留；尺度實為十點 `mse_grid_v1`，不得直接作新版 routing | 未使用 |
| V3 corrected | 固定 32/64 manifest、八項 metric gate | manifest 降格為 structural diagnostic；gate 已補 evaluator/data contract 與 matched-policy 身分 | 未使用 |
| G0 CPU | integer boundary、exact scaled-codebook、SD4 bit mapping、routing | CPU reference契約、16-code pack/unpack、三parent共3,552筆exact/grid公平重算與36層routing v3均已完成 | 未使用 |
| V5-S1 | qSiLU＋A8／backbone early W8–W4 bit邊界 | 完整search完成；W8 green、W7 recover、W6–W4 reject | 已使用並釋放 |
| V5-S2 | qSiLU＋A8其餘九區isolated W8 | 8格green；只有backbone attention-safe recover；不碰formal val | 已完成並釋放 |
| V5-S3–S5 | 累積W8、mixed precision、SD4／ternary、Hardswish／poly_shift | poly_shift W8–W4、mixed、Fixed-SD4、Paper-TWN與耦合PTQ已完成 | 已使用並釋放 |
| V5-S6–S7 | paired QAT、formal、多seed、export | V19 sham已完成；fused-head contract與0.2 FP32比例誤判均已修正，QAT由epoch 0 last.pt精確續跑；formal只給鎖定finalists | 進行中 |

目前沒有formal winner。poly_shift PTQ accuracy候選為九區W8＋單一SD4 Detect predictor，最差total mAP50／mAP50–95 `-0.011415／-0.017322`；balanced SD4 Detect head約`3.611×`但mAP50–95接近門檻。詳見[`poly_shift PTQ、特殊格式與QAT入口報告`](docs/reports/2026-09-03-poly-shift-ptq-special-format-and-qat-entry.md)。所有total都包含activation替換，不能只看weight incremental或跨task平均。

## 二、不可變來源與硬邊界

### 模型

- 唯一目標是 `/home/uxin/yolo/yolo_combine/final/full35/` 的 shared Detect＋Pose accepted J3 experimental baseline。
- inference checkpoint 是 `weights/combined/inference/best_joint.pt`，SHA-256 `d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c`。
- 不把舊 Detect-only `quantize_spec.md` 的缺失 checkpoint 靜默替換成 Full35；它只保留方法參考地位。
- 不修改 `yolo_combine/final`、`yolo_activation` 或 site-packages。

### 資料

- COCO80 Detect 使用 `/home/uxin/yolo/coco2017.yaml`。
- BBAT5 Pose／二類 Detect 只使用不可變 `bbat5-v1` 的正式 YAML。
- Detect 與 Pose 保持兩個 task-specific loaders；不把資料夾物理合併，也不混合 raw mAP。
- V3 diagnostic manifest 只挑 canonical exemplar，不改 split、影像或 labels，也不是新資料版本；active runner會驗證manifest及全部檔案SHA-256，再固定取首張probe。它只證明結構／數值 smoke，不能選 winner。
- BBAT5 screening 使用既有 canonical `pose-search.yaml`；formal `pose.yaml` 只給預註冊 finalists 作確認，不以 formal-val 搜尋150格。
- 未來 30% QAT train view 尚未建立；必須一次建立版本化／hash manifest、COCO person與size coverage、BBAT `.rf.` group-safe，且不得包含formal val。

### Graph 保護

- Binary Q/K sign／bit-true contract 不插入一般 quantizer。
- attention PWL、MASF 拓樸與融合語意保留。
- TopK、gather、box decode、keypoint decode 保持既有精度。
- `reg_max=1`，因此沒有有效 DFL 量化工作；one-to-one end-to-end 是 NMS-free，不新增 NMS。
- O2M、pose_flow 與 pose sigma 保留作訓練 supervision，不算部署壓縮收益。

## 三、Activation 與 weight 必須耦合

不能把「activation 單獨最好」和「weight 單獨最好」事後拼接。每一格的完整身分必須包含：

```text
activation function + A-bit/quantizer
+ activation parent checkpoint SHA-256
+ weight region→format mapping
+ scale granularity/method
+ calibration/training recipe SHA-256
+ protected/training-only exclusions
```

第一輪固定三條 A8 parent：

| Parent | 角色 | Checkpoint SHA-256 |
|---|---|---|
| qSiLU＋A8 | accuracy fallback | `76791866…6190e` |
| Hardswish＋A8 | standard hardware operator | `79e0e4f6…7731` |
| poly_shift＋A8 | dyadic hardware extreme | `8783248a…9713` |

- 不新增 SiLU；SiLU 只保留 accepted historical control。
- 不使用被中斷的 qSiLU finalist checkpoint。
- `poly_quality` 從現行 matrix、CLI active choices、racing 與 QAT promotion 排除；舊 JSON、hash、圖表與報告不刪除。
- A3–A5 不晉級；weight map 穩定後才測 A7／A6。

Hardswish 有兩種不可混稱的身分：

1. **Uniform experimental parent**：使用 epoch-8 `best_joint.pt`。該 checkpoint 八項最差 delta 為 `-0.014121`，但原 run 的 final epoch 為 `-0.016884`、依上游 final-epoch selector 仍是 fail；完整邊界見 [`hardswish-uniform-parent-intake-v1.yaml`](artifacts/manifests/hardswish-uniform-parent-intake-v1.yaml)。
2. **Q3 regional policy**：權重根仍是 qSiLU `767918…6190e`，只把 `neck_attention`、`masf` 或 `backbone_attention` 單區換成 Hardswish。Q3 沒有測多區組合，也沒有 A8/W-bit；因此它只提供 placement 優先序，不是量化結果。

## 四、V1：Corrected graph 與 dual views

### Corrected catalog

真實 Full35 共 251 個 Conv／Linear：

| 類別 | Modules | Weight elements |
|---|---:|---:|
| 部署候選 | 148 | 22,571,840 |
| Training-only | 99 | 3,748,480 |
| Binary Q/K protected | 4 | 131,072 |

舊 v1 把 `one2one_cv4_sigma.[0-2]` 三層誤算為部署 predictor；官方 inference fuse 會移除 sigma，因此已改為 training-only。歷史六格結果不受影響，但舊 151-layer manifest 不得當成新基線。

### Dual views

`Full35WeightViewAdapter.build(model)` 一次建立：

1. unfused FP32 master：保留 BN 與所有 training supervision，供未來 QAT。
2. BN-folded deployment：移除 BN、O2M、pose_flow 與 sigma，供 PTQ、靜態重建及 export。

來源模型不被修改。manifest 包含 state SHA-256、179→0 BN 計數、148 條 deployment path parity、inference contract，以及 deterministic 64×64 CPU forward parity。只允許 one2many tensor paths 消失；one-to-one 輸出必須同結構、finite、NRMSE ≤ `1e-5` 且 relative-to-reference-peak max error ≤ `1e-4`。

## 五、V2 historical 與 G0 exact baseline

`WeightFormatAnalyzer.analyze(views, plan)` 已對現行三個activation parent（qSiLU、Hardswish、poly_shift）、raw與BN-folded兩個View的148個部署層完成8,880筆歷史分析；`poly_quality` 的2,960筆另行保存。稽核確認舊稱 `mse` 的尺度只搜尋十個 `absmax × fraction`，現統一標成 `mse_grid_v1`，不得再稱全域MSE最佳。

G0新增 `optimal_scaled_codebook` Module，對固定signed codebook掃描assignment boundary並在每個piecewise-quadratic interval取最佳尺度；CPU測試已用獨立暴力assignment search核對。它支援per-tensor、per-output-channel、group32、group64，uniform與Fixed SD4共用同一Seam。

2026-09-01已在三個active parent重跑最小公平矩陣：148層 × master／deployment × uniform W4 grid/exact與Fixed SD4 grid/exact，每parent 1,184筆、合計3,552筆。exact在888個uniform與888個Fixed SD4 cells均不劣於grid；cross-parent、cross-view strict intersection得到36層routing v3。舊8,880筆與34層routing v2保持原檔／原hash，只作歷史ablation。

### Uniform 主矩陣

```text
W8 → W7 → W6 → W5 → W4
```

- 歷史主比較是per-output-channel＋`mse_grid_v1`；新比較必須明寫scale method。
- uniform weight使用zero-point 0與完整two's-complement範圍：W8 `[-128,127]`、W7 `[-64,63]`、W6 `[-32,31]`、W5 `[-16,15]`、W4 `[-8,7]`；不再把設定誤寫成narrow symmetric range。
- uniform W4與Fixed SD4的強baseline都使用`optimal_scaled_codebook`、同parent、同layer與同granularity。
- 額外ablation：per-tensor、per-output-channel、group32、group64 × max／`mse_grid_v1`／exact；高位寬exact event數較多，先以W4完成公平基準。
- W5/W6/W7 可用於自訂 FPGA／ASIC 或 bit packing 搜尋；沒有對應 kernel 前，只報容量、metadata、BOP／traffic proxy，不宣稱 GPU speedup。

### Fixed SD4

- 4-bit、16 encoded nibbles／15 unique values；`0111`與`1111`分別是positive／negative zero，export固定用`0111`。
- normalized codebook 為 `0, ±1/64, ±1/32, ±1/16, ±1/8, ±1/4, ±1/2, ±1`。
- `SD4Encoding`已實作16種decode、canonical zero與high-nibble-first pack/unpack；奇數長度使用canonical zero padding。
- 先測max、`mse_grid_v1`與`optimal_scaled_codebook`；再和exact uniform W4、APoT4、Paper-TWN／exact-scaled ternary比較。
- 判斷依據是 codebook reconstruction、clipping、occupancy 與 layer sensitivity，不以「權重較小」單獨決定。

### Paper-TWN

依 PDF 印刷頁 33–38 的靜態式：

```text
delta = 0.7 × mean(abs(W))
alpha = mean(abs(W[abs(W) > delta]))
Wq ∈ {-alpha, 0, +alpha}
```

Paper-TWN、Channel-TWN 與 TTQ 是三種不同方法；目前只實作 Paper-TWN 靜態 proxy。Channel-TWN／TTQ 需要後續 matched QAT，不混稱為論文原式結果。

### 每層輸出欄位

- 身分：View、path、region、family、format、bit、granularity、scale method。
- 分布：min/max、mean/std、mean/max absolute、zero ratio。
- 重建：MSE、NRMSE、SQNR、cosine、maximum error、clipping、occupied codes。
- 成本：code bytes、scale count、metadata bytes。
- SD4/TWN 額外：codebook、threshold、alpha 與 suitability 對照。

這些都是 static proxy，不是 mAP。

## 六、V3 corrected：diagnostic manifest 與雙族十六指標 gate

已建立 `artifacts/manifests/full35-diagnostic-cal32-probe64-v1.json`：

- seed `20260830`。
- calibration：COCO train 32、BBAT formal-train 32。
- probe：COCO val 64、BBAT formal-val 64。
- COCO 兩個 split 都必須包含 person。
- BBAT 每個樣本來自不同 `.rf.` source group，ball／bat／pose 均有 coverage，train／val source-group overlap 為 0。
- 每張 image／label 都保留 path 與 SHA-256。

`WeightSensitivityStudy`現已把這份manifest解析成執行契約；v2 plan先獨立釘住manifest SHA-256，真正執行前再驗證檔內digest並逐檔驗證32＋32張calibration與64＋64張probe證據。現有comparison Interface一次只接受每task一張probe，所以使用manifest中固定排序的第一張，並在report內記錄subset policy；不能由CLI臨時改樣本數。

這份manifest使用formal-val exemplar的既有事實不可回寫。它只允許`diagnostic_pass`（finite＋same structure），`selection_claim=false`、`promotion_authorized=false`；後續screening改用canonical BBAT search-val，formal-val只作locked finalist確認。

最終 metric gate 不只看三個 headline，而是逐項檢查八個指標：

1. COCO box mAP50。
2. COCO person box mAP50。
3. BBAT box mAP50。
4. BBAT pose mAP50。
5. BBAT ball box mAP50。
6. BBAT bat box mAP50。
7. BBAT ball pose mAP50。
8. BBAT bat pose mAP50。

同一路徑另要求上述八項各自的mAP50–95；COCO／BBAT aggregate保留官方輸出，跨task平均與joint priority只作診斷／checkpoint排序，不冒充官方單一mAP。

門檻：

- 每項相對 accepted Full35 的總下降不得超過 `0.015`；total包含activation替換與weight量化，不可分開規避。
- W8 每項相對 matched activation parent 的 incremental 下降不得超過 `0.01`。
- QAT matched sham 每項相對 matched parent 的 absolute drift 不得超過 `0.01`；超過即為 invalid recipe。
- green必須同時通過total gate與適用的W8 incremental gate；若sham有效、尚未green且worst total drop不超過`0.04`，進recover；worst total drop超過`0.04`直接reject。

`Full35MetricGate`另強制candidate、accepted與matched snapshot使用同一`metric_contract_id`，且candidate與matched使用同一activation placement＋A-bit policy。threshold由單一validated spec序列化，不再在判定與報告各自hard-code。

2026-09-03已首次實際套用此gate。三角色都重跑完整COCO val與固定BBAT5 search-val；COCO person沒有省略。accepted是SiLU／FP32 weight，matched是qSiLU recovery checkpoint＋LSQ+ A8／FP32 weight，candidate只再加入`backbone_early` W8。因此只有candidate-minus-matched乾淨代表W8 incremental；matched-minus-accepted是完整parent policy差異，不誤稱純A8效果。

## 七、V5 staged search：poly_shift PTQ矩陣已完成

### V4：15 格 parent×bit bridge

```text
3 activation parents × [W8,W7,W6,W5,W4] = 15 cells
```

先跑三個matched FP-weight control與W8，再依序W6、W4；W7／W5完整保留為boundary bits，之後補齊。固定32/64只作結構／數值probe；promotion使用preregistered search contract，不能由單張probe或static NRMSE決定。

qSiLU＋A8／`backbone_early`已完成W8、W7、W6、W5、W4 grid及W4 exact完整search。W8在新mAP50總門檻下為green；W7只進recover；W6–W4淘汰。其餘九區isolated W8也已完成：8格green，只有`backbone_attention_safe` recover。exact W4與九區結果共同證明static NRMSE或參數量都不能取代task mAP。下一步是以MASF與neck為兩個起點建立累積policy，attention-safe則拆層。

### V4H：Q3 regional Hardswish bridge

- qSiLU checkpoint保持不變，分別建立 `neck_attention`、`masf`、`backbone_attention` 三個單區policy。
- 每個policy先跑同activation placement的FP32-weight matched control與W8，共6格。
- `neck_attention + masf`只有兩個單區W8都通過才建立matched control／W8兩格；不可把單區delta相加。
- 三個單區已完成CPU dual-view parity，activation counts分別為`1/189`、`3/187`、`3/187`（Hardswish/qSiLU）；這不是observer calibration或mAP。

### V5：150 格 isolated-region sensitivity

```text
3 parents × 10 regions × 5 uniform bits = 150 cells
```

region順序固定從backbone_early、backbone_deep、attention-safe、neck、MASF，一路到Detect／Pose predictors。150格全部保留為T0 static universe；T1 diagnostic在時間足夠時可全做，最低先做三parent×十region×W8/W6/W4＝90格再補W7/W5。T2 search validation最多30個預註冊promoted cells＋sentinel；T3 formal最多6個locked finalists。不得把150格全部反覆拿formal-val選winner，也不得自動進QAT。

### V6：Router 與特殊格式

- successive racing＋sentinel audit。
- 保留 accuracy、balanced、hardware 最多三個 Pareto 角色。
- 先用uniform map找臨界層，再把適合的層送入exact uniform W4／Fixed SD4／APoT4／ternary。舊34層Fixed SD4清單來自`mse_grid_v1`，只保留歷史；新版36層exact routing v3只提供GPU output sensitivity候選，仍不得由static MSE直接promotion。
- dominated strata 仍保留一格 sentinel；若 sentinel 通過或勝過 promoted cell，就擴展該 stratum。

## 八、V7–V9：QAT 與 LS-SD4，通過前置條件後執行

### 為什麼要重新訓練

- W8 PTQ 若八項 total gate 與 W8 incremental gate 都通過，可以不做 QAT。
- W7–W4、LS-SD4 與 ternary 多半需要 low-LR recovery，但不是從零訓練。
- 每個 activation parent 使用自己的 FP32 master checkpoint，不互換 optimizer state。

### QAT 篩選

- 最多8個policy跑S15（最多6個主候選＋2個預註冊sentinel）；最多3個進D20；pass／recover才續D60，D60最多2個。
- observer：32/task no-grad calibration → 1 epoch EMA `0.9/0.1` → 5 epochs scale-only → joint。
- Detect logical batch 128；Pose logical batch 16。physical batch 依 OOM 下降，但 accumulation 維持 logical exposure。
- Detect/Pose loss weight固定 `1.0/0.25`。
- quantizer／observer／scale保持FP32。QAT在解決fold-aware contract前禁止執行：必須選`fold_aware_shadow_qat`或`folded_graph_qat`，量到deployment真正使用的`W_eff = gamma×W/sqrt(var+eps)`。BN running statistics固定；affine是否可訓練必須隨方案明定，不能沿用未驗證的「affine trainable」。
- 每個QAT checkpoint重建BN-folded view並重新驗證graph／forward parity。
- gradient clip `10`，scheduler 使用 double warmup、60-epoch cosine、final factor `0.5`、不 restart。

### Optimizer

AdamW control 使用 accepted J3 semantic LR × `0.1`：

| Role | LR |
|---|---:|
| attention | `5e-8` |
| backbone | `3.8e-7` |
| neck | `1.9e-6` |
| MASF | `3.8e-6` |
| Detect head | `5e-6` |
| Pose head | `5e-6` |

`betas=(0.948,0.999)`；model decay params使用weight decay `0.00027`，quant params不 decay。

HPO 最多 8 個短 trial：AdamW 4、MuSGD 4；model LR multiplier `[0.05,0.1,0.2]` of J3，qparam ratio `[0.5,1,2]`，epochs `[5,10,15]`。MuSGD只先測 balanced policy，必須另跑 matched sham，且不能載入 AdamW optimizer state；若 qscale 不穩，才開明確標記的 `MuSGD+AdamW-Q` rescue。

### LS-SD4

- 先完成exact Fixed SD4與exact uniform W4；LS-SD4不能只因勝過十點grid就宣稱有效。
- codebook 固定，`s=softplus(rho)+epsilon`。
- FP32 master＋STE。
- 先 layer-wise；穩定後才做 grouped／per-channel upper bound。
- backbone_deep balanced policy比較 autograd、`1/sqrt(N)`、`1/sqrt(7N)`、LR-only 四種 gradient recipe。

### Ternary QAT

- Paper-TWN、Channel-TWN、TTQ 分開命名與比較。
- T15：3 epochs TW32A、6 epochs progressive ramp、6 epochs fully ternary W/A8。
- D60：10/20/30；paper-random bitmap只作 control，monotonic priority是主線。

### Lower activation bits／A-SD4

weight map完成後才測qSiLU A7/A6、Hardswish A7、poly_shift A7/A6。A-SD4另與LSQ+ A4、APoT A4、Fixed A-SD4、Learned A-SD4作同parent／同calibration／同budget比較；不能把W-SD4結果直接套到activation。ternary分支若使用signed A4，另驗證論文的安全範圍`[-7,7]`。

## 九、Augmentation

主線完全沿用 accepted Full35：HSV `0.015/0.7/0.4`、translate `0.1`、scale `0.5`、Detect horizontal flip `0.5`、Pose flip `0`；mosaic、mixup、cutmix、copy-paste與額外 noise 都為 0。

目前環境沒有 albumentations，因此 Blur／MedianBlur／ToGray／CLAHE 的候選程式實際是 no-op。主線不新增 Gaussian、shot、sensor 或 JPEG noise。

只有 top3 後可另做 matched noise pilot：20%影像隨機使用 Gaussian `2/255–8/255`、JPEG quality `70–95` 或 `3×3` blur；quant與sham同步，clean 八指標任一比 no-noise 差超過 `0.002` 就拒絕。

## 十、Formal、export 與最終優化

- W8第一格之前的CPU integer-boundary contract已完成：Add／Concat reference requant、RNE、saturation、148層INT32 MAC bound及MASF／attention PWL／Binary Q/K保護島均已manifest化。第一個GPU bridge已用固定32＋32 calibration凍結124個LSQ+ observer並完成mAP search；實際boundary saturation、bias／padding correction與native integer graph仍未完成，不能把fake quant結果當成已可部署。
- top3 先以 100% train、seed1 到 100 epochs；top2續到120並跑 seeds `0/1/2`。
- 每個 seed 的八項mAP50 total delta都必須通過 `-0.015`，而且包含activation替換；報 mean／std／worst seed。
- finalist才完成packed artifact與target-specific最佳化；integer feasibility與boundary tax不得延到finalist才第一次檢查。
- W5/W6/W7沒有原生 kernel前不宣稱速度。
- PQ、FREQ、KD、MASF-FREQ 等最終優化目前只保留 seam／計畫；挑出 base finalists 後才做單因子，再做有效組合。

## 十一、停點

目前已完成V0–V3 CPU實作、Hardswish／regional policy CPU parity、G0安全修正、integer boundary CPU reference、三parent exact W4／SD4重算、routing v3、qSiLU歷史PTQ，以及poly_shift W8–W4、mixed-bit、Fixed-SD4／Paper-TWN和耦合PTQ。V19 QAT GPU smoke已通過；首次sham在epoch 0 gate後暴露fused-head checkpoint contract錯誤，已red→green修正、封存失敗run並乾淨重啟。現由每300秒檢查一次的fail-closed佇列監測，完成manifest與best_joint hash通過且GPU空閒後才自動啟動同一V19 QAT。使用者已授權依v5逐步完成；每一階段仍先產生明確cell清單與hash，且formal validation不會被PTQ或QAT search runner自動觸發。

```text
COMPLETE: V0–V3歷史證據、exact solver、SD4 encoding、manifest/gate/runner修正與V4設定
    ↓
COMPLETE CPU P0: integer boundary reference＋三parent exact W4/SD4＋36層routing v3
    ↓ 2026-09-03窄範圍授權
COMPLETE: qSiLU＋A8／backbone early／W8–W4 diagnostic＋完整mAP50 search
    ↓
COMPLETE: 其餘九區isolated W8 diagnostic＋search，8 green／1 recover
    ↓
COMPLETE: poly_shift累積W8 → W7–W4 exact → mixed precision → Fixed-SD4／Paper-TWN
    ↓
COMPLETE QAT P0: fold-aware graph＋雙指標selector＋GPU smoke＋search-view validator
    ↓
IN PROGRESS: matched sham → W8／mixed／LS-SD4 paired QAT → qSiLU／regional Hardswish coupling
    ↓
鎖定finalists → formal／多seed／export
```
