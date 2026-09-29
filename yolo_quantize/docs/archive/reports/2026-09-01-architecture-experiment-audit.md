# Full35 量化架構與實驗配置稽核報告

日期：2026-09-01  
範圍：yolo_quantize 現行程式、Q3 activation 證據、V0–V3 產物、V4+ 矩陣、BBAT5 資料治理與相關第一手文獻  
執行限制：未使用 GPU、未跑 PTQ calibration、mAP validation、QAT 或正式訓練

## 結論

整體方向可以保留，但舊計畫不能直接開跑。最重要的四個問題是：

1. active weight runner 仍指向舊 parent 與 151 層 catalog，而且 PTQ 跑在未 fold BN 的 graph。
2. 單張 raw-output smoke 被寫成 passed，容易被誤讀為精度或 promotion 通過。
3. Fixed SD4 與 uniform 的舊 mse 其實只是十點 grid，對 LS-SD4 是不夠強的 baseline。
4. 150 格若都反覆使用 formal validation 選 winner，會造成 selection leakage。

本次已修正前三項的 CPU 工程 Seam，並把第四項改成 static／diagnostic／search／formal 四層。舊產物與 hash 全部保留，但降格為 mse_grid_v1 歷史證據；active runner也已綁定固定diagnostic manifest。integer boundary是V4 GPU前P0，fold-aware weight則只阻擋QAT，不再把不同階段的blocker混在一起。目前仍不可執行V4。

推薦主線不是先選一個 activation winner，而是三個 A8 完整 parent 各自配 matched FP-weight control：

1. qSiLU：accuracy fallback。
2. Hardswish：standard operator hypothesis。
3. poly_shift：dyadic operator hypothesis。
4. 每個 parent 依 W8 → W6 → W4 建 coarse curve。
5. 再補 W7／W5 完整 boundary curve。
6. 做十區 sensitivity 與 activation×weight interaction。
7. 對 routed regions 公平比較 exact uniform W4、exact Fixed SD4、APoT4 與 ternary。
8. 只有 recovery-band 或特殊格式才進 matched QAT。

因此目前沒有「qSiLU 一定最好」、「Hardswish 一定最快」或「SD4 適合某層」的正式結論；三者都是待驗證的完整 system policy。

## 架構稽核

### 修正後的 Module 與 Seam

| Module | Interface／Seam | 責任 | 稽核結果 |
|---|---|---|---|
| Full35ActivationAdapter | build | 建立指定 activation placement 與 A-bit parent | 保留；Q3 regional 與 uniform parent 不混稱 |
| AppliedActivationQuantization | rebind | 將 observer／freeze 控制綁到等價 clone 或 fused graph | 本次新增；核對 path、policy id、bit 與 signed contract |
| Full35WeightViewAdapter | build | 產生 unfused master 與 BN-folded deployment View | 保留；PTQ runner 已改用 deployment |
| optimal_scaled_codebook_scales | groups＋fixed codebook → exact scales | uniform／SD4／ternary 共用的全域 MSE 尺度 Module | 本次新增；不是十點 grid |
| SD4Encoding | encode／decode／pack／unpack | 保存 16 個 bit patterns、雙 zero 與 canonical export zero | 本次新增 |
| WeightQuantizationAdapter | reversible isolated-region PTQ | 權重量化與完整還原 | 保留；新增 exact scale method |
| Full35MetricGate | candidate＋accepted＋matched＋gate spec | 八指標與 matched sham 決策 | 本次加深；contract／policy mismatch 直接 fail closed |
| WeightSensitivityStudy | versioned YAML → deterministic cells | active plan 解析與執行授權 | 本次改成 v2；預設 execution_authorized=false |

這個分層讓每個 Module 隱藏自己的複雜度：activation replacement 不需要知道 BN fuse，weight runner 不需要重寫 observer 控制，metric gate 不需要猜資料或 parent 身分。Locality 也改善：scale fitting 集中在一個 Module，不再讓 uniform 與 SD4 各自默默維護不同的 MSE 語意。

### 已發現與處理的問題

| 嚴重度 | 問題 | 影響 | 本次處理 |
|---|---|---|---|
| P0 | runner 預設 v1，含 poly_quality、排除 Hardswish、151 層舊 catalog | active execution 和使用者決策相反，且會 catalog mismatch | 新增 v2，parent 改為 qSiLU／Hardswish／poly_shift，master 251 與 deployment 152 分開 |
| P0 | PTQ 直接量化 unfused BN graph | static analyzer 與 GPU probe 不是同一 deployment tensor | runner 先建立 BN-folded deployment View，再以 rebind 控制 A8 observer |
| P0 | 一張 probe finite＋same structure 即標 passed | 容易被誤寫成 mAP pass | 改為 diagnostic_pass，固定 selection_claim=false 與 promotion_authorized=false |
| P0 | CLI 看見 CUDA 即可執行 | GPU 共用期間可能誤跑 | 需要顯式 execute acknowledgement，且 YAML 也必須 authorization=true；目前 v2 為 false |
| P0 | gate 沒核對 evaluator／資料／activation policy | 可把不同 parent 或 split 錯當 matched control | 新增 metric_contract_id 與 policy_id，不一致直接拒絕 |
| P0 | threshold 判斷與輸出各自 hard-code | 設定可能 drift | 新增單一 Full35MetricGateSpec |
| P0 | Fixed SD4 mse 只試十個 scale | LS-SD4 可能只打敗弱 baseline | 舊語意改名 mse_grid_v1；新增 exact scaled-codebook |
| P0 | 文件說 16 codes，但 projector 只留 15 個 logical IDs | 不能宣稱 bit-true export | 新增 16-code mapping、雙 zero、canonical padding 與 round-trip 測試 |
| P0 | runner執行時以path helper臨時找圖，且manifest只有self-declared digest | calibration／probe身分可漂移，內容與self hash可一起被替換 | v2 plan獨立釘住固定32／64 manifest SHA；執行前再驗證manifest與全部檔案hash |
| P0 | YAML宣稱narrow symmetric range，但程式使用完整two's-complement | bit容量與重建結果契約不一致 | 設定改為zero-point 0及W8 `[-128,127]`到W4 `[-8,7]` |
| P1 | zero tensor 的 SQNR 會 log10(0) | static analyzer 可因合法全零 group 崩潰 | numerator 與 denominator 使用相同 finite floor |
| P1 | V5 150 格預設都可跑 full validation | formal-val 被當搜尋集 | 分成 T0–T3；formal 最多 6 個 locked finalists |
| P1 | 真實View測試用activation配錯預設checkpoint | 可能把正確的parent-specific parity誤判為fuse失敗 | 測試改用qSiLU recovery SHA；保留原嚴格門檻並鎖定124個部署observer |

### 稽核後工程狀態與各自阻擋的階段

1. integer boundary CPU reference已完成：Add／Concat、RNE、saturation、148層INT32 MAC bound與MASF／attention／Binary Q/K保護島均已manifest化。第一個未執行工作改為GPU calibration後的scale／offset凍結、bias／padding correction與boundary parity；尚不宣稱native integer kernel。
2. Full35 exact重算已完成：三parent共3,552筆，exact Fixed SD4 routing v3得到36層static候選。它解除「弱grid baseline」問題，但仍需GPU layer-output／八指標驗證才可promotion。
3. fold-aware QAT（只阻擋QAT）：若unfused master的BN affine更新，deployment有效weight為gamma×W/sqrt(var+eps)；QAT必須量到此tensor，或改用folded-graph QAT。
4. hardware target（阻擋速度／能耗winner宣稱）：尚無固定device／runtime／compiler／kernel；目前只能報bytes、metadata、BOP或traffic proxy。

## 實驗矩陣修正版

### V4：uniform parent×bit bridge

| 次序 | cells | 用途 |
|---|---:|---|
| matched FP-weight controls | 3 | 每個 activation parent 自己的 incremental reference |
| W8 | 3 | graph／integer bridge 與低風險 baseline |
| W6 | 3 | coarse 中點 |
| W4 | 3 | coarse 低點；另接 exact uniform comparator |
| W7／W5 | 6 | 保留非 2 冪次位寬，補完整曲線與 Pareto knee |

quantized universe 仍是 15 格，不刪 W5／W7；只是用 W8→W6→W4 先看曲線，再補 W7／W5。

### V4H：Q3 regional Hardswish

- qSiLU checkpoint 不變。
- neck_attention、masf、backbone_attention 各跑 FP-weight control 與 W8，共 6 格。
- 只有 neck_attention 與 masf 各自通過，才加二者組合的 control／W8 兩格。
- 不把單區 delta 相加，也不把 activation region 當作等同 weight region。

### V5：region universe 與 validation budget

完整 universe 是 3 parents × 10 regions × 5 bits＝150 static cells，但權限分層：

| Tier | 最大範圍 | 可以做什麼 | 不可以做什麼 |
|---|---:|---|---|
| T0 static | 150 | weight reconstruction、code／metadata bytes、occupancy | 選 winner |
| T1 diagnostic | 優先 150；最低 90 個 W8/W6/W4 | structural/output probe、排序 | 宣稱 mAP 通過 |
| T2 search | 最多 30 | COCO val＋BBAT canonical search-val 八指標，決定 promotion | 使用 BBAT formal-val 搜尋 |
| T3 formal | 最多 6 | locked finalists 正式確認 | 回頭反覆調參 |

COCO 目前仍可能重用 val2017 作 selection／報告，這是限制；對外正式發表時優先增加 COCO test-dev 或明確獨立 test contract。

### 特殊格式

被 region routing 選中的層至少比較：

- optimal uniform W4。
- optimal Fixed SD4。
- APoT4。
- Paper-TWN。
- exact-scaled ternary。

LS-SD4 只能在 exact Fixed SD4 之後開始。可研究的重點不是「學一個 scale」本身，而是 activation-conditioned、task-output-loss、worst-task constraint 與 fold-aware deployment 是否帶來額外收益。

A-SD4 仍延後；其 matched 矩陣是 LSQ+ A4、APoT A4、Fixed A-SD4 與 Learned A-SD4，不能由 W-SD4 結果外推。

## 指標與判定

每個可選擇 cell 必須報八項：COCO box、COCO person、BBAT box、BBAT pose、ball box、bat box、ball pose、bat pose 的 mAP50-95。

硬門檻維持使用者決定：

- 每項相對 accepted Full35 總下降不超過 0.04；這是 0–1 AP 座標的 4 AP points，不是相對 4%。
- W8 每項相對 matched activation parent 下降不超過 0.01。
- QAT matched sham 每項 absolute drift 不超過 0.01。
- green必須同時通過total與適用的W8 incremental gate；若尚未green但worst total drop不超過0.06則recover，因此total已過但W8 incremental未過也不是green；超過0.06才拒絕。

static MSE、NRMSE、SQNR、TopK overlap 與 single-image output error只能作診斷，不可取代八項 AP。

## QAT 配置

QAT 不是重新從零訓練；只有 PTQ 落入 recover band 或 SD4／ternary 等特殊格式才進入。

| 階段 | 上限 | epoch |
|---|---:|---:|
| S15 | 8 個＝最多 6 個主候選＋2 個 sentinel | 15 |
| D20 | 3 個 | 20 |
| D60 | 2 個 | 60 |
| formal top2 | 2 個、seeds 0/1/2 | 100，必要時 120 |

其餘保留既有 matched recipe：

- AdamW 為 control；semantic LR 是 accepted J3 各 role 的 0.1×。
- MuSGD 只作另立版本的 pilot，必須有自己的 matched sham，不載入 AdamW optimizer state。
- Detect logical batch 128；Pose 依實體顯存試 16→8→4，gradient accumulation 保持 logical exposure。
- Detect／Pose loss weight 為 1.0／0.25。
- augmentation 沿用 Full35：HSV、translate、scale、Detect flip；mosaic、mixup、cutmix、copy-paste 與額外 noise 皆為 0。
- 不因「可能更 robust」加入 noise；只有 locked top3 才可另做 matched noise pilot，而且 clean 任一指標差超過 0.002 即拒絕。

## 研究方向

完整第一手來源與新穎性邊界見[量化文獻與研究新意稽核](../../research/2026-09-01-quantization-literature-and-innovation-audit.md)。最值得優先做的四個可否證候選是：

1. activation-conditioned、worst-task-constrained mixed format routing。
2. shared Detect＋Pose 的 joint task-output-loss block PTQ。
3. mse_grid_v1 → exact PTQ → activation-weighted scale → task-loss LS-SD4 的 exact-to-learned continuum。
4. Q3 regional activation placement×weight bit 的二因子 interaction。

其中第一與第三最貼近目前已有資產；第二可能有較高精度收益但實作成本較高；第四最直接回答「activation 與 weight 到底是否強耦合」。這些只能稱研究假說，不能在沒有更完整 prior-art／專利查核與實驗前宣稱新穎。

## 現在應該怎麼做

原列的兩個CPU P0已於本稽核後完成：

1. `IntegerBoundaryContract`與hash-bound Full35 manifest已完成。
2. 三parent exact uniform W4／Fixed SD4 static profile與routing v3已發布；舊v2未覆寫。
3. 目前依指示停在V4 W8 GPU bridge前；下一步需GPU與明確執行授權。
4. fold-aware shadow QAT或folded-graph QAT仍須在任何QAT前決定，但不阻擋先做PTQ GPU bridge。

## 驗證與限制

- red→green 測試涵蓋 activation rebind、metric contract、custom threshold、exact codebook、zero tensor、SD4 16-code、manifest型別／self hash／plan pin、active runner guard、乾淨CLI與hash-bound qSiLU View parity。
- exact solver 以小型獨立 assignment 暴力搜尋核對全域誤差，並驗證不劣於 mse_grid_v1。
- 沒有刪除／改寫舊 artifact、checkpoint 或 dataset。
- 沒有使用 GPU、validation 或訓練。
- 新routing v3已有36層static候選；目前仍沒有量化後mAP、正式activation winner或硬體latency／power結果。

本輪CPU P0的完整數值、artifact SHA與限制見[`2026-09-01-cpu-p0-integer-exact-routing.md`](2026-09-01-cpu-p0-integer-exact-routing.md)。
