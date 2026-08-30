# SIPA–BCSP 學術主張與完整實驗計畫

日期：2026-08-25
狀態：預註冊草案；Phase 0 純函數／固定點／ONNX 已完成，真實 detection 與板上實驗等待
baseline 資料夾及 target profile
證據邊界：本文區分「數學上已證明」、「原型已量測」與「模型／硬體待驗證」，不把工作名稱
或組件組合直接當成新穎性證明。

## 1. 學術上真正可能成立的主張

目前不能安全主張「使用 SiLU 對稱性」、「多項式近似」、「dyadic 係數」、「YOLO 混合
activation」或「Pareto 搜尋」本身是新的。完整 primary-source claim chart 見
[SIPA–BCSP 新穎性稽核](sipa-bcsp-novelty-audit.md)。本研究的候選貢獻單位必須是下面的完整、
可否證組合。

### C1：受硬限制推導的 SIPA 函數族

不是先自由擬合曲線，而是把下列條件同時設為解析硬限制：

\[
A(x)=\frac{x}{2}+H(|x|),\qquad A(x)-A(-x)=x,
\]

- 原點值與斜率固定；
- 有限門檻外精確接到 ReLU；
- 在兩個尾端保持函數、一階及二階導數接合；
- 只有少量共享 shape／threshold 自由度；
- 存在一個 power-of-two threshold、dyadic/APoT 常數的可部署成員。

其中任何單一元素都有相鄰工作；可能的學術貢獻是「上述限制共同定義的低自由度 detector
activation family」及其 accuracy–hardware 性質。是否真的新穎仍須完整文獻／專利檢索與實驗支持。
尤其 SmeLU 已有 exact-ReLU tails，S-ReLU 已有 \(C^2\) polynomial exact tails；SIPA 必須靠
zero anchor、SiLU 式非單調負谷、受約束積分族與整數不變量的完整組合區辨。

### C2：保持解析不變量的固定點實現

`poly_shift` 的 Q-format datapath 不只最小化平均誤差，也把下列兩項設為 bit-level 驗收條件：

\[
Q(A(x))-Q(A(-x))=Q(x),
\]

以及正負兩側 finite ReLU tail 的 0-LSB 誤差。這可成為次要貢獻，但必須與一般 round-to-nearest、
獨立輸出量化及同位寬 PWL／LUT 比較後，才能判斷是研究貢獻或單純實作技巧。

### C3：硬體與跨任務限制下的靜態 placement

目前 BCSP 已做到逐資料集 Pareto、region-level static policy、beam width、changed-region 與 kernel
budget；這本身較接近良好工程，不能只靠名稱宣稱新演算法。要升格為主要演算法貢獻，正式版本
將加入跨任務 worst-case normalized regret：

\[
r_d(p)=\frac{AP_d(\mathrm{SiLU})-AP_d(p)}{\delta_d},\qquad
R(p)=\max_{d\in\{\mathrm{COCO},\mathrm{BBAT5}\}} r_d(p),
\]

在不平均兩個資料集 raw AP 的前提下，搜尋最小化
`(R(p), measured/proxy hardware cost, kernel count)` 的共享靜態 policy。這項延伸必須正面比較
ActNAS-style layer search、random、greedy、uniform 與 per-dataset oracle，才可能形成獨立貢獻。

## 2. 核心研究問題與反證條件

| ID | 研究問題 | 支持證據 | 反證／失敗條件 |
| --- | --- | --- | --- |
| RQ1 | SIPA 的共同硬限制是否比 Hardswish 或同成本近似更忠實地保留 SiLU？ | empirical-distribution function／derivative error、tail、range | 只在均勻網格好，實際 feature distribution 不改善 |
| RQ2 | `poly_shift` 經等成本 recovery 後能否留在兩資料集各自的 accuracy gate？ | COCO 與 BBAT5 各自 stage-matched ΔAP 與 CI | 任一資料集超過預先凍結的 loss budget |
| RQ3 | symmetry、C2 tail 與 dyadic projection 是否各自有必要？ | matched-cost ablation | 拿掉限制沒有穩定差異，則不得把該限制列為貢獻 |
| RQ4 | BCSP 是否用較少 trial 找到不被 uniform／greedy／random 支配的 policy？ | search regret、trial 數、wall time、frontier hypervolume | 同 evaluation budget 下沒有優於簡單搜尋 |
| RQ5 | 一個共享 policy 是否能跨 COCO／BBAT5 穩健，而非只對單一資料集過擬合？ | independent、shared-profile、shared-policy 三組 | shared policy 的 worst-case regret 超過 budget |
| RQ6 | dyadic／整數 datapath 是否在指定 target 真的有端到端效益？ | latency、energy、memory、DSP/LUT/BRAM、AP | 只有 operator proxy 改善，實機沒有改善 |

所有負結果照樣報告；不能因結果不利而改 family 定義、accuracy budget 或主要指標。

## 3. 固定研究邊界

### 3.1 模型與 activation site

- 起點是使用者交付的不可變 baseline folder；不預設其一定是 YOLO26m。
- 只替換人工核准 manifest 中原本為 `nn.SiLU` 的 site。
- softmax、detection output sigmoid、raw output Conv、`act=False` 永不替換。
- 每個實驗重新 strict-load 相同 checkpoint，再注入 policy；不沿用上一個候選的 mutated model
  或 optimizer state。
- region 先固定為 `early`、`deep`、`attention_ffn`、`neck`、`head_one2many`、
  `head_one2one`；實際不存在的 region 不產生實驗，多重匹配直接阻擋。

### 3.2 資料

- Canonical BBAT5 v1 Detect 只使用
  `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml`；維持 formal assignment，
  不重切、不抽樣、不改 labels，也不建立 test split。
- COCO80 Detect 只使用 `/home/uxin/yolo/coco2017.yaml`。
- 兩資料集分開 checkpoint、訓練、evaluation table 與 frontier；不平均 raw AP。
- profile 統計只來自 training split；validation prediction 不回饋 profile 擬合。若日後需要較小
  profiling view，必須先取得明確授權並保存 image-list hash。

### 3.3 精簡候選

| 角色 | Policy | 使用階段 |
| --- | --- | --- |
| Accuracy reference | all-SiLU | 所有 stage 的 stage-matched control |
| Hardware-neighbor control | all-Hardswish | zero-shot、short recovery、可能進 finalist |
| Primitive floor | all-ReLU | zero-shot/export；預設不 recovery |
| Primary proposed | all/region/mixed `poly_shift` | 全流程 |
| Coefficient ablation | all-`poly_quality` | zero-shot、short recovery；預設不做 region search |

PReLU、PWLU、FReLU、Mish、GELU、ReLU6 與動態 activation 不進第一輪。只有主要候選失敗或
claim-critical ablation 需要時，才選一個對應先前技術補位。

## 4. 分階段實驗漏斗

### E0：已完成的數學／匯出 gate

- float64、`[-12,12]`、24,001 點的 function／derivative／tail 檢查；
- SiLU、Hardswish、ReLU、`poly_shift`、`poly_quality` operator proxy；
- Q16.10 integer emulator、symmetry 與 tail LSB；
- ONNX opset 18 graph，禁止 proposed path 出現 `Exp/Sigmoid/Div`。

這一階段只支持「公式與 reference implementation 可運作」，不支持 detection AP 或硬體加速。

論文主張前仍須補 M0 formal gate：證明四個 boundary／integral constraints 的解空間、可行
\(a\) 範圍、負谷與導數 extrema，並判定 degree-4 derivative／degree-5 integral 是否為滿足
zero anchor、可調 non-monotonicity 與 \(C^2\) exact tails 的最小自由度；若不是就縮減函數族。

### E1：交付契約與 baseline reproduction

1. 稽核 model config/source、checkpoint SHA-256、framework commit、環境版本、兩資料集 YAML／
   fingerprint、recipe、baseline metrics 與 activation manifest。
2. strict-load checkpoint，確認所有 tensor key／shape；輸出 eligible、unmapped、excluded site。
3. 在原 evaluator 重現 all-SiLU validation。重現容忍值定義為
   `max(報表顯示精度的一個最小單位, 兩次 deterministic evaluation jitter 的兩倍)`，並在看候選
   結果前寫入凍結設定。
4. baseline 無法重現就停止；不得用重切資料、換 evaluator 或手動調分數規避。

### E2：activation distribution 與 range profiling

- 在各資料集 training split 收集每個 manifest site 的 count、min/max、quantiles、直方圖、
  negative/zero/tail ratio、tensor elements 與 memory traffic proxy。
- 以相同 bins 計算 function error、derivative error、overflow margin 與各 region cost weight。
- profile 統計只用來解釋及建立 hardware proxy；第一輪不重新 fit `poly_shift`，避免把 profile
  tuning 與 placement 同時改動。

### E3：uniform zero-shot

每個資料集從同一乾淨 checkpoint 評估四個替換：Hardswish、ReLU、`poly_shift`、
`poly_quality`。固定 imgsz、batch、precision、BN mode、NMS、augmentation 與 evaluator。

Zero-shot 只淘汰 graph/export 失敗、NaN/Inf 或災難性錯誤；不因一次較差 AP 就提早否定可恢復的
固定函數。ReLU 到此為止，保留作 primitive floor。

### E4：等成本 short recovery

- 同時跑 all-SiLU sham-recovery、Hardswish、`poly_shift`、`poly_quality`。
- 每個 run 都從原交付 checkpoint 開始，重新建立 optimizer/scheduler。
- 暫定 budget 為原完整 recipe optimizer steps 的 10%，上限 10 epochs；所有候選使用相同步數。
- 初始 LR 暫定為原 recipe 的 0.1 倍；optimizer、weight decay、augmentation、EMA、BN 與 image
  size 其餘不變。實際數字須在 baseline pilot 後、看候選結果前凍結。
- 所有 short-recovery ΔAP 以同 budget SiLU sham-recovery 為基準，不能只和未繼續訓練的交付
  checkpoint 比較。

### E5：region sensitivity

只讓 `poly_shift` 進 region search。對每個實際存在的 region 做 one-at-a-time replacement，其餘
site 保持 SiLU，並使用 E4 的相同 short-recovery budget。六區全存在時每個資料集最多六個 job。

另計算一階替換 proxy：

\[
s_r=\mathbb E\left[\left|\frac{\partial L}{\partial h_r}
\left(A(h_r)-\operatorname{SiLU}(h_r)\right)\right|\right],
\]

只把它當 cheap ranking，不直接當 AP。以 Spearman correlation 比較它和 region recovery 的真實
ΔAP；若無相關，後續不得用該 proxy 剪枝。

### E6：BCSP mixed-policy search

- 只從當前 non-dominated region policy 擴充一個 region；
- `beam_width=4`、最多三個 changed regions、最多三種 deployment kernels；
- 正式執行前新增每資料集最多八個 mixed-policy evaluation 的總上限；目前程式只限制每輪四個，
  在補上總上限前不得啟動昂貴 search；
- mixed policy 全部使用 stage-matched SiLU short-recovery control；
- accuracy budget `δ_COCO`、`δ_BBAT5` 與 AP_S budget 在 E1/E4 pilot 後預註冊，正式結果後不得改。

若兩資料集的 manifests 可語意對齊，追加三種跨資料集設定：

1. `D0 independent`：profile 與 assignment 各自搜尋，作 oracle 上界；
2. `D1 shared-profile`：共享 `poly_shift` 公式，assignment 可不同；
3. `D2 shared-policy`：公式與 region assignment 都共享，以 worst-case normalized regret 決策。

### E7：finalists

每資料集最多保留兩個非支配 finalist：accuracy extreme 與 hardware extreme。每個 seed 都必須有
同 recipe、同 seed 的 all-SiLU control。

1. seed 1：完整 recovery／完整官方 recipe；
2. seed 2：只在資源允許時，對 SiLU 與同一組 finalists 執行；
3. 若日後要做正式穩定性或 SOTA 主張，1–2 seeds 不足，必須另行授權增加 seeds；
4. 若使用 from-scratch，SiLU 與 finalists 都必須 from-scratch，不能混用預訓練與 recovery。

### E8：claim-critical ablation

只對最終 proposed policy 執行，不重新打開 activation zoo：

| Ablation | 控制變因 | 回答問題 |
| --- | --- | --- |
| symmetry | 同 degree、同 operator count 的 unconstrained asymmetric polynomial | 差分對稱硬限制是否有價值 |
| tail smoothness | matched-degree C0、C1、C2 variants | exact-ReLU／C2 接合是否影響 recovery、range 或量化 |
| coefficient | `poly_quality` float → dyadic projection → `poly_shift` | 硬體係數限製造成多少 AP／曲線代價 |
| placement | uniform、best-single-region、greedy、random、BCSP、ActNAS-style | 改善來自 activation 或搜尋方法 |
| profile sharing | per-layer oracle、per-region shared、global shared | 少 profile 的代價與泛化 |
| search budget | 2/4/8 mixed trials；縮小問題另做 exhaustive oracle | BCSP 的 sample efficiency 與 regret |
| prior art | SmeLU、S-ReLU、matched-cost PWL／GRAU-style PoT/APoT | SIPA 是否超越最接近的已知方法，而非只勝過 Hardswish |

所有 search baseline 使用完全相同 trial budget；不得拿 BCSP 的八次搜尋和 random 的一次相比。

### E9：量化與 pre-board deployment

1. FP32 reference；
2. float graph + dyadic constants；
3. Q16.10 bit-exact emulator；
4. target-supported INT16／INT8 PTQ calibration；
5. 只有 PTQ 超過凍結 accuracy budget時才做 QAT；
6. export ONNX，核對 forbidden ops、dynamic gate 移除、one2many fuse 規則及 graph hash；
7. CPU/CUDA/ORT 等僅報各自實測，不外推到未指定板卡。

### E10：上板驗證

Target profile 到位後才凍結 backend、compiler、clock、precision、batch、imgsz、warm-up、repeat、
synchronization 與 power protocol。至少比較 fused SiLU、Hardswish、uniform `poly_shift` 與 BCSP
finalist：

- 端到端 median／p95 latency、throughput；
- peak memory、模型與 coefficient storage、memory traffic；
- power 與 energy/inference；
- FPGA/HLS 時另報 LUT、FF、DSP、BRAM、Fmax、cycle 與 bit-exact mismatch；
- AP 必須使用實際部署輸出重新 evaluation，不能只驗 activation core。

只有此階段可使用「在 target X 上 hardware-efficient」；在此之前只能稱 hardware-oriented 或
shift/add compatible。

## 5. 指標與統計

### 5.1 Detection

- COCO：AP@[.50:.95]、AP50、AP75、AP_S、AP_M、AP_L、必要 class AP。
- BBAT5：formal-val aggregate、ball AP、bat AP、AP50、AP75；明記沒有 test split。
- 每個表都分開 zero-shot、short recovery、full recovery、PTQ、QAT。
- 同時記錄 loss、gradient norm、NaN/Inf、收斂速度、activation tail/saturation/negative ratio。

### 5.2 低 seed 條件下的誠實統計

- seed 1 是探索主結果；seed 2 只補 finalists。
- 對同一 validation image IDs 的 baseline/candidate predictions 做 paired image-level bootstrap
  （預設 2,000 replicates、固定 bootstrap RNG），報 ΔAP 95% CI；這只反映 evaluation-set
  uncertainty，不能冒充 training-seed variance。
- 有兩個 training seeds 時，逐 seed 與同 seed SiLU 配對，報 mean、range 與所有原始值；不以
  `n=2` 做誇張的顯著性宣告。
- 所有失敗、OOM、export failure、NaN 與被 gate 淘汰者保留在 machine-readable results。

### 5.3 選型規則

候選只有在下列條件都成立時才可晉級：

1. 兩資料集各自通過預先凍結的 AP／AP_S budget，或被清楚標成 dataset-specific policy；
2. 不被同階段、同 seed 的另一候選在 accuracy 與完整 hardware objective 同時支配；
3. export graph 合法，沒有 runtime dynamic gating；
4. 若宣稱硬體效益，必須有 target measurement 及信賴區間／重複量測；
5. 若宣稱 SIPA 機制有效，claim-critical ablation 必須支持，而非只看最終 AP。

## 6. 預估實驗數量

假設六個 regions 全部存在、mixed search 總上限八、最多兩個 finalists：

| Stage | 每資料集最多 jobs | 成本類型 |
| --- | ---: | --- |
| SiLU reproduction | 1 | evaluation |
| Uniform zero-shot | 4 | evaluation |
| SiLU + 3 candidates short recovery | 4 | short training |
| Region sensitivity | 6 | short training |
| BCSP mixed search | 8 | short training |
| SiLU + 2 finalists seed 1 | 3 | full training/recovery |
| SiLU + 2 finalists seed 2 | 3 | optional full training/recovery |
| **合計** | **29** | seed 2 不執行時為 26 |

兩資料集最壞上限為 58 jobs，其中 10 個是無訓練 evaluation、36 個是 short recovery、6 個是必要
seed-1 full jobs、6 個是 optional seed-2 full jobs。Ablation、PTQ/QAT 與上板是 finalist 後的獨立
預算，不在 58 內；每一類在啟動前另列 job manifest。實際 region 少於六、frontier 提早收斂、
candidate 失敗或只剩一個 finalist 時，planner 必須自動減少 jobs，不能為填滿配額而新增候選。

## 7. 執行順序與停止條件

```text
delivery audit
  → math/model integration
  → per-dataset SiLU reproduction
  → uniform zero-shot
  → stage-matched short recovery
  → region sensitivity
  → bounded BCSP search
  → seed-1 finalists
  → optional seed-2
  → claim-critical ablation
  → PTQ；必要才 QAT
  → target-board validation
```

立即停止並回報的條件：baseline 無法重現、manifest 未核准、dataset fingerprint 不符、候選產生
NaN/Inf、比較缺少 stage-matched SiLU、search 超出凍結 trial budget、validation 被用於重新擬合、
或硬體 measurement protocol 尚未凍結。

## 8. 目前已完成與尚未完成

已完成：SIPA float reference、dyadic profile、Q16.10 emulator、ONNX gate、manifest、安全替換、
逐資料集 BCSP planner、delivery audit、toy dry-run 與自動測試。

正式執行前仍要完成：

- production YOLO loader/train/val/export adapter；
- 最小度數／可行參數範圍的形式化證明，以及 SmeLU／S-ReLU reference；
- stage-matched SiLU sham-recovery job 與 ΔAP reference；
- BCSP 每資料集 mixed-search 總 job cap；
- empirical activation profiler 與一階 region proxy；
- paired-bootstrap result pipeline；
- robust shared-policy objective；
- target-specific benchmark／HLS/RTL adapter。

這些未完成項目不能被文件敘述取代；baseline folder 到位後必須先通過 E1，才可啟動 E2 以後的
真實模型工作。
