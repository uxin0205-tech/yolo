# 三類優化：詳細實驗順序與停止條件

日期：2026-09-08。person-only 與第二輪暫緩，保留 COCO80 Detect＋BBAT5 Pose。原生、低 LR 與 BN 對照已完成，沒有新 BEST，GPU 已停止；下方歷史／另案模型實驗仍是未來規劃。

> **本輪當前狀態（2026-09-08）：** `PSEL` 已選定為原 J3 `best_joint` EMA，並完成同口徑重驗。原生對照 E1–E5、LR×0.25 對照 E1–E2 及 BN-only 評分均已完成；精度安全線未過，沒有新 BEST，目前 GPU 已停止，`training_ready=false`。結果見[native5 報告](<native5-results.md>)，後續順序見[方向1 master plan](<direction1-master-plan.md>)，optimizer 規則見[optimizer policy](<optimizer-policy.md>)。HOG 尚未正式訓練；未來仍從原 PSEL 做獨立比較，不接退化 E2／E5，也不要求原生對照必須先超過原 BEST。
>
> 正式 baseline 維持 AdamW guard；MuSGD 雖已有 builder 支援，仍是另案 paired challenger，不強制、不以大 batch 觸發，尚無 AP 證據；中途切換的 optimizer state／LR 校準尚待驗證。「BN 較大」只指模型／參數改動較大，不是 batch 或 BatchNorm。HOG、MASF 移位、Q/K、optimizer 不同時首次開。
>
> 本頁的 `P-SRC`（無 MASF／FP-QK source）、重建 J0、舊 LR 配方與 J0/HOG 主訓練均保留為歷史／另一個因果研究流程，不是本輪既有權重 recovery 的必要步驟。新 HOG 名稱為 `W-HOG10`，不與歷史 `F2-PRE-HOG9` 混用。MASF 移位須先由既有模型取得可接受的 no-MASF recovery bridge；Q/K baseline config 明確禁止 `STE=true`，先留診斷／另案 challenger，不說直接改 YAML 可跑；第二輪 KD 僅在有 teacher 與有效梯度時啟動。新 recovery 的 hooks、梯度與超參數以[訓練模型恢復計畫](<../trained-model-recovery/plan.md>)為準。
>
> 以下 S0–S8 及其 P-SRC／J0／HOG／MASF／QAT 參數只適用於上述歷史／另一個因果流程；本輪 recovery 的後續參數不由本頁推定。

本頁 S0–S8 與代號僅供歷史／另案追溯；本輪 parent、順序與 optimizer 以[方向1 master plan](<direction1-master-plan.md>)及[optimizer policy](<optimizer-policy.md>)為準；不建立自動 queue。

## S0：把現有證據與未來實驗分開

現在可做且本次已做：讀取報告、CSV/JSON、YAML與程式；核對互相矛盾的規格；整理工作順序。後續可安排的 CPU 工作包括重算既有 prediction JSON 的 metrics、檔案 manifest與純數值單元測試，均須另列實際執行結果。本次文件檢查不冒充這些模型驗證。

已核對的基準來源：

| 來源 | 用途 | 不可拿來做什麼 |
|---|---|---|
| Full35 RELEASE_STATUS／gate-deltas.csv | 現有 joint seed0 的八項正式指標 | 不當成乾淨 FP-QK baseline |
| Full35 inference/full-resume paths | 區分推論、續訓與 exact state | inference checkpoint 不可 exact-resume |
| V1-DYN/SHEAD/P2 舊報告 | global dynamic 僅比 P2 +0.000505537 的历史 scale 證據 | 不把新 head／graph 的 effect當成同一 delta |
| V1-BR retained checkpoint 的歷史路徑 | 舊 binary lineage 的 site 診斷候選 | 不是新 FP-QK parent；本次未載入驗證 |
| W-DIR metrics | 證明先前已做過 full-model recovery | checkpoint 已刪除，不能承接 |

Full35 internal Bit-True：COCO overall 0.4980223365、person 0.6203814884、BBAT box 0.6300355218、pose 0.9037171741；這些值来自[原始 gate CSV](<../../../yolo_combine/final/full35/metrics/gate-deltas.csv>)。與 standalone 相比 person -0.005804、ball pose -0.016354，是既有 joint 回歸，不等於 BinaryQK-only 的因果掉點。

所有未来表格保存 evaluator 名稱／版本／參數、backend、dataset registry、parent、source history、圖配置、seed與實際 steps。canonical COCO person AP與這些 internal AP另欄保存。

## S1：建立合法 parent，不改任務

主線保留 80-class Detect 和 2-class ball/bat Pose，imgsz 640、P3/P4/P5、現有資料 assignment。

資料入口固定如下，不重新切分、抽樣或修改影像／標註：

- COCO80 Detect：`/home/uxin/yolo/coco2017.yaml`。
- BBAT5 Pose：`/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml`。
- BBAT5 ball/bat Detect 正式入口：`/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml`；只在相應 Detect 工作需要時使用，不取代 joint 的 COCO80 Detect。
- 全域 registry：`/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`；runtime View 只能保留 canonical assignment 與 labels，不成為新資料版本。

| 符號 | 定義 | 本次狀態 |
|---|---|---|
| P-HIST | 目前 Full35，含 shared MASF／BinaryQK | 文字 artifacts 已核對；只作歷史診斷 |
| P-SRC | 經稽核的來源配對及無 MASF、FP-QK joint 建圖契約 | **未驗證可直接取用的 checkpoint** |
| P-J0 | 由 P-SRC 完成 J0 的 full-resume，aux 已註冊但關閉 | 尚未建立 |
| P-TRAIN | F/G 比較勝出、J3完成、aux dormant 的未 fuse模型 | 尚未建立 |
| P-REP | 有做 RepConv 則用其 winner，否則等於 P-TRAIN | 尚未建立 |
| P-MASF | MASF/control比較勝出、未二值化／未 fuse的 checkpoint | 尚未建立 |
| P-QAT | P-MASF 經 site policy＋QAT/KD 決選的 checkpoint | 尚未建立 |
| P-DEPLOY | fuse／係數固定／export後的最終模型 | 尚未建立 |

FP-QK 指 score 使用 qᵀk／sqrt(d)，不是 Float 字樣。PWL normalization、relative bias與V路徑須在FP/binary兩側一致，只改 QK 是 BinaryQK消融；若同時換softmax/PWL則是另一因素。

現有 `_HARDWARE_FROZEN_PARTS` 對 qkv.q、qkv.k、score.gamma 有獨立凍結規則。P-SRC 若使用不同 attention class，必須先驗證 stage role adapter；不能用字串找不到參數就當已全解凍。取得／建立這組合法 parent是排實驗前置，未完成就維持 proposed，不能以現有 shared-MASF權重直接搬位置。

## S2：共同 J0 與控制配方

沿用本地 J-stage 上限：J0 8、J1 20、J2 80、J3 20 epochs，來自 [stage_policy.py](<../../../yolo_combine/src/yolo_combine/stage_policy.py>)。這是上限，歷史 early-stop長度不是新候選的必然長度。

1. F0/F1/F2/G1 共用相同 graph/code、aux 初始化、optimizer groups與 P-J0。
2. 首輪固定原 Conv、无 MASF、FP-QK；保持資料、augmentation、task weight、AMP、logical batch与BN policy。
3. 正式 J-stage 間延續 optimizer moments；不是每到一個 stage 就 fresh optimizer。
4. 為省計算，先跑 F0，鎖定該 seed 各 stage 的實際更新步數及 LR trace，再讓候選重播同一長度／trace；候選不因 val較差而延長。相同 early-stop規則不等於相同步數，報告需區分。
5. 若要保留候選各自 early-stop／plateau，則這是 matched-recipe比較，不能宣稱嚴格等steps單因子；首轮優先前述 F0 trace replay。

J0只更新 Pose head；J1開Neck和heads；J2開late backbone；J3低LR收尾。既有 shared BN running stats 維持 eval，trainable head BN按stage更新。

## S3：訓練方法先選一個

### T-HOG：先 F0/F2，有效果才補 F1

為配合 MASF 主訓練後加入，本整合版使用 **raw P3**。新處理 ID 為 F2-PRE-HOG9；舊 post-MASF F2-HOG9 只作設計參考，不混用名稱／數據。

| Arm | 原生loss | Auxiliary | Projection |
|---|---|---|---|
| F0-MATCH | 原配方 | 關閉forward、mu=0；aux role已註冊 | off |
| F2-PRE-HOG9 | 同F0 | raw P3→Conv1×1(256→9)，GT box內HOG分布監督 | off |
| F1-PRE-LUMA9 | 同F0 | 相同head/mask/loss，9-bin局部亮度target | off |

固定：cell8、9個unsigned orientation bins、在完成augmentation的RGB上產生target、每圖正規化後batch sum。mu：J0/J1 warmup=0；J1下一個epoch升至mu0後保持；J2前8 epochs降至0；J2其餘與J3=0。這個1-epoch ramp是本整合版的明確提案，尚未驗證。

工程前置要補足的兩個細節：

- HOG的普通L2 block normalization不自動等於9-bin概率分布。CE前需非負且sum_bins=1，空能量cell權重0；記錄實際算法。若只是把每cell乘共同contrast因子後又L1 normalize，因子會相消，不能宣稱保留MaskFeat相同的contrast機理。
- F1/F2共用同一mu若gradient尺度差很大，不能把收益差全歸因於orientation。先做train-only的gradient-ratio校準；兩臂使用相同校準規則，目標中位數0.1（接受0.05–0.15），係數鎖定後不看val調整。這是相同優化負擔下比較兩種target，不是完全相同數值mu。

有效cell／對齊／batch-sum／mu=0等價／AMP replay／resume必須先過。HOG head很小不代表便宜：回傳到P3仍增加backward graph及target成本。空框影像只讓aux=0，native negative loss照常存在。

F0先完成，F2依相同trace跑。只有F2相對F0出現joint或person至少+0.001的訊號，才補F1。正式通過需joint≥+0.001、八項各≥-0.001、person或ball pose≥+0.002。F2若未勝F1，只能說companion recipe有收益，不能宣稱HOG專有效或論文級創新。

aux在J3保留為dormant module以維持resume groups；移出部署模型需顯式strip。若後續recovery從strip版新建optimizer，那是warm start：兩臂同樣新建，不能稱exact resume。

### T-CONFLICT：只在量測支持時追加

沿用COCO80 Detect／Pose的原生梯度。觸發條件：至少**同一個stage**負cosine率≥20%，且該stage負事件correction ratio中位數≥0.02；finite與scope一致。歷史32–41%負事件率不直接當新parent的trigger。

G1-APC從P-J0重跑相同J1→J3；關閉HOG，只對active shared Pose梯度的負投影分量修正。Detect及兩個task heads不回寫。若F0的graph、aux groups、state、trace、BN與data全部相同，可把F0作G0別名，新增1個job；否則G0/G1需要2個。

優先在HOG未通過且衝突仍有實質幅度時開G1；HOG已通過就先保留，投影列下一輪。兩者都研究時各自獨立，首輪不加組合arm。Gate同上並核對dot/correction數值；AdamW下raw gradient正交不保證實際update不衝突，報告不作強保證。

## S4：RepConv是條件式增準實驗

若首輪GPU預算只容主要三線，跳過S4。它融合後MACs與原Conv相同，沒有增準就缺乏加入理由。

| Arm | layer17 | layer20 | 起點 |
|---|---|---|---|
| R0 | Conv | Conv | 同一P-TRAIN |
| R1 | RepConv無identity | Conv | 同一P-TRAIN |
| R2（條件式） | Conv | RepConv無identity | 同一P-TRAIN，不從R1接續 |
| R3（條件式） | RepConv無identity | RepConv無identity | 同一P-TRAIN，不拼R1/R2權重 |

首輪只R0/R1；先layer17是靠近P3的工程優先順序，不是既有AP證明。layer20只有在R1通過且仍有跨尺度問題、或明確要研究site差異時加R2。R1/R2單獨都過才考慮R3。

共同scope：Neck＋Detect/Pose heads，backbone／attention凍結；新1×1 branch歸Neck，shared BN stats沿舊policy固定。首版固定5個recovery epochs、AdamW、neck LR=1.9e-5、head LR=5e-5，1 epoch warmup、兩臂相同LR trace；這是沿J3量級的**待驗證提案**，不是以前已成功的5-epoch配方。安全probe失敗停止，不暗中增加epochs。

3×3 branch複製原Conv+BN；新增1×1 branch BN gamma/beta=0；stride2不含identity。保證初始函數／state locality／fuse tolerance；融合後按八項完整validation和PTQ增量損失檢查。R1需overall≥+0.001、八項≥-0.001且AP_S／sports-ball／baseball-bat無犧牲。只非劣、沒有實測部署好處就保留R0。

## S5：MASF主訓練後兩臂 recovery

沿用[原MASF計畫](<../p3-masf-detect-entry/plan.md>)的主要機制，parent用P-REP，不得含shared-seam MASF，QK仍為FP。

| Arm | Detect P3輸入 | Trainable |
|---|---|---|
| CTRL-P3 | raw P3 | P3 box/class predictor，one-to-many與one-to-one |
| MASF-P3 | raw P3→Partial75 MASF | 同一P3 predictor＋MASF |

DualHead graph的owner應是Detect head內的p3_masf；舊單head的model.23.cv2等名稱需解析到實際owner，不可直接套白名單。Pose/P4/P5 heads、raw P3 producer及其下游主Neck全部frozen。

初始gate=0時output等價；gate-on/off不能改raw P3/P4/P5或Pose輸出。gate為0的第一步context branch梯度為0是數學結果，須記錄gate何時離零，而非把「沒有梯度」直接判bug。

P3 predictor LR=1.9e-4、MASF LR=3.8e-4為舊提案。舊計畫寫「沿用Stage A長度」但沒有可直接核對的固定epoch數；本次明確定義首版**5 epochs** matched recovery作新規格，1 epoch warmup、同一scheduler、不另掃LR。真正實作前用短安全probe核對量級。

成功條件：overall≥+0.001、八項≥-0.001、AP_S/sports-ball/baseball-bat不退化，新增推論成本在事前預算內；失敗就保留CTRL。只改p3_det不代表模型輸出完全跨尺度獨立：最後top-k競爭仍可能改最終detections，所以raw-feature隔離測試與完整AP都需要。

## S6：同一新parent的BinaryQK site定位

先把P-MASF與FP metrics凍結，再於同一parent建立：

| ID | site10 | site22 | scale |
|---|---|---|---|
| Q-FP | FP | FP | FP reference |
| Q-BOTH | Binary | Binary | per-head/basis fixed PoT |
| ISO-10-BIN | Binary | FP | 相同校準程序 |
| ISO-22-BIN | FP | Binary | 相同校準程序 |

這是0個training jobs、最多4次完整evaluation，不是GPU-free。FP metrics若已是相同parent/evaluator可重用。舊V1-BR兩個isolation只用来诊断，不取代新图这些 comparisons。

歷史代號僅供找文件，不是可重用的結果 alias：

| 本輪代號 | 舊文件的相近語意 | 不可混用之處 |
|---|---|---|
| Q-FP | same-lineage FP reference | 須先確認真正 FP-QK，不採 Full35 Float 名稱推定 |
| Q-BOTH | B0／FIXED-P2-BOTH | B0 的 V1-BR 與本輪 P-MASF 不是同 parent |
| ISO-10-BIN | B1／ISO-10-BIN | 同名 site 不代表相同圖或權重 |
| ISO-22-BIN | B2／ISO-22-BIN | 同上 |
| FP-CTRL | B5／FP-CTRL | 本輪為 joint scope 與 P-MASF，不直接沿用單任務配方 |
| BQK-CAND | B6／BQK-CAND | 本輪 policy 與 QAT 配方重新鎖定 |
| BQK-KD | B7／BQK-CAND-KD | 單一 KD 的新對照；不得把不同 teacher/parent 結果合併 |

係數使用train/calibration View離線觀測，不用formal val估scale再用同一val選winner。FP/binary兩側保留相同normalization/bias/V路徑；basis/site之外state一致。每site記score誤差、top-k/ranking、KL/entropy、sign ratio、STE saturation與下游drift。

優先選精度可接受且binary coverage有用的一個policy。若hybrid沒有比both好，但both尚有合理精度／成本，就保留both進一次QAT；「兩個hybrid失敗」不能直接推出「所有BinaryQK都無效」。只有候選皆無價值或工程不通過才停止。

## S7：QAT先確認真的能更新Q/K，再看KD／scale

### 先固定training scope

現行stage_policy.py的hardware_frozen會優先封住qkv.q、qkv.k和gamma，即使tune_attention=true亦然。S7需列出參數name、requires_grad、optimizer-membership、gradient finite與實際update；只印trainable总數不夠。

QAT候選設計允許兩site的浮點Q/K projection參數經STE適應；部署再fold/量化成固定參數，不新增per-image scale。這是獨立QAT scope，不改S2/S3的production stage規則。fixed coefficient槽保持硬體表示；Q/K權重若更新，係數按事前固定節奏用train-only batches重校準並在評估前凍結。不能讓gamma名義trainable卻因fixed coefficient路徑不讀gamma而永遠沒有有效梯度。

若deployment契約要求Q/K投影數值不可變，則不啟動開放scope，清楚標為downstream-only recovery，並把Q/K可塑性列為未處理原因。

### 最小兩臂

| Arm | QK | 起點 | 更新scope |
|---|---|---|---|
| FP-CTRL | FP | P-MASF | 與candidate相同的joint參數白名單 |
| BQK-CAND | S6唯一policy | 同一P-MASF | 同scope＋必要STE，沒有teacher |

首版joint候選採J3量級role LR：backbone3.8e-6、neck1.9e-5、heads5e-5、MASF若存在3.8e-5、attention包含Q/K為5e-7；固定最多20 epochs、warmup1。這是新joint QAT提案，沒有把W-DIR的5e-5全模型LR當已驗證值。先由FP-CTRL取得共同更新／LR trace再replay，兩者同起點fresh optimizer或同名state migration，規則必须一致。

令D為同一evaluator的COCO overall AP：

~~~text
gap_before = D(Q-FP) - D(選中site policy的zero-train版本)
gap_after  = D(FP-CTRL) - D(BQK-CAND)
closure    = gap_before - gap_after
~~~

- gap_after≤0.001 且八項相對FP-CTRL各不低超過0.001：accuracy候選。
- gap_after>0.001但closure≥0.001，且protected metrics不惡化：有恢復訊號，依診斷决定下一步。
- 无closure且无可解释的ranking／gradient问题：停止，不只增加epochs。
- 舊YOLO26的0.010540或0.011不是新joint本輪gap_before；不能跨lineage計closure。

### 只有一個KD arm

若closure有訊號但ranking／KL仍差，新增BQK-KD，從同一P-MASF出發、同scope/trace。teacher固定為**訓練前P-MASF**，不用已fine-tune的FP-CTRL結果模糊teacher血緣。ranking差就用ranking loss，probability差就用temperature-aware KL，二擇一。KD需比BQK-CAND至少+0.001 AP且不傷保護指標；未過保留no-KD，不堆第二套loss。

若Q/K仍被凍結／STE梯度飽和，先解決scope或window；teacher只能提供梯度，不能讓被冻结參數更新。

### Scale分支最後且最多一個候選

只有magnitude殘差仍明顯才開C0/B4/A8 cached replay：[原scale計畫](<../binaryqk-scale-codebook/plan.md>)。

- B4為4組固定channel-scale；無每圖reduction，但partial popcount/shift可能變多。
- A8為每token從8個PoT值選擇，仍需每圖算magnitude和index；在禁止動態scale的deployment只當fidelity上界。
- 原兩site、4heads、2bases下C0固定槽16、B4固定槽64；3-bit A8 index數據不能冒稱免運算。
- 相同cached replay先篩，實際target kernel成本過才帶1個完整validation；過+0.001才新增C0-QAT/CAND-QAT兩臂。
- 現在GPU禁止，所有裝置kernel gate留待後續；Python CPU時間不當GPU或FPGA證據。

## S8：融合、固定與最終驗證

1. 保留未fuse的full-resume來源；另產生deploy copy，移除HOG／teacher。
2. RepConv fuse成單3×3，做数值parity；FP等價不保證INT8等價。
3. 用固定train-only calibration規則產生最終PoT係數；若fuse改變Q/K過零符號，要回查誤差而非假定等價。
4. 在最終圖做INT8 catalog／calibration；BinaryQK與P/V、Conv/Linear量化邊界分開。
5. 完整八項、canonical person附欄、actual binary coverage、packing＋全部epilogue後的target延遲／記憶體／功耗（有設備才報）。
6. 初篩通過的最終control/candidate補paired seeds1/2；各seed從對應祖先重新產生parent，或清楚標為僅測recovery seed，不能兩者混稱端到端3seeds。

每層增益不能直接相加宣稱整套收益。逐步選winner的最終比較只證明整套recipe；若RepConv＋MASF＋BinaryQK出現交互抵銷，按歷史證據回退一個因子再比較，不預設追加完整factorial矩陣。

## 本次完成與後续操作分工

主代理：已完成三類範圍、順序、parent/gate修正與新方向排序。Luna max：依固定規格摘錄舊方案、同步入口、檢查文件；不能改hypothesis、選winner、改dataset或啟動GPU。

之後要交給執行代理的工作單必須寫明：精確檔案範圍、parent digest、arm、配置、預期输出、驗證方式、停止條件與可用設備。欠缺parent／scope／epoch／calibration契約，保持proposed而不是ready。

返回[總計畫](<README.md>)。
