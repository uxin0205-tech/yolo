# YOLO 整合研究計畫：完整 Softmax 硬體友善化與 qSiLU RepConv

本次修訂日期：**2026-11-17**；初次保存 2026-09-22，沿用原檔名。第 1～9 階段已完成本輪研究與交付，最後 768×512 於 2026-11-13 00:07 完成，特徵圖研究累計 **120E**。本研究已停止，不自動啟動額外階段。

正式模型為使用者於 2026-11-08 指定的 **qSiLU 訓練版 Rep17＋20 E1**，部署 qSiLU／D1／B2／S1，原 batch rect640 letterbox 保持。Stage9 六組各 10E＋半 LR 10E，未有固定尺寸通過全部採用門檻。下方詳細計畫是研究規格；已完成實際範圍、差異與負結果以狀態表和正式結果為準。

**[全輪研究總報告](yolo-research-summary-2026-11-17.md)** · **[可直接使用的交接](../../result/README.md)** · [原 100E](../../yolo_actshape/result/stage9/README.md) · [768×512 的 20E](../../yolo_actshape/result/stage9-portrait/README.md) · [本次整理紀錄](../worklogs/2026-11-17-integrated-research-handoff.md)

歷次授權與方法修訂保存在 [中文工作紀錄索引](../worklogs/README.md)；原任務書保留於 [yolo_cleanup.md](../../yolo_cleanup.md)。`yolo_optimize` 唯讀、BBAT5 v1 不變、每次研究事前核對 optimize 的規則持續適用。訓練與部署算子分開，最終驗收使用部署 AP。

## 階段狀態

| 階段 | 範圍 | 狀態 |
| --- | --- | --- |
| 1 | Binary branches：B1～B3 | 已完成並交接至 result/stage1：選 B2，官方 AP 0.509889；恢復 5 epochs 無改善 |
| 2 | 新版 scale／bias：S1～S2 | 已完成並停止：選 S1 Q10，官方 AP 0.514152；兩組各五輪、交付驗證通過 |
| 3 | Attention 與完整 PWL Softmax | 已完成：D1＋KD 第五輪 AP 0.514621，使用者覆核採用並交接 |
| 3A | normalization 比較：N1～N9 | 已完成：N1～N6 正式 AP，N7～N9 診斷排除 |
| 3B | PWL exp＋reciprocal：D1～D3 | 三組整數 reference 與完整 AP 通過 |
| 3C | 整合選型與必要恢復訓練 | 採用 D1＋KD 第五輪；保留原自動門檻失敗紀錄與使用者覆核理由 |
| 4 | Detect／Pose MASF：M1～M2 | 已完成並停止：含延長 L1／L2，兩任務均建議保留原 parent，候選可覆核 |
| 5 | HOG：H1～H4 | 已完成並停止：H1 通過，H2 一輪後觸發精度門檻，H3 全量重載一致，保留原 parent；H4 僅提出 coverage 研究方向 |
| 6 | Detect–Pose combine：C1～C8 | 已交付本輪結果並停止：KD E2 → Pose head E1，官方 COCO 0.512676、BBAT box／pose 0.609202／0.899850；原四項精度目標未達，不自動升格 |
| 7 | Activation 比較與 qSiLU 主線建立：V1～V6 | 已完成：40 輪、完整 AP 與交付驗證；PolyQuality 比較最佳，qSiLU E8 作 RepConv parent |
| 8 | RepConv 訓練與硬體部署、SiLU／ACB 比較 | 兩輪完成；使用者選 qSiLU 訓練版 Rep17＋20 E1，部署 D1；見 result/stage8-retrain |
| 9 | 全模型硬體檢查、特徵圖形狀分析與最終成果 | 2026-11-13 完成累計 120E 並停止，保留原 qSiLU Rep17＋20／rect640 |
| 9A | 完整部署圖與剩餘運算檢查 | CPU 算子查核與 GPU 任務驗證完成；剩餘浮點已列明，未做整網整數量化或 RTL／板端驗證 |
| 9B | 長方形／二次冪特徵圖的可行性與效益；工作區 `yolo_actshape/` | 六組各 10E＋半 LR 10E、完整 AP 與合計 336 項 GPU replay 完成；固定輸入下的內部 stride 變體尚未實作 |
| 9C | 最終模型、報告與清理交接 | result/stage9 與 stage9-portrait 完成；根層 result 統一交接。歷次及本次清理分開記錄 |

狀態表只記錄本輪研究，不代表既有研究、權重或結果不存在。實際 parent、checkpoint SHA、預算與採用門檻在指定階段啟動前固定，不以規劃內容冒充已執行結果。

## 一、核心原則與執行順序

**RepConv 的訓練與部署算子分開定義。** 訓練遵循原本 RepConv 多分支流程，可使用常規算子，不強制使用已簡化的 qSiLU／PWL／reciprocal。每個候選的最終部署必須符合已選硬體友善契約，包括 qSiLU、D1 完整 PWL exp＋LUT256 reciprocal、Binary QK、Q10 scale／bias；最終依部署版本的精度與成本驗收。

Activation 研究與已完成 qSiLU parent 保留為參考。未來訓練入口的 activation／normalization 可與部署不同，需明確記錄共同 parent、實際訓練設定與轉換步驟，不得把訓練版本的 AP 當作部署版結果。此調整解除舊有「訓練前向也固定 qSiLU／D1」的要求；實際已完成的授權與結果見狀態表；本計畫不授權啟動額外研究。

完整順序：

```text
既有 Detect 模型
  → Binary branches
  → 新版 scale／bias：Q10／二次冪
  → normalization 方法比較
  → PWL exp＋reciprocal 硬體友善化
  → Detect／Pose MASF
  → HOG
  → Detect–Pose combine
  → Activation 比較與敏感度
  → 確立並驗證 qSiLU 主線模型
  → Rep17／Rep20／Rep17＋20，全部使用 qSiLU 與同一硬體友善 Attention／reciprocal
  → 全模型硬體檢查、長方形／二次冪特徵圖分析
  → 最終模型、精度、成本、清理與報告
```

研究採由上而下繼承：上一階段選定模型，下一階段延續其權重與設定；同階段候選則從共同 parent 分別出發。

固定邊界：

- `yolo_optimize` 完全唯讀，作為主要方法、設定與結果參考。
- 不新增未改方法的 baseline／control 加訓。
- 不新增 sign-only、scaled-sign、shared MASF 或 ReLU 候選。
- Pose 使用 `yolo_pose_training` 已訓權重，在 MASF、Combine 階段接入。
- Attention 能力集中於 `yolo_attention`。
- BBAT5 v1 不重切、不抽換來源、不改標註。
- 每階段由使用者指定開始；完成後先交付結果並停止，等待使用者指定下一階段。

### 跨階段硬體友善要求（2026-10-17 新增）

每個階段都檢查實際前向與部署運算，涵蓋模組、functional 呼叫及匯出後的算子；先區分訓練專用、離線轉換、模型推論與前後處理，再判斷哪些成本會在部署時發生。硬體友善是逐項可驗證的要求，不以 qSiLU、PWL 或 BitTrue 名稱代替整張運算圖的檢查。

| 階段 | 本階段硬體檢查重點 |
| --- | --- |
| 1. Binary branches | sign／XNOR／popcount、Hadamard 變換軸與長度、residual 的動態幅度／除法、分支共用及 score 累加；共同 parent 的 normalization 保持一致。 |
| 2. scale／bias | Q10 常數乘法、二次冪 shift、signed bias、累加位寬與捨入；固定係數與每個輸入重新計算的係數分開計成本。 |
| 3. Attention | row-max、exp、權重累加、動態分母 reciprocal、probability 與 P·V；查核是否隱含回退除法或其他浮點運算。 |
| 4. MASF | Conv／BN 的部署折疊、activation、alpha 常數乘加、實際存在的 pooling／resize；bridge beta 僅屬梯度行為。 |
| 5. HOG | 訓練支路、target 與 loss 的運算獨立記錄；部署圖確認 HOG head 與其計算已移除。 |
| 6. Combine | 共享 trunk、兩個 head、routing、decode 與座標還原；兩個任務都繼承選定硬體契約，初始化插值與訓練 task weights 分開記錄。 |
| 7. Activation | qSiLU 固定係數的乘加／shift、實際 call-sites、alias、MASF 與 head 路徑；核對其他非線性與剩餘除法。 |
| 8. RepConv | 訓練算子可用常規實作；R1～R3 的部署 qSiLU／PWL／reciprocal 必須有效。分開驗證相同算子下 folding 等價與部署優化後精度。 |
| 9. 最終整合 | 對完整模型與前後處理再檢查一輪，連同形狀、padding、記憶體、資料搬移及剩餘運算一起評估。 |

除法與其他可能昂貴運算依下列方式處理：

- **固定二次冪分母**：評估以 shift 實作；先固定有號值、fraction bits、捨入與飽和，不能直接把 Python `/` 改成 `>>` 就當作等價。
- **其他固定分母**：評估離線計算 reciprocal、固定係數乘法或可證明等價的常數除法；常數折疊與執行時成本分開記錄。不可任意把分母改成最近二次冪。
- **動態分母**：分析範圍與更新頻率，優先重用第 3 階段驗證的 reciprocal 方法；Softmax 每列計算一次，其他場合依其實際定義計算，零分母、溢位與量化誤差都要明確處理。
- **其他非線性與隱含運算**：檢查實際存在的 exp、sigmoid、sqrt／rsqrt、mean／pooling normalization、resize 與座標除法。可評估 LUT、PWL、固定係數、乘加／shift 或離線 folding；BN folding 的離線平方根不能誤算成每次推論成本。

每階段交付一份運算清單，記錄 module／call-site、公式、常數或動態輸入、執行 backend、資料型別、位寬、替換方式、誤差／精度與成本證據。可行的替換在該階段完成必要實作與驗證；尚不適合者保留原因與下一步，不為湊齊硬體友善標籤而降低精度或隱藏剩餘成本。

新增替換須先固定比較方式與預算，再以單元誤差、真實資料診斷及必要任務評估確認，避免一次改動多項後無法判斷原因。已選設定逐階段繼承；若要修改上游契約，記錄變更及重新驗證受影響部分，不自動重跑全部研究矩陣。

硬體友善與整網全整數化分開描述：QKV、P·V、projection、Conv 或其他路徑若仍為浮點，逐項明示。此次補充不自動啟動 `yolo_quantize`、整網 PTQ／QAT 或硬體綜合，也不在未量測時宣稱實測加速。

## 二、先保存計畫，再按階段清理重建

2026-09-22 的初次工作只保存本計畫：

- 正式文件沿用 `docs/research/yolo-research-plan-2026-09-21.md`，記錄本次修訂日期。
- 根層 README 加入入口。
- `yolo_cleanup.md` 標明以整合計畫接續。
- 更新中文工作紀錄與索引。
- 所有研究階段先標記「待指定」。

計畫已保存；第 1～7 階段已完成本輪交付。第六階段經 J1 E6 診斷、雙教師 KD 與 Pose head 恢復，選定候選及未達目標已記錄於 result/stage6；第七階段已完成並交付；第八階段兩輪均已完成交付，最新五組 Rep／ACB 比較已完成，使用者選定 rep17_20 E1（qSiLU 訓練，部署 qSiLU／D1），Stage9 已於 2026-11-08 獲授權，在 yolo_actshape 接續。

### 開始工作時的切入點

優先完成啟動前盤點，再從第 1 階段 `yolo_binaryqk` 的 B1～B3 開始準備；若使用者另指定階段，則以該階段為範圍。

1. 核對本機工作狀態、實際 Detect parent、checkpoint SHA、模型結構、既有結果及資料／evaluator 設定。區分本機未提交研究內容與 GitHub `Cleanup-v1` 發行快照，不以已清空的遠端 main 覆蓋本機。
2. 唯讀參照 `yolo_optimize`，抽取 Binary branches 所需方法、設定、結果與依賴；列明可沿用內容、待修正差異與受保護資產。
3. 固定 B1／B2／B3 的共同 parent、normalization、比較口徑與交付項目，先整理必要實作及 CPU 驗證，再依指定階段與預算執行正式實驗。COCO 球棒／球指標一併納入該階段評估設計，不新增未改方法的 baseline 加訓。

以上為 2026-10-03 保存的接續方式；2026-10-17～18 使用者已明確授權開始第 1 階段。

### 每項工作前的 optimize 核對（2026-10-18 明確要求）

所有階段及階段內每項實作、方法修改、候選實驗與訓練，均須在動手前唯讀參考 `yolo_optimize` 的對應方法；不得等精度不如預期才回頭查。來源包括實際程式、run 設定、parent／checkpoint 血緣與完成結果；若 optimize 使用上游研究，須追到該階段的原始配方，不能把後期微調當成初次恢復。

每項工作的 protocol／實驗紀錄須先寫清楚：**參考來源與結果 → 當前 parent 的適用性 → 採用內容 → 差異與理由 → 驗證、預算與停止條件**。核對訓練 scope、LR、BN／EMA、loss scaling、batch／累積、dtype、scale／bias、normalization、資料及 evaluator；已有稽核可引用，但新候選或配方必須再次確認適用性。

關鍵來源或差異未釐清時先查核、診斷，不猜配方直接長訓；無對應來源的新方法明記為專案延伸。啟動 GPU 前向使用者說明參考來源、本輪差異與資源預算，沿用既有授權，不新增反覆確認流程。此要求適用於目前第 1 階段的修正版恢復，以及後續所有階段；`yolo_optimize` 保持唯讀，使用者指定的候選與硬體契約仍優先。見 [規則紀錄](../worklogs/2026-10-18-optimize-reference-before-work.md)。

指定階段開始後，依序：

**確認 parent → 參照 optimize → 抽取必要舊資訊 → 核對依賴 → 清除被取代內容 → 重建 → 驗證與實驗 → 分析交接。**

每個資料夾第一次執行時處理舊內容；後續候選保留本輪新實作與結果，不反覆清空。

### 跨階段 result 交接（2026-10-19 補充）

依使用者最新要求，切換階段時將上一階段選定的成果資料夾整理為該專案的 `result/stageN/`，包含選定權重、manifest、部署常數、指標、設定、parent／SHA 與必要續訓狀態。下一階段讀取這份成果，在自己的目標資料夾實作及保存新結果，不要求一直在原先的專案或 run 內工作；共用 Attention 算子仍集中維護。

搬移前確認工作完成、active process 及依賴；搬移後核對 SHA、載入與契約。歷史路徑如仍被報告引用，可保留指向同一實體的相容符號連結，不另複製權重。尚未完成的 run 不搬動。正式候選比較與負結果仍可追溯，錯誤／被取代且不再需要的內容直接刪除，不封存。

第 1 階段 B2 已搬到 `yolo_binaryqk/result/stage1/`，第 2 階段直接從此處接續；第 2 階段 S1 已搬至 `result/stage2/`，第 3 階段由 `yolo_attention` 接手。階段停止與使用者指定下一階段的規則不變。

不永久保存整套 legacy、archive、backup 或另一套 final。必要唯一權重、active resume、正式資料、尚未抽取的唯一證據，以及 optimize 仍使用的外部依賴，須先完成替代才能移除。

未指定的資料夾不提前清理。

### 無用資料夾的實際移除（2026-10-17 補充）

使用者已要求刪除無用資料夾。指定階段開始後，對該範圍內已確認無用、可重建或已完成替代的內容，先留存必要小型證據並核對依賴，再實際移除，不只提出建議或搬進另一份 archive。2026-10-17～18 第 1 階段已執行經驗證的清理，詳見該階段紀錄。

每項移除記錄原路徑、用途、移除理由、替代位置、必要 hash／大小、依賴檢查及驗證結果；處理後記錄刪除清單與實際空間變化。只有在新入口、config、checkpoint loader、active run 及 optimize 外部依賴都不再需要時，才移除整個資料夾。

`yolo_attention_final` 在必要能力遷入 `yolo_attention` 並完成驗證後移除；`yolo_p2` 在相關依賴與必要證據處理完成後移除。被取代的重複 final、失敗候選的大型產物與暫存目錄依相同條件處理；不能只憑名稱或「不在主線」判定無用。需要新版本解除依賴的內容，先完成替代與驗證，再執行刪除。

`yolo_optimize` 保持唯讀；BBAT5 canonical 資料、原始 lineage、受保護的已訓 Pose、必要唯一權重、active resume 及尚未抽取的唯一證據保留。`yolo_quantize` 未指定前不清理或重建。用途仍不明或仍有依賴的項目列出具體原因與待解事項，不猜測刪除。第 9 階段再核對本輪各階段留下的過期內容，完成整體交接。

## 三、逐階段實驗與交付

### 第 1 階段：Binary branches

**資料夾：`yolo_binaryqk`**

| 編號 | 候選 |
| --- | --- |
| B1 | 原始 sign＋residual |
| B2 | 原始 sign＋Hadamard |
| B3 | 原始 sign＋residual＋Hadamard |

B3 共用原始 sign 分支，三項 score 合併後只做一次 normalization 與 P·V。

兩個 sites、每處四個 heads 下：

- B1／B2：16 個 scale slots。
- B3：24 個 scale slots。

固定共同 parent 的 normalization，參照 optimize 的精確二值前向與梯度修正。核對 Q／K 實際更新及 residual 形成成本，不因輸出尺度固定就忽略內部動態計算。

**交付：**branch winner、比較結果、未採用原因及下一階段 parent。

### 第 2 階段：新版 scale／bias

**資料夾：`yolo_binaryqk`；共用算子由 `yolo_attention` 維護**

| 編號 | 候選 |
| --- | --- |
| S1 | 可學習固定 Q10：`m/1024`，`m=1…1024` |
| S2 | 可學習固定二次冪：`2^k`，`k=−10…0` |

兩組從相同 branch winner 獨立開始，部署時係數固定。

bias 沿用 optimize 新版分解式相對位置表：

```text
bias(i,j) = by[yi−yj] + bx[xi−xj]
```

每個 site／head 各有水平、垂直表，訓練可更新，推論固定；使用 signed16／1024。增加 basis 不增加 bias 表數量。

兩組使用相同初始化來源、bias 政策、訓練預算、資料與 evaluator。比較精度與硬體運算代價，不預設二次冪必勝。

不展開其他尺度訓練矩陣；必要 bias 開關僅作診斷。

**交付：**選定尺度、固定常數與 bias、下一階段 parent。

### 第 3 階段：Attention 與完整 PWL Softmax

**資料夾：`yolo_attention`**

整合 `yolo_attention_final` 的必要演算法、載入、訓練、驗證與交付能力，只維護一套 Attention。

#### 3A：normalization 比較

固定 S 階段 winner，先保持共同的分母計算方式。

| 編號 | 候選 |
| --- | --- |
| N1 | Float PWL `[-8,0]`／16 段 |
| N2 | Float PWL `[-10,0]`／20 段 |
| N3 | BitTrue PWL `[-10,0]`／20 段 |
| N4 | Score-indexed LUT |
| N5 | Power-of-two／SHIFT |
| N6 | Normalized HardSigmoid |
| N7 | Multimax Top-1 |
| N8 | Multimax Top-3 |
| N9 | Multimax Top-5 |

先做數值與替換診斷，再評估值得保留的候選；必要時只安排少數候選恢復訓練。

取消 ReLU、exact Softmax 正式對照組與 `[-8,0]` BitTrue 組。Exact Softmax 僅作數值參考。Integer LUT、BDCN 整理既有方法與結果，不重啟完整訓練矩陣。

#### 3B：PWL exp＋reciprocal

參考指定論文的倒數 LUT 與乘法 normalization；該論文仍保留共用 divider，其 Newton–Raphson 主要用於 LayerNorm 平方根。下列 reciprocal PWL 與 reciprocal Newton 是本專案延伸。[論文原文](https://www.mdpi.com/2079-9292/14/12/2337)

固定 PWL exp，只比較分母處理：

```text
score
  → row-max subtraction
  → PWL exp
  → 整數權重累加
  → 每列一次 reciprocal
  → 權重縮放
  → probability
  → 原有 P·V
```

| 編號 | 候選 | 初始規格 |
| --- | --- | --- |
| D1 | Reciprocal LUT | 分母 leading-one 正規化，256-entry 尾數表 |
| D2 | Reciprocal PWL | `[1,2)` 分成 16 段近似 `1/m` |
| D3 | 小 LUT＋一次 Newton | 64-entry 初值，加一次修正 |

```text
D = m × 2^e，1 ≤ m < 2
1/D ≈ approximate_reciprocal(m) × 2^(-e)

Newton：
r1 = r0 × (2 − m × r0)
```

初始 reciprocal 格式採可表示 1 的 unsigned Q1.16；中間乘積保留完整精度，再於明確邊界捨入。D2 斜率使用有號格式。

重用本地 BDCN 的相關公式與測試，但將浮點 `frexp`／`ldexp` 參考轉成明確整數行為。若誤差不合格，先處理近似、位寬與捨入，不立即用訓練掩蓋問題。

不把分母直接粗略取成最近二次冪作主候選。S2 的 score scale 與分母近似分開研究。

#### 3C：整合選型

依序完成：

1. 定點單元與輸出誤差。
2. 兩個 Attention sites 的真實資料診斷。
3. 有價值候選的完整任務評估。
4. 精度／成本選型。
5. 必要時少數候選恢復訓練。

既有浮點 reciprocal 作診斷參考，不新增 control 加訓。恢復訓練前向使用候選部署路徑，反向代理另行記錄。

PWL 支線保留完整結果；其他 normalization 若更適合主線，與完整 PWL 候選比較後再提出選擇，不預設 winner。

交給後續階段的方案必須說明完整部署運算與誤差契約；PWL exp 與 reciprocal 是同一條需要驗證的路徑。Float PWL 可作數值與任務比較，但只有浮點參考或仍靠最後一次一般除法正規化時，不能宣稱已完成完整 Softmax 硬體友善化。替代 normalization 同樣要交付分母處理與剩餘運算，不能因跳過 exp 就忽略其成本。

**交付：**選定 normalization、reciprocal、定點契約、常數、部署運算清單與下一階段 parent。

### 第 4 階段：MASF

**資料夾：`yolo_masf`**

| 編號 | 實驗 | 起點 |
| --- | --- | --- |
| M1 | Detect task-specific P3 MASF | 第 3 階段 Detect |
| M2 | Pose task-specific P3 MASF | 已訓 Pose |

不做 shared MASF，不增加無 MASF control 加訓。

驗證 alpha-on／off、bridge 梯度、raw P3 與後續尺度的資料流及成本。alpha 是 residual 強度，bridge beta 是梯度縮放，不混用。

**交付：**Detect、Pose 分別採用或不採用 MASF 的建議及權重。

### 第 5 階段：HOG

**資料夾：新增 `yolo_hog`**

| 編號 | 工作／實驗 |
| --- | --- |
| H1 | coverage 與梯度診斷 |
| H2 | optimize HOG 方法候選適應 |
| H3 | 移除 HOG head 後推論等價 |
| H4 | 僅在診斷支持時提出新 coverage 候選 |

接續 Detect，不增加 control 加訓，不自動擴展到 Pose。HOG 只用於訓練，無收益就維持 parent。

### 第 6 階段：Detect–Pose combine

2026-10-30 結果：已完成 H0／H1／P、三個 ρ、M0／J1、分組診斷、雙教師 KD 及 Pose head 恢復，總計 118 個不同段落的 epoch。J2／J3 原條件流程未直接執行，改以小 LR KD 處理後段共享權重，再固定共享層恢復 Pose。選定 E1 已完整重載驗證並完成 optimize 同設定比較；原獨立模型四項目標仍未達。交付 [第六階段正式報告](../../yolo_combine/result/stage6/README.md) 後停止，後續階段待指定。

2026-10-29 使用者最新要求：以找出最佳實際融合結果為目標；optimize 是參考。代理依診斷自主調整階段內配方，不因單次波動或參考停止條件直接結束整個研究。每段修改仍須事前參考、固定預算、保存血緣與完整比較；真實問題或已無可行改善時說明，整階段完成後停止。見 [最新研究決策](../worklogs/2026-10-29-stage6-best-fusion-objective.md)。

2026-10-29 重建規則：先從 Stage5 Detect 接上既有 Pose head，完成 H0／必要 H1 及完整 Pose 分支 P 適應，再比較 ρ=0／0.1／0.25，選一份接 M0／J1／J2／J3。每段 fresh AdamW、fresh EMA age0、warmup 1 輪與 cosine；J1 條件總上限 60、J2 條件總上限 80、J3 上限 20，不要求跑滿。所有上限、校準及停止條件見 [重建 protocol](../../yolo_combine/artifacts/stage6/reference-review.json)。

**資料夾：`yolo_combine`**

使用上游 Detect，以及已訓 Pose 或選定的 Pose MASF 衍生模型，不重跑原始 Pose 訓練。

| 編號 | 工作／實驗 |
| --- | --- |
| C1 | 架構、參數、head 相容與映射 |
| C2 | `ρ=0、0.1、0.25` trunk 初始化比較 |
| C3 | task weights／梯度比例診斷 |
| C4 | 必要時 J0 head 適應 |
| C5 | J1 neck 適應 |
| C6 | 視需要接續 J2 後段 backbone |
| C7 | 視需要接續 J3 小 LR refinement |
| C8 | routing、MASF 開關與部署檢查 |

初始化比較後選一個接續，不把三個比例各跑完整流程。

共享模型必須繼承已選 Attention／reciprocal 契約，不得在重建時默默回退浮點除法。

**交付：**融合模型、八項 AP、COCO 球棒／球分類別 Detect 指標、獨立模型參考及共享成本。

### 第 7 階段：Activation 比較與 qSiLU 主線建立

2026-11-03：四候選各 10 輪與重載完整 AP 已完成；可攜模型、圖表與清理見 [Stage7 結果](../../yolo_activation/result/stage7/README.md)。比較最佳為 PolyQuality，RepConv parent 固定 qSiLU E8。下方 10-31 為歷史準備狀態。

歷史 2026-10-31：依使用者要求整理並準備接續，已完成 [Stage7 CPU 前置核對](../../yolo_activation/stage7/README.md)，不複製權重。正式候選替換、數值診斷、AP 與恢復訓練尚未執行；目前模型仍為 SiLU，沒有 RepConv。

**資料夾：`yolo_activation`**

| 編號 | 工作／實驗 |
| --- | --- |
| V1 | qSiLU 等函數的公式、分段、導數與尾端驗證 |
| V2 | qSiLU、Hardswish、PolyShift 替換診斷 |
| V3 | region／site sensitivity |
| V4 | 有依據的候選恢復訓練 |
| V5 | 必要時比較 selective policy |
| V6 | 保存已驗證 qSiLU 模型作 Rep 部署參考；新訓練算子另行定義 |

不新增 ReLU，不原樣重訓既有 activation 作 control。

**本階段分開產出「Activation 比較結論」與「RepConv 接續模型」。**

- 其他 activation 即使部分指標較好，也保留為比較結果，不自動改變 RepConv 最終部署使用 qSiLU 的要求。
- 已完成 RepConv 從 qSiLU 主線繼承；未來訓練可顯式採用常規算子，parent 與轉換方式在新 protocol 固定。
- qSiLU 沿用 optimize 的 `ShiftPiecewiseQuadraticSiLU`／`qsilu_pq` 契約，包含固定分段係數。
- 替換範圍、實際 call-sites 與共享 module alias 寫入 manifest。
- 若 parent 已符合 qSiLU 契約且結果可用，直接沿用，不重複訓練。
- 若需要新的訓練入口轉換或適應，在新方案中明列並驗證，再固定三候選共同 parent。
- V6 同時驗證 qSiLU 與已選硬體友善 Attention／PWL／reciprocal 等完整契約，列出剩餘浮點與其他未替換運算。不能交付只有 qSiLU 正確、其他模組已退回預設路徑的 parent。

Activation 改變後，重新檢查 Attention score 範圍與既定 PWL／reciprocal 契約，不自動重啟全部 normalization 搜索。

**交付：**比較報告、敏感度圖、已驗證的 qSiLU parent。

### 第 8 階段：RepConv 訓練與硬體友善部署

**資料夾：`yolo_achitechure`**

2026-11-04 使用者明確確認：訓練時的 RepConv 不必使用原先要求的簡化算子，優化要求作用於部署／推論。依此分開設計訓練與部署，不再要求訓練圖也固定 qSiLU／D1。11 月 7 日先核對 optimize 並修正草案，其後使用者授權重訓與少量變體比較。五組各 5E、SiLU 另 3E 硬體適應均於 11 月 8 日完成，推薦適應前 rep17_20_silu E1；最終部署仍為 qSiLU／D1。

已完成第一輪三組各 5 輪，仍保留於 [Stage8 結果](../../yolo_achitechure/result/stage8/README.md)。該輪實際使用 qSiLU E8 parent，SHA `2f91ec1862ad62304f1e2727e9b9ee64a27c5a46a45137f83a74dc7f14c46a69`，訓練前向也使用 qSiLU／D1；這是歷史實驗事實，不再當成未來訓練限制。

| 編號 | 候選 | 訓練設定 | 最終部署要求 |
| --- | --- | --- | --- |
| R1 | Rep17 | 原本 RepConv 多分支流程，允許常規算子 | 單 Conv3×3＋qSiLU，完整 D1 與既定硬體契約 |
| R2 | Rep20 | 與 R1 相同訓練政策及共同 parent | 同上 |
| R3 | Rep17＋20 | 與 R1 相同訓練政策及共同 parent | 同上 |
| R4 | Rep17＋20 SiLU | 使用者後續授權；僅兩個 Rep 相加後 SiLU，Float PWL | 同上；本輪選定 E1 |
| R5 | ACB17＋20 | 使用者後續授權；3×3＋1×3＋3×1，qSiLU／Float PWL | 同上；球 Pose 門檻未通過 |

共同 parent、實際訓練 activation／normalization、BN 政策、可訓範圍、LR／預算及恢復策略在新方案明列，不能因釐清部署要求就假定其他配方已經獲准。三組採一致政策，只比較 Rep placement；不新增原樣 baseline／control 加訓。

```text
訓練：
x → Conv3×3＋BN ─┐
                 ├→ 相加 → 訓練用 activation
x → Conv1×1＋BN ─┘

折疊等價檢查：
x → Folded Conv3×3 → 同一訓練用 activation

最終部署優化：
x → Folded Conv3×3 → qSiLU
Attention → 已選 D1 PWL exp＋LUT256 reciprocal
```

activation 放在分支相加之後；Rep folding 只處理線性分支，不把 activation 折進 Conv。訓練與部署使用不同算子時，分開驗證 folding 等價與算子替換誤差，最後以部署版本完整 AP 選型。

驗證與交付：

- 原本 RepConv 實作、初始化、多分支梯度、BN 設定及已訓非零分支 folding。
- 同一 activation 下的 folding 前後，在 FP32 與實際 dtype 記錄線性及完整輸出誤差。
- 部署 qSiLU 類別、固定係數、位置、call-sites 及 alias 正確；重載／匯出後一致。
- 部署 B2 sign＋Hadamard QK、兩處四 heads、16 個 S1 Q10 scale 與 signed16/1024 bias 正確。
- 部署 D1 保留整數 exp、reciprocal、乘法 normalization、UQ1.24 probability 與 token 範圍；不以額外浮點除法修飾 row-sum。
- 分列訓練算子版本、folded 同算子版本與最終優化部署版的指標，包含 person、COCO 球／棒及 BBAT box／pose；最終以部署版驗收。
- Manifest 明確標註 training／deployment 設定、轉換步驟與權重 SHA。多分支 resume、未優化 folded reference 與正式部署 checkpoint 不混用。
- 實際圖列出其他非線性、動態除法、BN、head／decode 及剩餘浮點；未做 RTL／HLS／板端量測時不宣稱實測加速。

**交付已完成：**[五組正式結果與可攜模型](../../yolo_achitechure/result/stage8-retrain/README.md)、[含無 Rep parent 與 PolyQuality 的比較](stage8-comparison-2026-11-08.md)，以及訓練／部署設定、轉換精度、成本與必要續訓權重。五組 folding、重載完整 AP、獨立 CPU 可攜驗證通過。已停止，Stage9 已獲後續授權。

### 第 9 階段：全模型硬體檢查、特徵圖形狀分析與最終成果

**位置：第 9B 階段使用 `yolo_actshape/`；第 9A、9C 階段於各專項結果與根層 `docs/research/` 整合。**

本階段從第 8 階段選定模型開始，重新檢查完整部署流程，包含使用者提出的長方形及二次冪特徵圖。先完成可行性與診斷，再對有依據的少數方案做正式比較；不將「重新檢查整個模型」解讀為全部階段重新訓練。

#### 9A：完整部署圖與剩餘運算

追蹤實際輸入前處理、backbone／neck、Attention、MASF、雙 head、decode、routing 與實際使用的後處理。核對 qSiLU、PWL／normalization、reciprocal、scale／bias、folded RepConv 及模型重載、匯出行為，整理每個仍存在的除法、非線性、浮點運算與資料搬移。

評估尚可安全替換的部分，先定義等價或誤差門檻，再完成必要驗證；未採用者說明原因。運算數、儲存／流量估算、軟體量測、RTL／HLS 綜合與板端量測各自標示，避免以算子少了或尺寸整齊代替實測效益。

#### 9B：長方形／二次冪特徵圖

**專屬工作區：[`yolo_actshape/`](../../yolo_actshape/README.md)。** 依使用者 2026-11-07 指定，本階段的形狀追蹤、候選設定、相容驗證、必要適應及結果交接在此進行；從第 8 階段最終選定的成果接續。2026-11-08 已獲使用者授權啟動，並要求增加輪數；固定 qSiLU 訓練版 rep17_20 E1 為共同 parent。有限尺寸 E0 完整驗證後，最多兩組各 10E＋半 LR 10E，總上限 40E；配方與選型見 `yolo_actshape/protocol.json`。

**研究問題：特徵圖為何是目前的 `H×W`？哪些位置需要方形、stride 對齊或特定分組？長方形、二次冪維度或僅調整 padding／硬體分塊，能否在精度與成本上更合適？**

一般輸入介面可以接受長方形；Ultralytics 官方 predict 文件說明 `(height, width)` 與依 stride 補邊的處理，並區分 predict／export 與 training 的尺寸介面。這只能作為通用介面參考，本專案的自訂 Attention、融合模型、trainer 與匯出 backend 仍需逐一驗證。[官方尺寸說明](https://docs.ultralytics.com/modes/predict/#fixed-shape-vs-minimum-rectangle-rect)

先由實際運算圖推導每層形狀，記錄 Conv kernel、stride、padding、dilation、下採樣與 concat 對齊。對尺寸可整除、且採對應 padding 的範例，累積 stride 為 16 時：

| 執行時輸入 `H×W` | 對應特徵圖 `H×W` | 範例用途 |
| --- | --- | --- |
| `640×640` | `40×40` | 說明 40 可來自輸入與 stride，並非必須固定為 40。 |
| `512×512` | `32×32` | 二次冪維度的方形範例。 |
| `512×1024` | `32×64` | 二次冪維度的長方形範例。 |
| `640×1024` | `40×64` | stride 對齊也可以包含非二次冪維度。 |

上表是條件式算例，不是目前兩個 Attention sites 的實測形狀，也不是已選定的新解析度。stride 的倍數、每一軸為二次冪、token 總數為二次冪，以及 channel／Hadamard 變換軸長度，必須分開判斷。

分析與確認依序進行：

1. **形狀與限制追蹤**：記錄實際輸入、P3／P4／P5、兩個 Attention sites 及雙 head 的 `H、W、C`、flatten 順序、每列實際 token 數與 window／area 分組。查找硬編碼 `40×40`、由 `sqrt(N)` 猜方形、位置索引、cache 與 reshape 限制，明確傳遞 `H、W`。
2. **有限候選**：保留原形狀作同權重推論參考，再依實際資料長寬比、stride 與目標部署限制提出少數長方形、二次冪維度候選。分開比較改變有效輸入解析度、改變 letterbox canvas、只在內部補齊 buffer／tile 三種方式，不預設所有層都必須改成二次冪。
3. **相容與數值驗證**：核對 neck concat／upsample、MASF、head grid／stride、bbox／keypoint 座標還原、相對位置 bias 表的有效範圍，以及 shape cache 與匯出支援。不得以截斷、未記錄插值或隨機補參數處理不相容 bias。padding 需要驗證邊界運算與 Attention mask；補零 token 不能直接當成對 normalization 無影響。
4. **重查 Attention 定點契約**：依每個 site 真正參與 normalization 的 token 數重算分母範圍、累加位寬、reciprocal 指數範圍與輸出誤差，重新觀察 score 範圍與 PWL 尾端。空 mask／零分母要有明確契約，形狀改變不得靜默超出已驗證範圍或回退浮點除法。
5. **精度與成本比較**：先做同一 checkpoint 的載入、前向、數值及匯出檢查，再評估值得保留的候選。以 shape／padding 為受控變因，其餘資料、evaluator、dtype 與硬體契約固定；分開標示有效影像縮放、padding 比例、物件像素大小、總 token／每列 token、運算量、記憶體、資料搬移及可量測延遲。報告八項主要 AP、COCO 球棒／球與小物件影響，不能把縮小影像帶來的成本下降全歸因於二次冪。
6. **選型與必要適應**：若有證據支持，再固定少數候選與預算進行必要適應，保留選定 RepConv parent、共同硬體契約及真實血緣。無收益則保留原形狀；形狀不變的控制模型只做推論參考，不新增 control 加訓，也不自動重跑 B～R 矩陣。

同權重形狀診斷與適應訓練後的精度分開報告；第 9B 的尺寸是刻意改變的實驗條件，不能與其他階段要求同 imgsz 的比較混用。二次冪尺寸可能有利於位址或分塊，但須檢查額外 padding、token 增長與實際 backend，不能預設速度或儲存一定較好。

所有 resize／letterbox 與 bbox／keypoint 映射只發生於執行時，完整保留同一批正式資料與 split，不修改 canonical 影像、labels、來源或 registry，不建立新的資料版本，也不以裁切或抽樣讓比較口徑改變。

**交付：**實際 shape trace、相容限制、候選設定、精度／成本比較、採用或保留原形狀的理由，以及更新後可重載的完整部署契約。

#### 9C：最終交付與清理

交付：

- 各階段採用方案與真實模型血緣。
- 八項主要 AP、COCO 球棒／球分類別 Detect 指標、參數、檔案大小及已量測成本。
- Binary、scale／bias、完整 Softmax、MASF、融合、qSiLU 與 Rep folding 圖。
- 負結果、缺少的對照、單 seed 與硬體量測限制。
- 可在無 GPU、無 checkpoint 下重畫的數據與程式。
- 全模型運算清單、剩餘浮點／除法及其理由，連同實際 shape trace、長方形／二次冪尺寸分析與選型結論。
- 本輪無用資料夾與被取代產物的實際刪除清單、替代對應、驗證結果及尚需保留的具體原因。

新研究模型不自動覆蓋 optimize 的既有正式模型。

## 四、介面契約與必要驗證

### COCO 球棒／球評估目標（2026-10-03 新增）

**狀態：待實作／待評估，目前只納入目標，不執行評估。**

後續凡正式評估 COCO80 Detect 或融合模型的 Detect 分支，除既有整體 COCO 與 person 指標外，另行報告下列兩類：

| 評估對象 | COCO 類別名稱 | 本專案 `coco2017.yaml` 的 YOLO 類別索引 |
| --- | --- | ---: |
| 球棒 | `baseball bat` | 34 |
| 球 | `sports ball` | 32 |

- 使用既有完整 COCO2017 validation split 與相同 evaluator，不另行重切或只挑含球／球棒的影像。COCO `sports ball` 按原類別口徑評估，不將其改標或當成棒球專屬資料。
- 兩類分別記錄 bbox AP50、AP50–95、Precision、Recall，以及對應影像／標註數；不只提供合併平均。Precision／Recall 的 confidence 與 IoU 口徑須固定並列於報告。
- 記錄 parent 與候選各自的 checkpoint SHA，使用相同 split、imgsz、evaluator、推論 backend 與評估設定，分別呈現兩類相對 parent 的變化。既有數據不足時標為待評估，不用整體 mAP 推算分類別結果。
- 執行前核對模型 `names`、YOLO 連續類別索引與 COCO annotation `category_id` 的映射；上表索引不可直接當成 annotation category ID。
- 此目標補充既有整體 COCO／person 與八項主要 AP，不自動改寫既定選型門檻；若要把兩類表現納入採用門檻，須在對應階段開始前固定規則。
- COCO Detect 與 BBAT5 v1 的 ball／bat Detect、Pose 指標分開呈現。BBAT5 的 canonical 路徑、來源、影像、labels 與 split 保持不變；不新增訓練任務或資料版本。

### 模型必須攜帶的設定

模型 manifest 至少記錄：

- Binary bases、sites、heads、scale slots。
- scale 格式、bias 格式及部署常數。
- `exp_backend`、PWL 範圍與分段。
- `reciprocal_backend`、LUT／係數、Newton 次數。
- 位寬、fraction bits、捨入、飽和及 token 範圍。
- activation 類別、固定係數、替換範圍與 call-sites。
- RepConv 位置，以及 training／folded 狀態；訓練與部署各自的 activation／normalization／dtype、轉換流程及驗證結果。
- 各運算的實際 backend／dtype、常數除法與動態 reciprocal 政策、仍保留的浮點／昂貴運算及原因。
- 輸入 `H×W`、resize／letterbox／padding、各層 stride、Attention sites 的 `H、W、C`、每列 token／分組／mask、bias 表範圍與支援的 shape profiles。
- 形狀改變後的分母上限、累加位寬、reciprocal 有效範圍，以及 bbox／keypoint 座標還原契約。
- 匯出格式、靜態／動態尺寸、實際部署算子查核與驗證結果。
- parent、checkpoint SHA、資料與 evaluator。

後續階段必須繼承對應模式的設定。訓練與部署可明確指定不同算子，但缺少必要配置時必須失敗，不得因缺配置而靜默回退。正式部署不得誤載訓練用浮點 normalization 或預設 SiLU。遇到未支援的 shape profile、token 範圍或 bias 索引時同樣明確失敗；完成新形狀驗證並更新 manifest 後才能接入。

### Softmax 驗證

確認 PWL 的 `exp(0)=1`、權重非負且不超過 1，再依實際 token 數推導分母與累加位寬。

測試相同 scores、單一主導值、並列最大值、二次冪邊界、查表分段邊界、最大 token 數、尾端飽和與非法零分母。

分開量測 exp、reciprocal、輸出量化誤差，以及 row-sum、probability、P·V 和任務 AP。

```text
sum(P) = D × approximate_reciprocal(D)
```

不能最後再用浮點除法強制正規化，掩蓋倒數近似誤差。

首輪完成定點 reference 與成本分析。未做 RTL／HLS 綜合或板端量測時，不宣稱實測加速；QKV、P·V、projection 是否仍浮點另行標示。

### qSiLU／RepConv 驗證

state dict 本身不能完整表達 activation 類型，因此載入時依 manifest 與 training／deployment 模式重建；正式部署必須核對 qSiLU。

folding 等價檢查前後保留相同 activation，分別驗證線性折疊與完整輸出，在 FP32 及實際 dtype 記錄誤差。後續切換 qSiLU／D1 的誤差與任務 AP 另外量測，不能把 folding 通過當成算子替換無損。不能只測全零新增分支，也不能因權重載入成功就當作部署 activation 正確。

## 五、資料夾責任、監測與交付

| 資料夾 | 本輪責任 |
| --- | --- |
| `yolo_binaryqk` | branches、Q10／二次冪 scale／bias |
| `yolo_attention` | 唯一 Attention、normalization、PWL＋reciprocal |
| `yolo_masf` | Detect／Pose task-specific MASF |
| `yolo_hog` | HOG |
| `yolo_combine` | 共享初始化與雙任務模型 |
| `yolo_activation` | activation 比較、敏感度及 qSiLU parent |
| `yolo_achitechure` | 原本 RepConv 訓練流程；最終 qSiLU／完整 D1 硬體友善部署 |
| `yolo_actshape` | 第 9B 階段：特徵圖形狀、長方形／二次冪、padding／分塊及精度成本研究；2026-11-08 已授權進行 |
| `yolo_pose_training` | 受保護的已訓 Pose 來源 |
| `yolo_optimize` | 唯讀主要參考 |
| `yolo_attention_final` | 遷移必要能力後處理移除 |
| `yolo_p2` | 退出主線，核對依賴後處理 |
| `yolo_quantize` | 未指定前不重建、不啟動整網量化 |
| `original`、registry | 保持資料與 lineage |
| 根層 `docs` | 計畫、研究索引、交接與工作紀錄 |

實驗啟動前固定 parent、資料、配置、預算、selection rule 與停止條件。參照 optimize 對應階段配方，依實際任務核對，不直接照搬所有超參數。

GPU 工作每 600 秒監測，完成或異常立即處理。

每階段必須交付可執行資料夾、實際結果、推薦理由、下一階段模型與完整設定、小型數據、圖表來源、清理清單及中文工作紀錄。

CPU smoke、正式 AP、訓練、定點 reference、硬體綜合與實板量測分開標示。

**先保存計畫；之後依指定階段清理重建與實驗。RepConv 訓練不強制使用簡化算子，所有候選的最終部署必須保留 qSiLU 及完整硬體友善契約；最後再完成全模型與特徵圖形狀檢查，以及已確認無用內容的移除。**

初次保存與驗證見 [2026-09-22 工作紀錄](../worklogs/2026-09-22-yolo-integrated-research-plan.md)；工作起點與 COCO 球棒／球目標更新見 [2026-10-03 工作紀錄](../worklogs/2026-10-03-coco-ball-bat-evaluation-goal.md)；硬體友善、特徵圖形狀與清理補充見 [2026-10-17 工作紀錄](../worklogs/2026-10-17-hardware-friendly-shapes-plan.md)。返回[根層 README](../../README.md)或[工作紀錄索引](../worklogs/README.md)。

## 2026-10-18 執行補充

第 1 階段的共同 parent、固定 scale 校準、Attention 初篩與選定分支全模型恢復、evaluator、硬體運算及停止規則以 [Binary stage1 protocol](../../yolo_binaryqk/docs/stage1-protocol.md) 記錄。長訓退出 working 前必須確認程序與監測存活；預估時間依實測更新，背景紀錄不等於聊天推播。完成該階段後停止，沒有下一階段的預先授權。
