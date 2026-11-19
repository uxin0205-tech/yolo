# 2026-09-01：量化架構、實驗矩陣與文獻稽核及v4安全修正

## 目標與授權邊界

依使用者要求，完整稽核目前Full35 activation-aware weight quantization架構、V0–V3歷史產物、V4+實驗配置與可用學術方法，修正可在CPU完成的工程問題，並把可研究的新方向寫成可公平否證的計畫。

本輪沒有使用GPU，沒有執行PTQ calibration／probe、mAP validation、QAT、正式訓練、export或硬體benchmark；也沒有修改`yolo_combine/final`、`yolo_activation`、checkpoint、BBAT5影像、標註或split。active runner仍為`execution_authorized: false`。

## 變更內容與原因

### 架構Module與Seam

- `AppliedActivationQuantization.rebind(model)`：讓已審核的activation observer／fake-quant控制能安全綁到等價clone或BN-folded deployment View；path、policy id、bit與signed contract任一漂移都fail closed。
- active weight runner先以`Full35WeightViewAdapter`建立unfused master與BN-folded deployment，再量化deployment；不再把未fold BN graph當成部署PTQ。
- `Full35MetricGateSpec`集中管理`0.04／0.01／0.01／0.06`門檻；snapshot新增`metric_contract_id`與`policy_id`，拒絕跨evaluator、split或activation parent比較。
- runner執行需CLI明確acknowledgement與YAML authorization同時通過；有限且同結構的單張輸出只標`diagnostic_pass`，固定`selection_claim=false`及`promotion_authorized=false`。
- package facade對weight-sensitivity CLI改用lazy public export，消除`python -m`重複匯入的`RuntimeWarning`，同時保留`from yolo_quantize import WeightSensitivityStudy`介面。

### 固定diagnostic manifest

- `DiagnosticManifest.from_json`會驗證canonical JSON digest、固定32 calibration／64 probe、BBAT5 v1 assignment與sample schema；v2 plan另外保存預期SHA，避免內容與self-declared digest一起被替換。
- `verify_files=true`時逐一重算所有image／label SHA-256；active runner在任何GPU工作前必須完整驗證。
- 現有comparison Interface每task只比較一張probe，因此固定取manifest排序第一張並寫入subset policy；移除`--calibration-per-task`臨時覆寫。
- COCO calibration／probe維持person coverage；BBAT5只沿用不可變v1 assignment，未建立新資料版本。

### 強化weight與SD4 baseline

- 稽核發現歷史`mse`實際只在十個`absmax × fraction`點搜尋，已保留舊artifact並將語意定名為`mse_grid_v1`。
- 新增`optimal_scaled_codebook_scales`，對固定signed codebook做exact event sweep；uniform與Fixed SD4共用同一Module。tiny tensor測試以獨立assignment搜尋核對objective，並確認exact不劣於舊grid。
- 新增`SD4Encoding`：保存15個logical values、16個encoded nibbles、`0111／1111`雙zero、canonical export zero、high-nibble-first packing及奇數padding round-trip。
- uniform設定與實作統一為zero-point 0及完整two's-complement：W8 `[-128,127]`、W7 `[-64,63]`、W6 `[-32,31]`、W5 `[-16,15]`、W4 `[-8,7]`。
- 修正合法全零weight group的SQNR `log10(0)`崩潰；reference與error energy使用同一finite floor。
- 舊8,880筆與34層SD4 route不覆寫，只能作`mse_grid_v1`歷史proxy。完整Full35 exact W4／SD4重算及routing v3尚未執行。

### 實驗矩陣與gate重整

- active uniform parents固定為qSiLU＋A8、Hardswish＋A8、poly_shift＋A8；`poly_quality`只保留歷史artifact。
- V4保留全部W8／W7／W6／W5／W4共15個quantized cells；執行順序為W8→W6→W4粗曲線，再補W7／W5 boundary。
- Q3 regional Hardswish保持qSiLU權重根，先做三個單區matched FP-weight／W8共6格；兩個指定單區都通過才做2格條件式組合。
- V5的150格拆成T0 static、T1 diagnostic、最多30格T2 search與最多6格T3 formal；BBAT5 search使用既有`pose-search.yaml`，formal `pose.yaml`只確認locked finalists。
- blocker分層：integer boundary阻擋首個W8 GPU bridge；Full35 exact profile阻擋W4／SD4 promotion；fold-aware effective weight只阻擋QAT；hardware target只阻擋速度／能耗winner宣稱。

### 學術查核與研究方向

- 以33組第一手論文／官方文件查核detector PTQ／QAT、mixed precision、LSQ／LSQ+、scaled codebook、APoT、ternary與activation方法。
- CVPR 2021 scaled-codebook工作證明舊十點grid不是足夠強的Fixed SD4 baseline，因此先補exact baseline，再允許LS-SD4。
- 建議四個可否證方向：activation-conditioned worst-task routing、shared Detect＋Pose task-output-loss PTQ、exact-to-learned SD4 continuum，以及Q3 activation placement×weight bit二因子interaction。
- 以上只稱研究假說；未完成引用網路、專利與最新索引查核前，不宣稱novelty。

## 診斷紀錄

### qSiLU BN-fold parity假警報

- Red：將真實Full35 View整合測試由歷史`poly_quality`改為qSiLU、但仍使用adapter預設accepted checkpoint後，既定parity失敗且可重現。
- 實測預設checkpoint＋qSiLU為NRMSE `1.4290916e-5`，超過`1e-5`；structure、finite與peak-relative `5.8319458e-5`正常。
- 使用active qSiLU recovery checkpoint SHA `767918…6190e`後，NRMSE `3.3907510e-6`、peak-relative `7.1056924e-6`，完整通過；124個deployment observer皆為observe模式。
- 根因是測試拆開了activation與parent checkpoint，不是fuse錯誤。修正方式是測試與runner都綁定真實checkpoint SHA，保留原嚴格門檻，沒有放寬容差。

### Weight catalog alias稽核

- CPU唯讀檢查251個catalog sites對應251個不同physical Conv／Linear，duplicate groups為0、cross-status duplicates為0；目前reversible region quantization不會因module alias重複套用。

## 驗證方式與結果

- 基線完整CPU回歸：`62 passed`。
- manifest binding目標測試：`7 passed`。
- 三個新增Red測試分別捕捉malformed BBAT source的`AttributeError`、全零weight的`math domain error`及錯parent checkpoint的parity拒絕；修正後同三測試`3 passed`。
- range／manifest／runner目標回歸：`20 passed`。
- plan-to-manifest SHA pin與乾淨`python -m` CLI均先Red再Green；非對稱two's-complement exact solver與逐檔manifest hash另有目標測試。
- 最終完整回歸在`CUDA_VISIBLE_DEVICES=-1`下為`83 passed in 63.24s`。
- `ruff check src tests scripts`通過；`ruff format --check src tests scripts`回報37個檔案均已格式化。
- 唯讀解析20份YAML、16份JSON與29份Markdown；本機連結缺失為0。
- active list-only smoke無stderr警告，輸出`graph_view=bn_folded_deployment`、`execution_authorized=false`、manifest SHA `b0eb2a…eef2`及2個指定cells。

## 遇到的困難與解法

### bwrap無法建立loopback

- 困難：一般sandbox命令與直接patch helper多次回報`bwrap: loopback: Failed RTM_NEWADDR`。
- 解法：仍使用規定的`apply_patch`，透過受核准TTY session套用；formatter只處理本輪Python檔案。沒有使用`cat`、Python或shell redirection寫專案檔。

### 歷史artifact命名過度宣稱

- 困難：舊`mse`名稱看似全域MSE最佳，且34層route已被文件寫成現行候選。
- 解法：不改舊bytes／hash，將其解釋固定為`mse_grid_v1`，另立exact solver、v4契約與未來routing v3。

### Git工作樹與遠端分歧

- 困難：父repo為`main...origin/main [ahead 1, behind 54]`，且`yolo_quantize`整體未受追蹤，不能把任何一側當成可覆寫基線。
- 解法：本輪沒有checkout、reset、pull、commit或push；只修改可寫子專案並保留所有歷史產物。發布需另走finish-work稽核。

### 近期文獻成熟度

- 困難：部分2025–2026相關工作仍為預印本，且尚未固定target hardware。
- 解法：同行審查與預印本分開標示；只提出protocol與假說，不做速度、新穎性或最佳方法宣稱。

## 未解事項與風險

- integer boundary contract尚未實作，第一個W8 GPU bridge仍被阻擋。
- Full35 exact W4／Fixed SD4 profile與routing v3尚未產生；舊34層route不能promotion。
- fold-aware shadow QAT與folded-graph QAT尚未二選一；任何QAT都被阻擋。
- 尚未固定FPGA／ASIC／runtime／compiler／kernel，不能以ideal packed bytes宣稱實測速度或能耗。
- 尚無新版PTQ八項mAP、正式activation winner、LS-SD4、A-SD4、Channel-TWN或TTQ結果。
- calibration size 32／128／512與bootstrap uncertainty ablation仍只在計畫中。

## 主要產物

- `IMPLEMENTATION_PLAN.md`
- `configs/experiments/full35-quantization-plan-v4.yaml`
- `configs/experiments/v4-plus-prepared-plan-v4.yaml`
- `configs/experiments/weight-region-sensitivity-plan-v2.yaml`
- `docs/reports/2026-09-01-architecture-experiment-audit.md`
- `docs/research/2026-09-01-quantization-literature-and-innovation-audit.md`
- `src/yolo_quantize/scaled_codebook.py`
- `src/yolo_quantize/sd4_encoding.py`
