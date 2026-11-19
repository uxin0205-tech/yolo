> **2026-11-17 全輪整理完成：** 正式交接統一至 [result/](result/README.md)，完整結論見 [研究總報告](docs/research/yolo-research-summary-2026-11-17.md)。Stage9 累計完成 120E；正式模型仍為 qSiLU Rep17＋20 E1／D1／原 rect640。本次另清除 6 份已完成 optimizer resume 與 92 個可重建快取，共 1.2074 GiB，沒有封存。MASF 與 optimize 外部依賴保留，逐檔原因見 [清單](result/cleanup-20261117.json) 與 [工作紀錄](docs/worklogs/2026-11-17-integrated-research-handoff.md)。

> 11 月 19 日已刪除舊整合參考目錄。歷史計畫中對該路徑的描述僅供追溯。現行步驟、兩份部署模型與形狀研究請從[根層 README](README.md) 進入。
>
> **以下是按日期保留的歷史計畫／清理證據，舊「尚未開始」及原自動選型不代表目前狀態。** 現行狀態以整合計畫、總報告與 root result 為準。

> **2026-11-04 Stage8 清理與啟動：** 刪除 6 個確認無依賴的舊 checkpoint，釋放 253.92 MiB，另移除可重建 cache；156 個受保護來源／權重 SHA 不變。三組 qSiLU RepConv CPU 驗證通過，GPU queue 已啟動。見 [清理清單](yolo_achitechure/stage8/cleanup-20261103.json)、[工作紀錄](docs/worklogs/2026-11-03-repconv-stage8.md)。下方為歷史紀錄。

> **2026-11-03 Stage7 整理完成：** 清除 810,934 個舊 runtime 連結、4,844 個檔案等，釋放 5.2484 GiB；保留外部仍依賴權重與小型來源，48 個受保護檔案 SHA 一致。本輪證據搬到 result/stage7，qSiLU／PolyQuality 原權重與可攜載入入口在 [handoff](yolo_activation/handoff/README.md)，CPU 等價驗證通過。Stage8 未啟動。[工作紀錄](docs/worklogs/2026-11-03-activation-stage7-handoff.md)。下方為歷史紀錄。

> **2026-11-02 清理：** 第七階段已指定。刪除舊 activation 的 226 個未使用權重／大型預測檔，釋放 26.137 GiB；四份外部仍用權重、release 唯一模型、必要小型證據與共用程式保留，SHA 核對一致。未封存。見 [清單](yolo_activation/stage7/cleanup-20261102.json) 與 [工作紀錄](docs/worklogs/2026-11-02-activation-stage7.md)。下方為歷史快照。

> **2026-10-31 清理完成：** 第六階段已直接刪除 17 份中間權重／optimizer、2 份 runtime labels cache，釋放 2.7115 GiB，只留選定 E1 實體模型；小型指標、配置、SHA 與圖表保留。第七階段已完成 CPU 準備，無 GPU 工作。[逐檔清單](yolo_combine/result/stage6/cleanup-20261031.json)、[工作紀錄](docs/worklogs/2026-10-31-stage6-cleanup-stage7-preparation.md)。下方為歷史快照。

> **2026-10-30 交付狀態：** 第六階段本輪研究完成並停止，選定成果已搬到 [result/stage6](yolo_combine/result/stage6/README.md)，原 artifacts 入口保留相容連結，未複製權重。KD 未採用大檔與已停止 resume 共 10 檔已直接刪除，釋放 742.60 MiB；更早重建時另釋放約 5.562 GiB。原精度目標仍有四項未達，不自動升格，第七階段待指定。未建立 archive／backup，optimize、資料及必要依賴保持。見 [完成紀錄](docs/worklogs/2026-10-30-stage6-results.md)。以下日期段落為歷史快照。

> **2026-10-21 資源清理：** 依使用者後續要求，未採用 HOG 的實體權重、E0 副本、正式 resume、預覽圖與快取已刪，另補清第 2 階段遺漏的四份 smoke 權重，本次釋放約 978.54 MiB。HOG 只留程式、小型證據及原主線連結，無 archive／backup。MASF 正式結果、候選與必要 resume 全數保留，待融合後再決定是否重試。當時第六階段待指定。[資源清理與前階段核對](docs/worklogs/2026-10-21-hog-resource-cleanup.md)。

> **2026-10-20 最新狀態：** 第四階段含延長研究已完成，兩任務建議保留原 parent。最佳候選與 resume 保留供覆核；延長流程另外移除 8 份 smoke 權重，釋放約 1.139 GiB。沒有封存，未啟動第五階段。見 [完成紀錄](docs/worklogs/2026-10-20-masf-stage4-results.md)。下方日期段落為歷史快照。

# Codex 任務書：YOLO26m 全專案整理、實驗證據鏈與論文圖表（省空間版）

> **第四階段接續：** 使用者覆核採用 KD 第五輪，`yolo_masf` 已清理並重建。抽取 123 份必要歷史證據後，直接刪除 479 個舊產物／程式檔，釋放 2.858 GiB。optimize 的兩個 CSV 依賴、受保護 baseline 資料與唯一來源均保留。見 [第四階段紀錄](docs/worklogs/2026-10-19-masf-stage4.md)。以下第三階段自動選型是覆核前歷史。

> **2026-10-19 最新接續：** 第 3 階段已完成並停止，選 D1 完整 PWL＋reciprocal，官方 AP 0.514189；交付於 `yolo_attention/result/stage3/`。已刪除／去重 322 個舊 checkpoint、預覽圖與快取檔案，移除 87 個空資料夾，釋放 **2.601 GiB**，沒有 archive／backup。四份必要唯一權重集中至 `yolo_attention/weights/dependency-*.pt`，舊位置保留相容連結；optimize 必要入口與小型證據保留。見 [第 3 階段完成紀錄](docs/worklogs/2026-10-19-attention-stage3-results.md) 與 [逐檔清理清單](yolo_attention/artifacts/stage3/cleanup-final-delivery.json)。後續 KD 五輪已於 16:48 完成，小幅提升未達採用門檻，仍保留 D1；另刪除兩份 smoke 權重，釋放 277.53 MiB，唯一正式 KD resume 保留。見 [KD 最終報告](yolo_attention/artifacts/stage3/kd-20261019-v1/RESULTS.md)。第 4 階段未啟動。下方是歷史快照。

> **2026-10-18 接續狀態：以 [YOLO 整合研究計畫](docs/research/yolo-research-plan-2026-09-21.md) 接續。** 第 1 階段 v2 於 23:10 完成並停止，選定 B2（官方 AP 0.509889）；全模型恢復 5 epochs 未改善，保留初篩最佳，見 [完成紀錄](docs/worklogs/2026-10-18-binaryqk-stage1-results.md)。使用者要求錯誤內容直接刪除、不封存，舊 v1 run／權重已移除，僅留小型數據與刪除紀錄。本輪有效比較、最佳與必要 resume 保留，沒有因精度未改善就刪除正式負結果。其他階段待指定。每個階段完成後先交付結果並停止；長訓穩定後退出 working，交代預估時間，保留每 600 秒背景監測。
>
> 新一輪研究的順序、候選、資料夾責任、清理時機與驗證契約以整合計畫為準。`yolo_optimize` 完全唯讀；Attention 必要能力遷入 `yolo_attention` 並驗證依賴後，再處理 `yolo_attention_final`。根層 `docs/research/` 承接本輪整合文件。
>
> 不新增未改方法的 baseline／control 加訓，也不新增 sign-only、scaled-sign、shared MASF 或 ReLU 候選。Rep17、Rep20、Rep17＋20 皆從同一份已驗證的 V6 qSiLU parent 獨立出發，訓練、folding 與部署均保留 qSiLU。
>
> 2026-10-17 補充：所有 RepConv 候選都要繼承 qSiLU、PWL／已選硬體友善 normalization、reciprocal 等完整契約，每階段核對除法與其他實際部署運算；第 9 階段再做全模型與長方形／二次冪特徵圖分析。
>
> 使用者已要求刪除無用資料夾：指定階段內完成必要證據抽取、替代驗證與依賴檢查後，實際移除已確認無用的內容，不只列建議或另存整套 archive。保留必要唯一權重、active resume、資料與 lineage，以及 optimize 仍使用的外部依賴；本次文件補充未執行刪除。
>
> 下文保留 2026-09-19 原始任務書，作為指定階段內的舊方法、結果與證據整理參考，不構成立即執行整份清理的指令。原有擴充 optimize、保留兩套 Attention、候選矩陣及 Activation winner 直接交給 RepConv 等規劃，若與整合計畫衝突，以整合計畫為準。

## 原始任務書參考

> 依使用者 2026-09-19 的需求整理。初步參考 GitHub `uxin0205-tech/yolo` 的 `main`，當時 SHA 為 `1d409c4fe21d3287d6f19be48f4457407a6bb50e`。
> 這個 SHA 只用於標示本任務書的參考版本，不是要求回退本機版本。本機可能有較新的程式、未提交成果及既有權重，必須先盤點，不得用遠端覆蓋。
> 本任務書尚未驗證使用者訓練電腦的權重存在性、正在執行的程序或可釋放空間；這些都由你在本機實際確認。

---

## 0. 我的目標與執行邊界

請整理目前用來研究的整個 `yolo` 工作區，不只是整理 README，也不是重新把所有模型訓練一遍。我要一套能看懂、能維護、能重畫論文圖、能追溯結果，而且不再累積大量重複權重的研究專案。

論文／報告的敘事順序固定為：

```text
研究背景與 Detect／Pose 基準
  → Binary QK
  → 不同 binary branches／bases
  → scale／bias 補償（包含 16 個固定 scale slots）
  → Softmax／PWL
  → 新版 MASF：分別確認 Detect 與 Pose 的效果
  → HOG
  → Detect–Pose combine
  → Activation 與替換敏感度
  → RepConv
  → 最終準確度、模型成本、限制與未採用結果
```

注意：這是「報告順序」，不代表所有現有權重都是照這個順序接續訓練。你必須另外建立真實的 checkpoint lineage。不得把不同 parent、不同模型世代或不同 evaluator 的結果畫成一條累積改善線。

請直接完成可安全執行的盤點、文件整理、必要的程式整理、CPU 測試與既有數據重繪，不要只回覆抽象建議。但以下工作需要另行確認：刪除 checkpoint／不可重建產物、啟動耗時的全量驗證或 GPU 訓練、停止現有工作、修改正式選用模型、推送遠端或改寫 Git 歷史。

本輪核心是量化前的架構與訓練研究整理。BinaryQK、PWL 與 Q10 常數本來就是此主線的一部分，必須保留；但不要擅自把它擴張成整網 PTQ／QAT、LS-SD4 或混合位元搜尋。`yolo_quantize` 及其他未列入主線的研究，先保留與建立索引，不因不在本次主線就刪除。


### 0.1 最高優先原則：保留「每個資料夾原本的研究職責」，更新的是內容，不是把功能抽空

這次整理**不是把所有功能集中搬進 `yolo_optimize`，再把其他資料夾變成只剩 README 的歷史殼**。每一個既有研究資料夾都要先辨識它「原本要解決的研究問題與可執行工作」，然後在清理後繼續具備該能力。

核心原則是：**preserve the mission, replace the contents**。

這次不是「把舊東西整理得更好後繼續留著」，而是把舊內容當成**一次性的 migration source**：

```text
讀取舊內容
  → 抽取真正需要的研究邏輯／結果／公式／設定／依賴
  → 重新建立乾淨、目前可用的 implementation／config／report／figure data
  → 驗證新版本已能完成原資料夾的研究任務
  → 建立 migration manifest，記錄舊檔 → 新檔／新資料的對應
  → 移除原本被參考的舊內容
```

也就是：

- 可以完整參考舊程式、舊 README、舊 reports、舊 CSV／JSON、舊圖、舊 run layout 與舊 checkpoint contract。
- 參考完後，不要因為「歷史可追溯」而把原內容整包搬到 `legacy/`、`archive/`、`old/`、`backup/` 再永久保存。
- 不建立「舊版一份＋新版一份」的雙份 workspace。最終每個研究資料夾只保留**新的 canonical 內容**與極小型 provenance／migration 記錄。
- 舊資訊若仍有研究價值，應先轉寫為新的 experiment registry、metrics CSV／JSON、method note、negative-results、checkpoint manifest、source-map 或 figure source；轉寫與核對完成後，再移除舊來源檔。
- 舊 YOLO11／舊 YOLO26 的方法與結果可以作為參考，但除非它是目前仍必要且不可替代的執行依賴，否則不保留整套 historical reproduction tree。
- 若舊 checkpoint 只用來抽取結果或轉換成新格式，先完成 load／hash／metric／mapping 驗證與必要的新 checkpoint 匯出；確認新版本可用後，舊 checkpoint 納入移除清單。
- 若某個舊 Python module 目前仍是客製 checkpoint 反序列化的唯一依賴，先做安全轉換或 state-dict migration；轉換完成並驗證後，舊 module 也應移除，而不是長期靠深層歷史路徑 import。
- 可以抽出真正共用的底層 implementation，避免同一個 BinaryQK、PWL、MASF、checkpoint parser 在多處各維護一份；但每個資料夾仍必須有自己的入口、設定、文件與 smoke test，讓使用者從該資料夾就能完成它原本的工作。
- 不要求保留舊檔名、舊目錄層級、舊 runner 或舊 `final/`；要求的是**研究目的、必要結果、可追溯性與主要可執行能力被遷移到新結構後仍存在**。
- 若某資料夾目前已壞掉、互相依賴混亂或只能靠舊路徑運作，這次可以直接重建；不要為了保存歷史而把壞的 code 留在 production tree。
- `legacy/`／`archive/` 只能作為**短暫 migration staging**，不能是最終交付。清理完成後應為空並移除，或只留下極小的 manifest／文字索引，不保留整份舊程式、run、圖或權重。

**重要例外**：`original/` 中的 canonical dataset／label lineage、仍在執行或需續訓的 active resume、尚未成功遷移且屬唯一來源的關鍵權重，不得因這條規則直接刪除。這些不是「為了懷舊保留」，而是目前仍有功能性或資料完整性需求；等新版本／替代品完成並驗證後，再依 cleanup plan 處理。

對每個既有頂層研究資料夾，都建立一份 `FOLDER_CONTRACT` 或等價文件，至少寫：

```text
folder
original_mission
current_mission
primary_entrypoints
required_inputs
expected_outputs
canonical_implementation
migration_from_original
original_items_consumed
original_items_replaced_by
external_dependencies
checkpoint_contract
minimal_smoke_test
full_reproduction_command
removal_gate
removed_original_inventory
known_limitations
```

**「整理完成」的定義不是資料夾變漂亮，而是該資料夾仍能完成原本那類工作。** 至少要能執行其核心流程的 dry-run／CPU smoke／config validation／checkpoint loading／結果重建之一，並在 README 明確說明正式 full run 的命令與所需資料。

`yolo_optimize` 的角色是跨階段整合、索引、比較、論文圖表與最終研究流程 orchestration；它**不是**用來取代 `yolo_binaryqk`、`yolo_attention_final`、`yolo_combine`、`yolo_activation`、`yolo_quantize`、`yolo_achitechure` 等原本的專項功能。


---

## 1. 資料來源與優先規則

### 1.1 三種來源各有用途，不用單一時間戳決定一切

- **本次使用者需求**：決定工作範圍、敘事順序與省空間目標。
- **歷史階段報告與 0922 個人簡報**分別保存使用者指定成果及報告編排參考。歷史階段報告的檔名日期是 0907，內頁標示 0908 個人會議。簡報的候選模型須以原實驗採用門檻及對應紀錄查核。
- **`碩一_陳宥炘_0831.pdf`**：歷史方法、實驗設計、消融與推導的參考，不把其中早期數值直接當成新版本結果。
- **本機原始紀錄、程式、設定、checkpoint 與遠端版本**：確認實際做了什麼、用哪個模型和資料、結果如何產生。文件與 raw evidence 不一致時，建立 discrepancy 記錄，不能悄悄改掉原結果。

若附件不在本機，先搜尋工作區內已有的相同文件或引用，找不到就列為缺件；不要假裝已讀過，也不要因此停止能由程式與原始數據完成的部分。不要搜尋整台電腦上與本研究無關的私人目錄。

### 1.2 優先閱讀的實際路徑

以下是本任務書初步查到的入口，先檢查本機是否存在及是否已有更新。

| 主題 | 優先入口 | 用途與警告 |
|---|---|---|
| 全域規範 | `AGENTS.md`、`README.md`、`CONTEXT.md`、`docs/agents/`、各子目錄 `AGENTS.md` | 注意根文件有 YOLO11／舊 remote 等歷史敘述，需與當前 YOLO26m 工作區核對，不可照抄舊 clone 指令。 |
| 整理範例 | `yolo_optimize/README.md`、`docs/WORKSPACE.md`、`docs/OPERATIONS.md`、`experiments/README.md` | 延用研究、結果、權重索引分離的做法；不要照搬「所有大檔永久保留」的舊政策。 |
| 正式版本 | `yolo_optimize/reports/current-model/README.md` | 與新候選、研究接續 parent 分開。 |
| 最近候選 | `yolo_optimize/reports/update-0914/README.md`、`experiments/post_binary_rep_v1/STATUS-20260914.md` | 實際 lineage 與採用狀態；不是所有完成者都 adopted。 |
| BinaryQK 歷史 | `yolo_binaryqk/README.md`、其 report／config／implementation | README 記載的是 YOLO11m 歷史研究，不能把 26 組歷史矩陣當成 YOLO26m 同口徑結果。 |
| YOLO26 Attention | `yolo_attention_final/final/README.md`、`BINARY_QK_PWL_ALGORITHM.md`、`RESULTS.md`、`yolo_attention/` | 對照 BinaryQK、Hadamard、PWL、checkpoint 契約與兩處 attention。 |
| 實際 Attention import | `yolo_achitechure/achitechure_1/final/code/yolo_attention/` 及本機載入來源 | optimize 文件指出某些 run 實際 import 此處；不能看成重複 final 就直接刪除。必須列出 `module.__file__`。 |
| 新 scale／bias | `yolo_optimize/experiments/attention_recovery_v1/`、`post_masf_hardware_v1/`、`post_binary_rep_v1/SCALE-BIAS-HARDWARE.md` | 核對固定 Q10 常數、bias tables、forward／backward 與硬體界線。 |
| MASF | `yolo_optimize/experiments/studies/pre-fusion-full35-b100/`、`experiments/combine/pose-masf/`、`pose_masf_priority_v1/`、`pose_masf_training_v1/` | 分清舊 shared MASF、Detect fork／bridge、Pose 專屬 MASF。 |
| HOG | `yolo_optimize/reports/direction1/README.md`、`docs/worklogs/2026-09-09-hog-result-ball-bat.md` 與對應 scripts／artifacts | 訓練輔助監督與小球覆蓋診斷；保留負結果。 |
| Combine | `yolo_combine/README.md`、`final/full35/`、`yolo_optimize/experiments/combine/bridge_v1/` | 原版與後續重融合不能混用權重、門檻及模型位置。 |
| Activation | `yolo_activation/README.md`、`reports/completed-activation-integrated-report.md`、`reports/full35-q3-cpu-final-report.md`、`src/activation_lab/`、`yolo_optimize/experiments/activation/` | 公式、候選、CPU 逐區敏感度與新融合 parent 的實驗需分開。 |
| RepConv | `yolo_optimize/experiments/post_binary_rep_v1/` 與較早的 studies／scripts | Rep17、Rep20、Rep17+20 的共同起點、fold 與 gate。 |
| 歷史失敗／非主線 | `yolo_achitechure/`、`yolo_attention/`、`yolo_quantize/` 等 | 僅索引、核對依賴及提出精簡建議，不預設無用。 |

### 1.3 已發現、必須處理的版本問題

本任務書參考版本的文件記載：舊正式選用模型是 qSiLU E2，含 BinaryQK＋PWL，MASF 只供 Detect P3 使用。0914 新候選則有 Detect／Pose 兩套 MASF，而且實際順序是：

```text
native QK E2 → Pose MASF B5 → BinaryQK scale/bias E5
                                      ├→ Rep17
                                      ├→ Rep20
                                      └→ Rep17+20
```

新 Rep 三組從同一 parent 分別出發，不是 Rep17 → Rep20 → 雙層依序接續。文件記載三組均未過原採用門檻，正式模型沒有自動替換。這些是遠端文件的狀態，仍須核對本機更新。

此外，請稽核下列問題並留下證據，不自行猜測修正：

1. 簡報所列 `Working candidate: Rep17` 與 repo 的「正式採用」是不同狀態；保留研究候選，不擅自升格。
2. 簡報表格的 `From (1)`、`From (2)` 等是否真的對應同一權重與 evaluator；尤其雙任務表格可能是在並列兩個獨立模型，不能直接稱同一 combined model。
3. 0831 報告 qSiLU 公式出現 `-144`，其後文字有 `-114`；以實際程式、分段展開與尾端約束測試核對，再明確記錄文件勘誤。
4. 簡報結果頁 ball／bat Pose 欄位可能與前面 baseline 順序不一致；以 class names、metrics keys 及 raw data 查核，不靠數值高低猜類別。
5. 參數節省有舊版 `45,580,762 → 26,529,701` 與簡報約 `42.8M → 26.6M` 等不同起點，不得交叉使用分子分母，或統一套同一百分比。
6. 簡報第 8 頁的 XML 在一般解析時有編碼錯誤，若本機也遇到，保留原檔，使用唯讀解壓／容錯讀取及 PowerPoint 實際顯示核對；不要直接重存或損壞原簡報。

---

## 2. 開始前的安全盤點

先記錄 repository root、`git status`、branch、commit、remote、tracked／untracked／ignored 大檔、磁碟實際可用空間、現有 Python 環境、GPU 與正在執行的研究程序。遵守所有適用的安全與資料規範。盤點先用 metadata 與既有 manifests，只有必要的 checkpoint／刪除候選再計算完整 hash，不對所有資料集影像重複做昂貴的全量掃描。不盲目反序列化來源不明的 checkpoint；先核對信任來源與格式。

優先在 `research/thesis-cleanup-20260919` 分支工作，或使用不衝突的新分支。工作樹不乾淨或存在 active run 時，不得自動 stash／reset／切換到會改動其來源的版本。先在不影響執行中的位置產出盤點及文件，必要問題集中詢問。

禁止自動 `git pull` 覆蓋現況、`git clean -fdx`、`git reset --hard`、force push、kill 訓練程序、改寫 Git 歷史、全碟清 cache、重建整個環境、重新下載整套模型或資料集。

Git branch 不是未追蹤 checkpoint 的備份，也不是資料集的復原方法。把檔案從 Git 移除與從本機刪除必須分開決策。

資料契約先沿用既有 registry：COCO80 Detect 與 BBAT5 v1 的 Pose／二類 Detect 是不同 task views。不得重切 train／val、抽換影像、修改 labels 或複製出第二套資料集。若本機根路徑不同，只做可追溯的路徑映射。

---

## 3. 統一實驗 registry、結果格式與證據鏈

先建立 machine-readable registry，再整理圖表。現有 CSV／JSON 優先匯入；沒有 raw source 的數字標示 `reported_only`，不能假裝已重驗。

每個 experiment／run 至少包含：

```text
experiment_id、run_id、family、model_family、task、status
source_directory、code_commit、local_change_manifest、config_path／hash
parent_run_ids、parent_checkpoint_paths／SHA256、initialization_method
architecture_fingerprint、head_type、attention_variant、MASF_topology
activation_policy、repconv_sites、deployment_backend／precision
source_checkpoint_kind（live／EMA／inference／resume／folded）
dataset_version、split／label_manifest_hash、class_names、keypoint_contract
imgsz、seed、train_scope、BN_policy、optimizer、LR_groups、scheduler
actual_epochs、updates、physical_batch、accumulation、macro_schedule
loss_weights、reference_batch_size、trunk_interpolation_ratio
validator／version、evaluator_settings、selection_rule、gate_reference_id
metrics_source、metrics_path、figure_data_paths
completion_state、acceptance_state、limitations、missing_evidence
```

「已完成」「通過原 gate」「使用者選作研究接續」「正式選用」分成不同欄位，不用一個 `best=true` 混在一起。`planned`、`cancelled`、`interrupted`、`negative`、`historical` 均保留真實狀態。

統一 metric keys，例如：

```text
coco/box/map50_95
coco/person/box/map50_95
bbat/box/map50_95
bbat/pose/map50_95
bbat/ball/box/map50_95
bbat/ball/pose/map50_95
bbat/bat/box/map50_95
bbat/bat/pose/map50_95
```

內部 AP 優先存 0–1；圖表可顯示 0–100，但標示清楚。`Δpp = 100 × (AP_new − AP_reference)`；相對百分比另算，不能把 AP 差值、百分點與相對降幅混為一談。缺值使用 null 並附原因，不能填 0、複製旁邊的數字或插值造出結果。

成本分開記錄 trainable parameters、training graph parameters、deployment parameters、Conv MAC、其他運算、實際檔案大小、runtime peak memory、latency 與 energy。若採 `FLOPs = 2×MAC`，標明其適用統計範圍；XNOR、popcount、PWL、Hadamard 不任意併成同一 MAC 數。

產生兩份不同的圖：`presentation_order` 與 `checkpoint_lineage`。lineage 的邊區分 checkpoint 繼承、只移植模組、共享參數插值、等價部署轉換、純推論開關及尚未執行的提案。圖上每個數值能追溯 run ID、checkpoint SHA 與 evaluator。

---

## 4. 逐階段整理與實驗邏輯

### B0. 原始 Detect／Pose 基準與既有權重

先找本機已訓練好的 YOLO26m Detect 與 ball／bat Pose 權重，以及其資料契約、架構、輸入前處理、類別順序與訓練紀錄。不要因為 GitHub 沒有 `.pt` 就認定不存在，也不要直接重新跑 Pose P1→P2→P3。

將既有 Pose 視為受保護 source asset。先驗證其可載入與推論，保存可追溯索引；如和目標模組不相容，輸出逐 tensor 的 shape／key／mapping 報告。禁止用 `strict=False` 悄悄忽略缺失，再宣稱成功繼承全部權重。

YOLO11m 的早期 P2／BinaryQK 實驗保留為歷史探索；正式 YOLO26m 基準另建 ID。資料或 evaluator 不同的 baseline 分開列，不強行合併。

### E01. BinaryQK → 不同 branches → scale／bias → PWL

**研究問題**：QK 二值化造成多少損失？不同二值基底如何補足資訊？幅度／位置補償與 PWL 各自帶來什麼取捨？

依現有證據整理 FP-QK、identity／sign-only、scaled-sign、residual dual-basis、Hadamard matched dual-basis 等真正存在的設計。保留歷史 T3／T4／T5 對應關係，但註明所屬模型與訓練設定。不要補造缺少的消融結果。

每個 branch 圖要畫清楚 Q、K 從哪裡分流、是否先轉換再 sign、score 如何融合，以及 Softmax／P·V 執行幾次。Hadamard branch 不能畫成第二個完整 YOLO 或沒有成本的運算。

核對二值點積 `Z = 2 × popcount(XNOR(sign(Q), sign(K))) − d` 的符號／零值契約與精確前向。測試 Q／K projection 真實梯度能否回傳；若使用 STE／surrogate，區分訓練反向代理與推論路徑，不得畫成推論還需要額外 FP-QK。

scale／bias 必須分開呈現：dynamic magnitude、固定 power-of-two、固定 dyadic／Q10，以及 no bias／dense 2D／decomposed 2D；只納入已有或明確待補的組別。**16 個 scale slots 是 2 sites × 4 heads × 2 bases，不是 16 個實驗、不是每張圖從 16 個值挑 1 個，也不是整網 16 個 activation scale。**

新候選文件中的形式是：

```text
score = (m0·Z0 + m1·Z1 + bx[Δx] + by[Δy]) / 1024
```

請從實際選定 checkpoint／constants.json 匯出兩個 sites 的 4×2 常數，不手抄舊報告冒充目前模型；舊 power-of-two 版本與新 Q10 整數碼分開。`/1024` 表示 fractional precision，不表示整個數值只需 10 bits。固定常數不等於所有乘法都只需一次 shift，也不等於 PyTorch reference 已是 packed-bit kernel。

PWL 需整理 row-max、score 範圍處理、exp 近似、row normalization 與 P·V。比較 existing exact／LUT／PWL／SHIFT 及失敗替代方案；[-8,0] 與 [-10,0]、16 段與 20 段的差異應用同一 score source 比較，並確認實際 clamp／飽和與捨入規則。

優先以既有資料製作 score histogram／CDF、尾端比例、probability mass、P／P·V 誤差與精度／成本比較。需要補蒐集時採 streaming histogram、online statistics 或固定小樣本，不保存所有影像的完整 N×N attention tensor。

硬體界線要寫清楚：Bit-True PWL 不代表整個 Attention 已純整數化；若 normalization 仍使用軟體浮點 reciprocal／division，或 QKV／V／proj 仍浮點，直接註明。沒有目標 FPGA 實測，不能宣稱實測加速或節能。

### E02. 新版 MASF：Detect 與 Pose 分別驗證

**研究問題**：新版 MASF 對哪個任務、哪個尺度有幫助？部署 head 的監督是否真的回到模組？增益是否只是多訓練幾輪造成？

先做 `MASF_VERSION_AUDIT`，列出舊 MFAM／Full35／Partial75、新 residual、shared neck、Detect fork、Pose fork 與 bridge 的實際差異。不要只因檔名叫 final 就選它，也不要把 YOLO11m 舊數據當 YOLO26m 新版驗證。

預期核對的新版形式為：

```text
context(x) = Project1x1(DW3x3(x) + DW5x5(x))
y = x + alpha × context(x)
```

Conv、BN、activation 的順序、通道數、alpha 初始化與凍結範圍以實作為準。若有 Partial75，說清楚 context 比例與 exact bypass，不將「75% bypass」寫成「75% 通道進 MASF」。

至少建立三張不同資料流圖：

```text
A. shared P3 MASF：可能連帶改變後續 P4／P5 的生成。
B. Detect-only fork：raw P3 保留供 neck 下採樣／Pose，MASF P3 只供 Detect。
C. independent Detect／Pose forks：兩個 task-specific MASF 各自作用於 head 輸入。
```

依程式追蹤 layer.from 與 tensor shape，不只照舊圖畫。舊正式版本只有 Detect MASF 時，不得宣稱 Pose 直接受它增強。

整理 Detect 無 MASF／新版 P3 MASF 的配對，與既有 Pose 無 MASF／Pose 專屬新版 MASF 的配對。P2、Partial75 或 combined 的 none／Detect-only／Pose-only／both 只納入已存在結果；缺少的先列成最小補驗方案，不自動展開全排列訓練。

one2one 路徑的 detach 是必要稽核點：若 MASF 輸出先被 detach，部署 head 的 loss 就可能無法直接教 MASF。bridge 的 gradient scaling 係數 beta、推論 residual 係數 alpha，以及 task loss weights 必須分開。若 beta 只控制回到 MASF 的梯度，不可改成乘整個 one2one loss 卻宣稱等價。

alpha=0 時，context 的任務梯度可能暫為零；測試 alpha 的真實更新與後續 context 梯度，不能只看總 loss 下降。凍結共享 trunk 時，同時檢查 BN running state／EMA，避免「權重 frozen 但特徵仍漂移」。

若原 B 組有 Pose head+MASF 重訓而 A 組同預算對照已取消，完整保留限制。alpha-off 的同權重測試不等於獨立的 no-MASF 配對重訓。將新增公平對照列為待授權補驗，不假稱已完成，也不擅自重啟已取消工作。

### E03. HOG

從 optimize 找出 HOG target、預測支路、feature level、mask、bins／normalization、loss 權重與啟用 schedule，清楚說明它在這個專案是訓練輔助監督還是實際部署模組。

若為訓練輔助，只畫在 training graph；推論圖不得憑空保留 HOG head。與同 parent、同回合的 control 比較，不用 HOG E4 對 control E5 來作因果結論。

保留原負結果、small-ball P3 cell-center coverage 診斷、loss／gradient 與完整主要指標。對小球 mask coverage 的觀察不能泛化成所有 ball／bat 場景已證實的原因。不要為了讓論文每節都「成功」而刪掉負結果或改寫結論。

### E04. Detect–Pose combine

**研究問題**：用同一共享 backbone／neck 配兩個 task heads，節省多少部署參數，付出哪些任務精度代價？

以 `yolo_combine` 的原始設計加上 optimize 後續修正為來源，確認真正採用版本，不把整個舊 final 直接複製成新專案。

必須分清四件事：

```text
1. Pose trunk interpolation ratio rho：共享參數初始化的方法。
2. Detect／Pose task loss weights wd、wp：訓練目標的相對權重。
3. dataloader／macro-step schedule：每次 update 實際取哪些 batches。
4. reference_batch_size：loss backward 的數值尺度。
```

若實際採共享參數插值，僅對完成對齊的 shared tensors 使用：

```text
theta_shared_init = (1-rho) × theta_detect + rho × theta_pose
```

rho=0／0.1／0.25 是初始化探索，不代表 loss weights 或資料比例；更不是每次推論把兩個完整模型輸出加權。head 各自繼承其任務權重。BN buffers、整數 counters、缺少的 MASF、channel order 與不相容 tensors 需有明確處理契約，不能盲目平均整份 state_dict。

兩個 dataloaders 與原生 task loss／visibility／keypoints 契約保留。依本機實際程式核對：

```text
detect_mean = sum(detect_raw) / actual_detect_sample_count
pose_mean   = sum(pose_raw)   / actual_pose_sample_count
joint_mean  = (wd*detect_mean + wp*pose_mean) / (wd+wp)
backward_scale = reference_batch_size
```

先確認原 loss 是否已做 batch scaling，避免再次除以 batch。J0 只有 Pose active 時的 denominator 需與 joint active-task policy 一致；不得只因看到 `wp=0.25` 就額外縮小 head-only 訓練。歷史 `2 Detect : 1 Pose` 與新版 microbatch 排程不同時，分版本列出，不統一編造一個 batch size。

整理 existing J0–J3 的 frozen scope、LR groups、BN policy、實際 epochs、gradient norm ratio／cosine、loss 與八項驗證指標。已訓練的 Pose baseline 直接重用；重融合不等於重頭重訓 Pose。

COCO Detect 是 80 類、person 是其子類指標；BBAT box 是 Pose head 的 bbox 分支，不能接錯成 COCO head 的二類結果。所有 gate、joint score 與接受理由保留原定義；研究接續模型不等於八項 gate 全通過。

參數節省使用同一架構版本、同一部署分支範圍計算。若完整系統仍先在整張圖做 Detect、再對 ROI crop 做 Pose，兩個輸入不同時不能直接宣稱 backbone forward 已共享一次；將「權重共享」與「端到端資料流的計算重用」分開。

### E05. Activation 與敏感度

依原資料整理 SiLU、qSiLU／qsilu_pq、Hardswish、ReLU、poly_quality、poly_shift，以及實際存在的 LeakyReLU／SIPA／BCSP。未完成的 mixed policy 不寫成完成結果；中止的 20-epoch finalist 不取代完成的 10-epoch 候選。

至少分成三層證據：

```text
A. 數學函數近似：函數值／導數／連續性／尾端與計算式。
B. 不訓練替換：同 checkpoint 的 zero-shot 任務指標下降。
C. 恢復訓練：相同 parent、可比預算下的精度恢復。
```

敏感度的正式名稱與粒度必須準確：per-layer、per-call-site、per-region 不可互稱。原有 22/22 CPU region 評估不能寫成全模型每層都測過。0907 的 weight-only quantization error／148 deployment Conv/Linear 掃描也不能冒充 activation task sensitivity。

先建 activation site manifest，包括 graph path、module object alias、tensor shape、region、training-only／deployment-only／both。檢查共享的 `Conv.default_act` 及 `named_modules(remove_duplicate=...)`，避免漏算位置，或想換一層卻把所有共享物件一起換掉。修改需用測試證明受影響的 call-sites 正確。

region 可沿用 early/deep backbone、attention、neck、MASF、Detect head、Pose head 等，但 site 數量從實際選定模型重新列舉。one2many、flow／loss path 與 one2one 部署 path 分開，不把訓練-only site 算成推論節省。

整理 region × activation 的八項 AP drop 或既有真正測得的 proxy heatmap，標清資料規模與粒度。只以 proxy 篩選者不能稱完成正式 AP 驗證。先利用已有結果挑少數有根據的替換方案，不展開全模型×全部 activation×全部 seed 的無限制掃描。

qSiLU 需測試 knot 1／2／4／8、分段公式、C1 條件與 exact ReLU tails。近似 MAE 小不保證任務 AP 一定最好。硬體比較分清 hinge 展開式與實際 piecewise evaluator，不把未實作的理論運算量當成軟體或 FPGA 實測。

### E06. RepConv

整理 layer17、layer20、17+20 與 control，優先使用已有實驗，不自動再訓。以實際 module path、stride、channels 與 graph 角色標示位置，避免模型搬移後 layer index 對錯。

畫出訓練多分支與推論 fold 成單一 Conv 的圖。依實作確認是否為 3×3+1×1、是否有 BN、activation 放哪裡。不要憑一般 RepVGG 示意圖加上不適用於 stride=2 或通道不符的 identity branch。

驗證初始化等價、訓練後非零分支的 fold 等價、eval BN folding、重載等價。測試 FP32 與實際使用 dtype 的合理誤差容差，不只測一個全零新增分支。fold 後 inference checkpoint 與可繼續多分支訓練的 resume checkpoint 是不同用途。

Rep17／Rep20／雙層應採同 parent 的平行比較；若論文新的預期主線要從新的 Activation winner 接續，而歷史實驗不是該 parent，分成歷史結果與待執行新實驗。禁止透過改名讓 parent 看起來一致。

「fold 後額外 Conv MAC=0」不等於整網 latency／power 已量測為完全相同。單一 seed 小增量不稱統計顯著，也不自動修改原 gate 讓候選通過。

---

## 5. 論文圖表交付規格

不要只把報告截圖放大，也不要憑文字把不同版本畫成同一架構。以實際模組、tensor shape、raw metrics 及 lineage 重畫。

建議圖組如下；有資料就產出，缺資料則在 figure manifest 標為待補，不生成假曲線：

| 圖組 | 核心內容 |
|---|---|
| F00 | 論文敘事順序圖與真實 checkpoint lineage 圖，分成兩張。 |
| F01 | 原生 Detect／Pose 與目前選定模型的完整 architecture；training／inference 分開。 |
| F02 | FP-QK vs sign／XNOR／popcount；8-bit toy example 明確標為示意，不冒充實測。 |
| F03 | Identity、residual 與 Hadamard branches 的資料流及 Softmax／P·V 次數。 |
| F04 | 2 sites×4 heads×2 bases 的 16-slot 配置、固定常數與 bias 查表示意、相關消融。 |
| F05 | PWL 曲線、score 分布／tail、[-8,0] vs [-10,0] 的可用誤差與精度結果。 |
| F06 | 新版 MASF 模組、shared vs task-specific placement、alpha／bridge beta 梯度路徑。 |
| F07 | MASF 的 Detect／Pose 分別比較及 P3／P2 成本取捨，標明獨立模型或融合模型。 |
| F08 | HOG training-only supervision、mask coverage 與 matched-control 負結果。 |
| F09 | 獨立雙模型 vs shared trunk＋雙 heads；trunk ratio 與 task weights 分開畫。 |
| F10 | Combine 分階段收斂、gradient balance／cosine 與八項任務結果。 |
| F11 | SiLU／候選函數、誤差／導數、zero-shot 與 recovery 的差異。 |
| F12 | Activation site／region sensitivity heatmap 與選擇性替換 policy。 |
| F13 | RepConv train-to-deploy folding 與三種 placement 的比較。 |
| F14 | 最終精度、參數／weight storage、已量測成本與 Pareto trade-off；未量測欄位明示。 |

每張結果圖都保留最小重建包：`data.csv/json`、plot source、caption、metadata。metadata 至少有 figure ID、run IDs、source hashes、單位、reference、生成指令、資料屬性（實測／報告引用／分析估算／示意）與限制。

論文版以 SVG／向量 PDF 為主；簡報版輸出適量 PNG，預設 300 dpi 即可，不無限輸出超高解析度或大量相同版本。架構圖保留 DOT／Mermaid／Python 等可編輯來源。英文圖中文字，附繁體中文 caption 與可用於簡報的 1–2 句說明；配色、縮寫、數字精度、字型及版式全篇一致。

每張圖的字體、箭頭、legend、單位、baseline 標記與裁切都要 render 後目視檢查。不要過度密集地把所有圖擠在一頁；figure ID 與目錄名稱穩定。

所有統計圖必須能在「無 GPU、無 checkpoint、只有小型結果資料」的條件下重建。需要模型產生的新診斷資料先單獨蒐集一次並記錄來源，再由 plot 層讀取；不能每次畫圖都重新載模型跑 val。

若 raw data 已遺失，只剩既有報告或圖片，明示 `reported_only` 或 `missing_source_data`。不得讀圖估數後將其標為原始精確值，也不得捏造 error bars、多 seed 平均或完整收斂曲線。

---

## 6. 工作區結構、各資料夾自治與統一入口

優先擴充現有 `yolo_optimize` 作為**跨專案研究總入口與 orchestration layer**，沿用其 `experiments/`、`reports/`、`docs/` 的分工；但不能把其他研究資料夾的功能全部吸進來。不要再建立另一份完整 yolo clone、另一份 optimize、另一份 final 或含全部權重的 portable workspace。

每個既有頂層研究資料夾都是一個有明確責任的子專案。先由 README、程式入口、config、checkpoint 與歷史報告推回它的 original mission，再依「保留任務、更新實作」原則整理。若文件與實際 code 不一致，以可驗證的實際功能為基礎建立 discrepancy，之後再修文件。


### 6.1 各主要資料夾整理後仍必須能做什麼

以下是目前已知的責任邊界；本機若還有其他研究資料夾，使用相同方法先建立 contract，再整理，不可因未列在本表就刪掉。

| 資料夾 | 整理後必須保留的核心能力 | 可以更新／重做的內容 |
|---|---|---|
| `yolo_binaryqk/` | Binary Q/K 的 method study、branch/basis 消融、training/eval 與結果比較 | 先用舊 YOLO11／既有流程萃取方法、數據與必要設計，再重建目前 canonical implementation；migration 完成後移除舊 historical tree，不長期保留 legacy 副本。 |
| `yolo_attention/` | Attention 硬體友善化的實驗／原型能力；若有獨特方法不能被 final package 吃掉後失效 | 先參考舊 prototype，將仍需要的方法遷移到乾淨入口與共用模組；驗證後刪除被取代的舊 prototype／report／run tree，不保留整套 historical reproduction。 |
| `yolo_attention_final/` | BinaryQK＋PWL 等正式 attention package 的可攜、可驗證、可推論／訓練能力 | 可更新 package layout、CLI、checkpoint contract、tests、docs；減少內含重複大權重，但不能把「portable/final 可執行」變成只有報告。 |
| `yolo_combine/` | Detect＋Pose shared-trunk 的初始化、joint training、resume、validation、inference、task routing | 可合併舊 runner、更新 dataset registry、loss normalization、BN policy、lineage、plots；既有 Pose source 必須能正確載入。 |
| `yolo_activation/` | Activation 定義、替換、數學驗證、region/site sensitivity、recovery 與結果分析 | 可重算 manifest、修 shared-module alias、更新 qSiLU/current baseline、重畫 sensitivity；不要只留舊圖。 |
| `yolo_quantize/` | Weight／activation quantization、layer sensitivity、PTQ/QAT/候選 quantizer 的研究能力 | 本輪不必啟動完整量化，但資料夾必須保持可用；可更新成讀取目前 activation／combine 模型與統一結果格式。 |
| `yolo_achitechure/` | 模型結構變更、architecture candidate／compression 等實驗與必要的客製 module loading | 先用舊 candidate 與 loader 作 migration source；失敗結果抽成小型 negative-results／registry。若 checkpoint 仍 import 舊 module，先轉換 checkpoint 或提供暫時 migration adapter；轉換驗證完成後移除舊 module 與 failed candidate tree。 |
| `yolo_optimize/` | 將上述研究串成實際主線、做跨階段比較、最新候選、benchmark、論文圖與 cleanup/registry | 可以大量重整與更新；但它是 integrator，不是其他資料夾功能的墓地或唯一實作入口。 |
| `original/`、`configs/`、`docs/` | canonical data lineage、共用設定、全域規範與工作紀錄 | 可整理索引與契約，不重切資料、不把 runtime output 反寫成新的 canonical source。 |

若某個資料夾原本同時承擔兩種以上責任，先拆成清楚的 subcommands／subpackages，而不是直接刪掉其中一種責任。例如：`train`、`evaluate`、`analyze`、`plot`、`doctor` 可以共用底層，但仍由該資料夾提供對應入口。

### 6.2 資料夾級相容與重構規則

每次對一個資料夾做大改動前，先記錄 `before` contract；改完後做 `after` contract 與 regression。至少檢查：

```text
README 是否仍說得清楚這個資料夾是做什麼
主要 CLI / script 是否仍可啟動
config schema 是否可讀
必要 checkpoint 是否能 strict/controlled load
dataset mapping 是否仍符合 canonical registry
最小 CPU smoke 是否通過
已有結果是否能由小型 metrics 重畫
舊 import path 若被替換，是否有 adapter / migration note
active / planned run 是否會被重構破壞
```

允許內部檔案「大幅更新或重新處理」，包括重新命名、合併 scripts、將 notebook／臨時程式改成 package、重新產生報表、重新抽取小型 metrics、修正已確認的 bug、更新模型介面與統一資料格式。只要符合以下條件即可：

1. 原研究問題仍能被該資料夾回答；
2. 舊結果 provenance 沒有被洗掉；
3. 新結果不覆寫舊 run identity；
4. checkpoint／dataset／evaluator 版本差異有明示；
5. 主要入口與最小測試有更新；
6. 沒有因集中共用 code 而形成未記錄的跨資料夾脆弱依賴。

真正共用的演算法 code 可以抽到一個 canonical package，但**禁止出現「刪掉 A 資料夾的實作後，只有去 B 資料夾某個歷史 final 深層路徑 import 才能工作」**。若共享，請使用穩定公共介面，並在各子專案保留薄 wrapper／adapter 與 pinned contract test。


以下是目標邏輯結構，不要求為了名稱漂亮立刻移動所有歷史 run：

```text
yolo/
├── README.md                            # 全專案導覽、正式版本與安全操作
├── configs/datasets/                    # 既有 canonical dataset registry
├── original/                            # 受保護資料／來源，不複製、不亂刪
├── yolo_optimize/
│   ├── experiments/                     # 既有實驗與真實 run identity 優先保留
│   ├── scripts/                         # 新增薄的統一 CLI，呼叫既有可用功能
│   ├── configs/research/                # stage map／selection／retention policy
│   ├── docs/audit/                      # source、dependency、disk、checkpoint 盤點
│   ├── docs/worklogs/                   # 中文變更、驗證與風險
│   ├── reports/thesis/
│   │   ├── README.md
│   │   ├── research-story.md
│   │   ├── experiment-registry.jsonl
│   │   ├── model-registry.yaml
│   │   ├── source-map.csv
│   │   ├── discrepancies.md
│   │   ├── metrics/                     # 小型精確數據
│   │   ├── figures/                     # 少量正式圖與可重建來源
│   │   ├── captions/
│   │   ├── negative-results.md
│   │   └── pending-experiments.md
│   └── weights-index/                  # 只有 path/hash/role，不複製整份權重
└── 原研究目錄                           # 保留資料夾使命；內容完成 migration 後以新 canonical 版本取代舊內容
```

source-map 將邏輯 `00_baseline`、`01_attention`、`02_masf`、`03_hog`、`04_combine`、`05_activation`、`06_repconv` 映射至實際目錄。報告排序可以改，run ID 與歷史 parent 不能改。

真正共用的底層程式可以只保留一份已驗證的 authoritative implementation，但各研究資料夾必須保留自己的穩定入口與 contract tests。整理之前先確認各份同名模組是否真的相同、是否具有不同研究語意。必要時可暫時使用 adapter／re-export 完成 migration，但它不是永久相容層：客製 `.pt` 若序列化 Python 類別，先成功載入、轉換為新 canonical checkpoint/state-dict、完成 strict load 與 forward 驗證，再移除舊 import path 與舊 module。**不可為了去重而讓某個資料夾失去原本可執行的研究能力，也不可為了相容而把整套舊實作永久留下。**

不要建巨型通用 framework。優先讓現有 CLI 透過薄 wrapper 提供以下能力：audit、collect-results、validate、plot、report、cleanup-plan、cleanup-apply。若新增入口，例如 `python -m <package>.research ...`，交付時必須給真正可執行的命令，不能把本文件示意名稱當成已存在命令。

所有舊入口、config／import／相對路徑、checkpoint loader、README links 都需回歸測試後，才移除被替代的程式。不要只更新 Markdown 連結，忽略執行中的腳本仍依賴舊路徑。

---

## 6.3 原內容的最終處理規則：Reference → Migrate → Verify → Remove

這是本次整理新增的**強制終態**。

對每個研究資料夾，舊內容只作為 migration source。完成下列條件後，舊內容要移除，不要留下另一份歷史副本：

```text
R1  Reference
    已讀取舊 code/config/report/result/checkpoint metadata，列出哪些資訊值得保留。

R2  Extract
    已把必要資訊轉成新的 canonical code、config、registry、metrics、figures source、method note 或 checkpoint。

R3  Map
    `migration-map.csv/json` 能回答每一個被採用的舊資產「搬到哪裡／轉成什麼」。

R4  Verify
    新版本至少通過 import/config、CPU smoke、checkpoint load、主要數學 contract、
    existing metric reproduction 或 figure rebuild 中與該資料夾相符的驗證。

R5  Dependency check
    `rg`／import trace／entrypoint／active process 檢查確認沒有新程式仍引用舊路徑。

R6  Remove
    移除已被取代的舊 code、舊 reports、舊 plots、舊 logs、舊 run tree、
    重複 checkpoint 與 migration staging。

R7  Audit
    產生 `removed-original-inventory.csv/json`，記錄被移除項目、原 hash／size、
    replacement、驗證結果與刪除時間；不要保留被移除檔案本體。
```

最終 workspace **不要**出現：

```text
legacy/
archive/
old/
backup/
final_old/
final2/
migration_backup/
```

裡面又完整塞著舊專案的情況。若這些目錄只是 migration 過程暫時使用，完成後要刪除。

歷史可追溯性改用**小型資料**保存，例如：

```text
migration-map.csv
removed-original-inventory.csv
experiment-registry.jsonl
checkpoint-manifest.json
metrics/*.csv
negative-results.md
method-history.md
source-map.csv
```

也就是「保留歷史資訊，不保留歷史副本」。

唯一不能直接套用 `Remove` 的，是：

- canonical dataset／labels／dataset lineage；
- active run 或明確要 resume 的 checkpoint；
- 尚未成功轉換且不可重建的唯一關鍵 source weight；
- 法規／授權／研究完整性要求必須保留的原始資料。

這些應列入 `protected-until-migrated`，而不是混入一般 legacy archive。等替代方案完成且使用者核准清理計畫後，再移除舊資產。

---

## 7. 省空間的核心：最小權重集合＋完整小型證據

我的電腦空間不足，不要為了「整理」再複製數十 GB。原 optimize 的「所有 checkpoint 都保留原位」是歷史政策；本輪要提出安全精簡方案，但不能把整個 `artifacts/` 視為垃圾。

### 7.1 優先保留小而重要的東西

永久保留 code、configs、環境版本、dataset／checkpoint manifests、run lineage、精確 CSV／JSON、必要的 per-epoch 指標、failure／cancellation 原因、採用規則、關鍵診斷摘要、plot source 與正式圖。

失敗結果仍值得保留，但不代表每個失敗 run 的所有 optimizer snapshots 都要保留。先抽取必要證據，再評估大檔的復現需求。

### 7.2 權重保留按角色，不按資料夾一律留 best／last

建立受保護根集合與依賴圖，至少保護：

```text
已訓練好的獨立 Detect／Pose 基準（本機唯一或不能確認可恢復者）
目前正式選用模型及必要架構／載入程式
確定要接續的研究 parent 與必要 rollback
目前 active／paused 且需續訓 run 的最新完整 resume
論文關鍵比較中，仍需重新驗證／診斷且無等價保留品的 inference weights
共享初始化／checkpoint 重建仍真正依賴的來源權重
```

同一 SHA 可以對應多個邏輯角色，不為每個角色複製一份。保留數目由上述用途決定，不硬限全專案只留 3 或 5 個，導致必要 Pose／baseline 遺失。

active run 預設保留一份最新完整 resume、一份必要 rollback，加上真正需要的精簡 inference best；更新採原子寫入與驗證後輪替。暫停且會續訓者同樣保留 resume。

已完成 run 的 optimizer／EMA／scheduler／scaler 不能直接全部剝掉；先確認不需要續訓，且 selected inference 重建、重要狀態及 metric provenance 已保存，再列入待刪清單。folded inference 不能取代 RepConv 多分支 resume。

「整個檔案 SHA 相同」才能直接認定 byte-identical duplicate。只有參數看似相同、但 EMA／BN／optimizer／metadata 不同的兩份檔案，不視為自動可刪的重複。

對每個歷史 run 標明保留層級：

```text
full_resume       = 可以從保留狀態續訓
inference_recheck = 可以重新推論／驗證，沒有完整 optimizer
replot_only       = 保留小型原始指標與作圖來源，不能立即重驗 checkpoint
reported_only     = 只有文件記載，缺乏更完整證據
```

`replot_only` 能重畫圖，不等於保有完整模型可重驗性。降級到這個層級前要列明影響，交給使用者確認。

### 7.3 可提出精簡的候選

優先找 byte-identical copies、已解壓且完全重複的大 archive、已被精簡 inference 取代且不再續訓的中間 snapshots、重複 exports、過量 preview 圖、可重建 cache、重複 TensorBoard／console logs 及未被引用的大型中間 tensors。

不可因為名稱是 `old`、`tmp`、`archive`、`final`、`artifacts`、`last.pt` 就判定可刪。資料集、標註、唯一來源權重、active run、尚需診斷的唯一 evidence 一律不是自動刪除候選。

不要把大檔搬到同一顆磁碟的 archive 目錄就宣稱釋放空間；不要壓成另一份巨大 ZIP／tar 先吃掉剩餘空間。Git checkout、copy、hardlink／symlink 都需考慮相容與可變性；不對仍會寫入的 checkpoint 共用 hardlink。

### 7.4 清理工具必須安全預設

先輸出 `cleanup-plan.csv/json` 與可讀報告，每個候選包含：

```text
path、allocated_bytes、logical_size、SHA256、mtime、run_id
artifact_role、reason、canonical_replacement、reference_count
protected／active／open_file_status、recovery_method
reproducibility_before／after、risk、expected_released_bytes
```

預設只有 dry-run。正式 apply 必須指定已檢閱的 plan ID／hash 與明確確認，且再次檢查檔案大小、mtime／hash、目前引用與 active process。未批准項目不刪。

限制所有操作在核准的 workspace roots 內，解析 realpath，拒絕 path traversal、意外 symlink target、外部資料目錄及不明權限。不要追著 symlink 刪到 canonical dataset。

計算空間時區分 logical size 與實際 allocated bytes，處理 hardlink 重複計算與被程序開啟的檔案。刪除前後量測可用空間與 retained manifest；不能只把檔案大小相加就宣稱真正釋放同樣容量。

checkpoint 刪除前，先完成必要的 loader／strict state／forward 等價與原始指標留存檢查；若唯一關鍵 checkpoint 還沒有可用替代，刪除閘必須拒絕。

本次任務的**目標終態已明確要求移除完成 migration 的舊內容**，所以 Codex 不要再把「保留 legacy/archive」當預設策略。不過實際刪除仍採兩階段：先產生可稽核的 cleanup plan；對已完成 replacement、dependency check、驗證且屬明確舊來源的項目可依本任務執行移除。對 active run、唯一不可重建權重、canonical dataset／labels、或用途仍不明的資產，仍需列為 blocked/protected，不得猜測刪除。

### 7.5 預防再長大

提出集中 retention policy：關閉不必要的每 epoch 完整快照與大量 val image／prediction 保存；保留精簡 CSV／JSON 與少量固定樣例。實作前確認現有 trainer 的 save／resume 行為，不能關掉唯一恢復點。

原始 metrics 定期 flush，小型狀態原子寫入；先估算一次 checkpoint 寫入、驗證暫存及保留 rollback 的峰值空間，再決定最低 free-space watermark。磁碟不足時停止新增寫入／新工作並清楚回報，不自動清空必要權重。

Git 只追蹤 code／configs／小型結果／必要論文圖。不要 blanket-ignore 所有 PDF 而連向量論文圖都忽略；也不要把 checkpoints、環境、dataset views 或大量推論影像上傳遠端。既有 tracked 大檔的 index 清理與 Git history 清理另列，不能擅自 force-push。

---

## 8. 驗證與計算預算

第一輪以 CPU、已有文件與 metrics 完成，不占用正在訓練的 GPU。既有環境可用就重用，不為每個階段再建立一份大型 venv，不自動升級 Ultralytics／PyTorch 改變歷史行為。

至少執行並記錄：

```text
語法／import／config schema 與必要既有 tests
實際載入模組來源與重複 package 檢查
checkpoint role、class／keypoint／shape、strict state 對應
兩處 Attention 的 shape、精確前向與 Q/K gradient contract
PWL table／range／backend 與 row-normalization 契約
MASF placement、alpha=0／Identity、bridge、BN freeze 與 head 輸入契約
combine 的 task routing／shared forward 次數、loss normalization 契約
activation alias／site replacement 與 qSiLU 分段／尾端測試
RepConv 初始化及非零 trained branches folding／reload 等價
結果單位／class mapping／null／finite values／gate reference 檢查
無權重模式下重建圖表、文件 links 及圖檔目視驗收
cleanup dry-run 的 protected／active／外部路徑拒絕測試
```

toy／small-input smoke 只是功能測試，不能取代完整 COCO5000／BBAT683 的正式精度。若模型的相對 bias 與輸入大小有固定契約，不能為省資源任意改 imgsz 後宣稱等價。

缺少正式驗證或新對照時，提出最小補驗清單，列 parent、目的、唯一變因、dataset、estimated compute、最大磁碟增量、保存策略、既定 selection rule 與停止條件。沒有核准預算就不自動啟動。不要用新跑實驗覆寫歷史結果；新增 run ID。

---

## 9. 分階段完成，不只停在規劃

### Phase A：盤點與研究對照

完成 source／disk／checkpoint／dependency audit，找出正在使用的 Pose 與正式模型，建立初始 registry、敘事 map、真實 lineage 與 discrepancy 清單。找不到的資產明示，不猜路徑。

### Phase B：可安全進行的整理、程式與圖表

延用現有 optimize 結構建立統一入口，匯入已有 metrics，完成有資料支持的正式圖、中文研究文稿、負結果索引與最小 retention 工具。同時逐一整理各既有研究資料夾：先寫 folder contract，再更新／重構內部內容，確保它仍能完成原本的核心工作。必要 refactor 小步執行，保留相容入口，完成 CPU regression。不要因等待刪檔確認就把這些工作全停住，也不要只整理 `yolo_optimize` 而放任其他資料夾變成不可執行的歷史殼。

### Phase C：交付確認清單

集中提出 checkpoint／大檔刪除候選、可釋放容量、會失去的復現能力、必須保留的權重，以及真正必要的補驗／訓練預算。正式模型有衝突時提出證據與選項，不擅自挑一個。

### Phase D：得到核准後才執行

依已核准 plan apply 清理；依已核准 compute budget 補驗。每步保存小型工作紀錄與結果，不再製造重複大備份。未被核准的操作維持 pending。

---

## 10. 最終交付與驗收標準

請實際產出並提供位置：

1. 全專案 README／研究閱讀入口、`research-story.md`、各階段方法／假說／結果／限制，以及**每個既有頂層研究資料夾的 folder contract、README、最小可執行 smoke／doctor 入口**。
2. experiment／model registries、source map、原始數據來源、敘事順序與真實 lineage。
3. 新版 MASF 定義、Detect／Pose placement 與分開驗證狀態；既有 Pose 權重索引與載入方式。
4. 完整小型 metrics、論文圖與 captions、可在無權重狀態重畫的命令。
5. 現在到底採用什麼的唯一說明，列 BinaryQK／PWL／MASF／HOG／Activation／RepConv 各自狀態，不把研究候選混成正式模型。
6. 負結果、取消／中止實驗、未完成對照與 discrepancy 記錄。
7. Disk audit、受保護 checkpoint manifest、retention policy、cleanup dry-run 與 apply 安全測試。
8. 中文 worklog，包含變更原因、實際測試、困難與解法、未解風險；沒有困難也記「無」。

驗收時必須回答：

- 從首頁能否按論文順序理解所有階段，同時不誤讀真實訓練血緣？
- 每張結果圖能否追到 data／run／checkpoint／config／dataset／evaluator？
- 不重新訓練、不載入 checkpoint，能否重畫已有統計圖？
- 既有 Pose 與正式模型是否仍可由保留程式正確載入？
- 每個既有研究資料夾是否仍能完成它原本的核心研究工作，而不是只剩連結、報告或被 `yolo_optimize` 取代？
- 新 MASF、16-slot scale／bias、trunk ratio／loss weights、deployment folding 是否都沒有混淆？
- 清理會刪哪些、保留哪些、真的釋放多少，以及失去哪一級復現能力，是否寫清楚？
- 未通過 gate、缺 raw data、沒有硬體實測的項目，是否如實標示？

最後回報「已完成」「尚待資料／決策」「尚待 GPU／全量驗證」「已核准並實際刪除／目前僅 dry-run」四類狀態。不要把規劃、檔案產生、CPU smoke、正式驗證與新訓練混成同一種完成。

**整體原則：留下少量真正必要的權重，留下完整而小型的研究證據；每個資料夾保留原本的研究職責，但內部實作可以大幅更新、重構與重新處理。讓資料夾更精簡、更能執行，而不是把功能抽空；同時不要讓論文失去可追溯性。**
