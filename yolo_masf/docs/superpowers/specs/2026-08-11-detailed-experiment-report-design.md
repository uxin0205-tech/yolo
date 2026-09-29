# MASF-YOLO 詳細實驗報告設計

日期：2026-08-11

## 目標

把 `EXPERIMENT_RESULTS_ZH.md` 擴寫成可獨立閱讀、可由 artifacts 追溯的正式中文報告。報告必須逐一說明十個已完成模型的實驗做法，並從準確率、Ball/Bat 操作性指標、理論成本與 RTX 5090 實測延遲詳細比較。

本次只修改文件，不改模型、權重、訓練設定、資料集或 GPU 排程。

## 文件邊界

- `EXPERIMENT_RESULTS_ZH.md` 是主要的人讀中文報告，承載完整方法、比較與結論。
- `artifacts/static-phase1/report.md` 是 pipeline 產生的原始報告，不手動修改，避免重跑時覆蓋人工內容。
- 正式數字以 `artifacts/static-phase1/evaluation/`、`profiles/`、`selection.json`、`final_audit.json` 與各訓練階段的 `resolved_args.json` 為證據。
- 不新增 per-model 報告，以免相同口徑分散在十份文件中。

## 報告結構

1. 實驗狀態、資料集、資料暴露警告與指標口徑。
2. 所有實驗共用的初始化、訓練、validation/test 與 profiling 流程。
3. B0、B1、M7、M0、M1、M2、M3、P3M、SP2、SP2P 的逐項方法與結果判讀。
4. 完整 validation/test、操作性指標與硬體指標表格。
5. 四組控制變因比較：kernel 組合、partial channels、模組位置、Selective P2。
6. B0 為何顯著較好，以及是否應「調整回 B0」的決策。
7. 限制、部署建議與下一步公平對照。

## 每個實驗必列欄位

每個模型段落都必須能回答：

- 模型改了什麼，以及 prediction scales/stride。
- 初始化 checkpoint 或父模型為何。
- 哪些張量可轉移、哪些模組為新建或重新學習。
- 是否採兩階段訓練；各階段的 freeze、epochs 與初始 learning rate。
- validation 與 test 的 overall mAP50–95、Ball AP、Bat AP。
- 相對公平基準 B1 的準確率、false positives、GFLOPs、參數與 latency 差異。
- 能支持的結論與不能支持的結論。

## B0 判讀與決策

B0 直接評估 `yolo11m_bat_detect_init.pt`。該 checkpoint 已接觸 BBT5，具有成熟的三尺度 P3/P4/P5 偵測權重，且未接受本次相同預算重訓。B1 則由 repository-owned 四尺度 P2 template 建模，只轉移形狀與語意相容的張量；新增的 P2 路徑、neck/head 專屬張量需要重新學習，之後才執行凍結 10 epochs 與全解凍 90 epochs。

因此 B0 與 B1 同時存在架構、初始化完整度與訓練歷史差異。B0 test mAP50–95 比 B1 高 0.040484，只能說明目前 B0 是較佳的 operational checkpoint，不能證明三尺度架構在公平條件下優於 P2。

本次不覆蓋或回退 B1/MASF 實驗：

- 部署選型可保留 B0，因為它目前的 test accuracy、false positives 與 latency 都優於 B1。
- 架構研究仍以同一 B1-B 父權重、相同 100-epoch 預算的變體互相比較。
- 真正的 B0/B1 因果對照應另建乾淨、未接觸 BBT5 的 initializer，讓三尺度與四尺度模型使用相同 split、augmentation、optimizer、10+90 epochs 與 seeds 訓練。

## 比較原則

- B0 永遠標為 data-exposed operational reference，不參與公平勝者排序。
- B1 是 M7/M0/M1/M2/M3/P3M/SP2 的主要公平參考。
- SP2P 額外繼承已訓練的 SP2-B 與 validation 選出的 M2，必須標示序列式較高訓練預算。
- M2/M3 只能依 validation 選擇；即使 M3 test 稍高，也不得事後改選。
- 單一 seed 的差距不能寫成已證實的泛化改善。
- AP 與固定推論門檻下的 precision/recall/false positives 分開解讀。
- GFLOPs/params 下降不等於 RTX 5090 latency 下降。

## 驗證與發佈

- 以程式從原始 JSON 重新抽取報告使用的核心數值，核對表格與差值。
- 檢查十個模型段落、共同方法、B0 專節、四組比較與證據路徑皆存在。
- 執行 Markdown 路徑與格式檢查，以及 repo 既有測試；文件修改不得造成測試回歸。
- Git 只 stage 本次 `yolo_masf` 內的報告、規格與實作計畫，不納入上層其他專案或資料集變更。
- 依使用者既有要求直接推送至 GitHub `main`，不建立 PR。
