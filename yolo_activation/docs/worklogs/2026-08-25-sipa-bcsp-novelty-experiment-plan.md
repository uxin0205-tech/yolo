# 2026-08-25：SIPA–BCSP 新穎性稽核與完整實驗計畫

## 任務與範圍

依使用者要求，直接回答 SIPA／BCSP 在學術上可能新穎的部分，並把完整模型、統計、量化與上板
實驗整理成可執行前預註冊計畫。本輪屬研究與文件工作；沒有修改訓練程式、沒有收到或載入
baseline folder、沒有啟動 COCO2017／BBAT5 訓練，也沒有產生 AP 或硬體實測。

資料使用狀態：沒有讀寫影像或 labels、沒有建立 Runtime Dataset View、沒有重新切分／抽樣／
混合資料。文件維持 Canonical BBAT5 v1 Detect 與 COCO80 Detect 的獨立契約。

## 變更內容與原因

### 1. Primary-source 新穎性 claim chart

- 新增 `docs/research/sipa-bcsp-novelty-audit.md`，逐項稽核 SiLU symmetry、exact-ReLU tails、
  C2 polynomial、積分設計、dyadic/APoT、fixed-point rounding、YOLO placement、Pareto 與 beam。
- 研究只採原始論文、官方 proceedings／作者程式碼，並把「來源直接陳述」與「由公開公式推論」
  分開標示。
- 稽核確認 Curl 已直接使用 SiLU even residual；SmeLU 已有 polynomial exact tails；S-ReLU 已有
  C2 piecewise-polynomial exact tails 且可由公式推出相同差分恆等式；ActNAS 已直接覆蓋 YOLO
  activation placement 與 hardware-aware search。
- 因此收窄候選學術主張為：zero anchor、SiLU 式 non-monotone negative valley、受約束一參數
  integral-polynomial family、finite C2 exact tails、dyadic instance，以及 invariant-preserving
  integer realization 的完整組合。這仍是候選，不是已證明的新穎性。
- BCSP 目前定位為節省 trial 並維持公平性的實驗 protocol；除非同搜尋預算比較顯示穩定 frontier
  優勢，不列獨立演算法 contribution。

### 2. 完整實驗計畫

- 新增 `docs/research/sipa-bcsp-experiment-plan.md`，定義 C1–C3 候選主張、RQ1–RQ6 研究問題、
  E0–E10 實驗漏斗、反證條件、控制變因、量測與停止條件。
- 第一輪模型候選維持精簡：SiLU、Hardswish、ReLU、`poly_shift`、`poly_quality`；SmeLU、S-ReLU
  與 matched-cost PWL／GRAU-style control 只在 claim-critical ablation 加入，不重開 activation zoo。
- 補上 stage-matched SiLU sham-recovery：short／full candidate 必須與同 seed、同 training budget 的
  SiLU 比較，不能把多訓練本身造成的收益算成 activation 效果。
- 規劃每資料集 mixed-policy search 最多八個 jobs；現有 planner 只有每輪 `beam_width=4`，未補
  global cap 前不得啟動昂貴真實搜尋。
- 定義 paired image-level bootstrap 只估 evaluation-set uncertainty，不冒充 training-seed
  variance；seed 1 為探索結果，資源允許才跑 seed 2，不能據此過度宣稱穩定性。
- 最壞上限為每資料集 29 jobs、兩資料集 58 jobs；其中 seed 2 不跑時每資料集 26。Ablation、
  PTQ/QAT 與上板另立 job manifest，不含在 58 內。
- 跨資料集研究分為 D0 independent、D1 shared-profile、D2 shared-policy；D2 才使用 frozen
  worst-case normalized regret，不平均 COCO 與 BBAT5 raw AP。

### 3. 文件同步

- 更新根層 `README.md`、`docs/research/README.md` 與 `training/README.md`，加入兩份研究文件入口。
- 本工作紀錄加入 `docs/worklogs/README.md` 索引。

## 驗證方式與結果

1. 完整閱讀 409 行新穎性稽核，逐項核對其 claim／不可 claim、來源角色及現行實作邊界。
2. 最終本地 Markdown link validator 掃描 14 個 Markdown、31 個 local links，結果 `broken=0`。
3. 研究文件 trailing-whitespace 掃描只找到標題 metadata 的兩空格 Markdown hard breaks；無意外
   trailing whitespace。
4. 快取 inventory 沒有 `.pytest_cache`、`.ruff_cache` 或 `__pycache__`。
5. 本輪沒有程式行為變更，因此沒有把既有 toy dry-run 冒充新模型驗證，也沒有重跑／聲稱新的
   detection tests；上一輪程式驗證仍是 26 passed。

## 困難與解法

- Prior art 比原先假設更接近：S-ReLU 已同時碰到 C2、polynomial、exact tails 與可推導的差分
  恆等式。解法是放棄寬泛「對稱平滑多項式」敘事，改成更窄、需要 M0–M6 反證的組合主張。
- 背景研究初稿的 fixed-point lemma 少一個 LaTeX `\\left`：完整閱讀時發現並修正，公式現為
  `floor(n/2) - floor(-n/2) = n`。
- Sandbox 持續出現 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`；新文件以
  `apply_patch` 建立。更新既有 README／研究文件時先嘗試 `apply_patch`，失敗後才使用精確、完全
  匹配的 `perl -0pi` 小範圍替換，並立即唯讀核對。

## 未解事項或風險

- 本次不是法律意義的專利／新穎性檢索；候選主張仍需更廣引用鏈與專利檢索。
- 最小 polynomial degree、可行參數範圍、負谷／導數 extrema 尚未形式化證明。
- 尚未實作 SmeLU／S-ReLU control、stage-matched SiLU job、global search cap、empirical profiler、
  paired bootstrap、robust shared-policy objective 與 production model adapter。
- baseline folder 與 target profile 未到，沒有 detection AP、training variance、latency、area、power
  或板上證據；目前不得宣稱新穎性成立、SOTA 或 hardware-efficient。
- 1–2 seeds 只足以支持探索性結論；若未來要做穩定優越或正式論文主張，需另行增加獨立 runs。
