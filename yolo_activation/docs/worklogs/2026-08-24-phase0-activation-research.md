# 2026-08-24：Phase 0 activation 文獻與數學設計

## 任務與範圍

依使用者確認啟動 activation Phase 0，閱讀既有 `activation.pdf` 與
`docs/Activation_research.md`，以 primary sources 盤點候選與 prior art，縮減候選數，並提出一個
可再討論的數學／架構型 working hypothesis。本輪只做研究、數學 sanity check 與實驗 protocol；
沒有模型程式、checkpoint、訓練、正式資料 profiling 或硬體 benchmark。

使用者補充 detection 資料同時包含：

- Canonical BBAT5 v1 ball/bat Detect Task View：
  `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml`，固定 formal train/val。
- COCO2017 COCO80 Detect：`/home/uxin/yolo/coco2017.yaml`，既有 train2017/val2017 manifest。

兩者只分開設計評估，沒有讀寫影像／labels、建立 runtime view、重切、抽樣、合併或啟動訓練。

## 變更內容與原因

- 新增 `docs/research/phase0-activation-landscape.md`：整理 activation primary-source 證據、
  shortlist、重訓需求、部署風險，以及 ActNAS、GRAU、DAPA、Curl、xIELU 等直接碰撞。
- 新增 `docs/research/phase0-mathematical-design.md`：定義導數互補、積分約束、exact-ReLU tail 的
  proposed integral-polynomial family，列出 Lite／Shape profiles、純函數 sanity check、雙資料集
  protocol、階層式逐層搜尋與 promotion gate。
- 新增 `docs/research/README.md` 與根層 `README.md`，提供研究、資料集規範與工作紀錄入口。
- 在 `docs/Activation_research.md` 加入 2026-08-24 決策覆寫，避免舊規格誤把 YOLO26m、單一
  PWL 路線或三 seeds 當成本輪已確認條件。
- 新增本工作紀錄與索引。

## 驗證方式與結果

1. 唯讀核對 `/home/uxin/yolo/AGENTS.md`、`CONTEXT.md`、
   `docs/agents/bbat5-datasets.md`、`/home/uxin/yolo/coco2017.yaml` 與完整 activation 規格。
2. primary-source 研究確認：ActNAS 已做 YOLO 逐層 mixed activation；Curl／公開專利申請已利用
   SiLU even residual；GRAU、DAPA、PWLU 與 polynomial 方法覆蓋多個相鄰組件。因此文件沒有使用
   `novel`、`first`、`SOTA` 或 `proven hardware-efficient` 作結論。
3. 使用 `/home/uxin/yolo/.venv/bin/python`、NumPy `float64`，在 `[-12,12]` 的 24,001 點均勻
   網格檢查公式。Shape profile `a=4,T=109/16` 對 exact SiLU 的 MSE 為 `0.000085719`、MAE
   `0.00653458`、max error `0.0210018`；這只記為 curve sanity check，不是 detector evidence。
4. 符號條件核對 `q(0)=0`、`q(1)=1/2`、`q'(1)=0`、`integral(q)=1/2`；並修正
   `H(u)` 必須包含 `(u-T)_+/2`，否則 clip 後無法精確接 ReLU tail。
5. 以本地 link validator 掃描根 README 與 `docs/**/*.md`，結果為 `local_links OK`；禁用主張搜尋只命中明確的否定／禁止語境。
6. 最終 NumPy assertions 輸出 `constraints=OK symmetry=OK tails=OK`，並重現 Shape profile 的 MSE `0.000085719`、MAE `0.00653458`、max error `0.0210018`。
7. `git status --short -- yolo_activation` 只顯示整個 `yolo_activation/` 為既有未追蹤子專案；本輪沒有修改工作區其他子專案或資料集。

## 困難與解法

- 執行環境多次出現 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`：所有必要
  shell 操作改成經核准的唯讀／工作區命令，未繞過資料與權限邊界。
- 系統沒有 `python` 指令：改用 repo 既有 `/home/uxin/yolo/.venv/bin/python`。
- 虛擬環境沒有 SciPy：二參數 toy search 改用固定、可重現的 NumPy grid search。
- `apply_patch` 後續更新受到同一 bwrap helper 故障阻擋：先重試；仍失敗後以核准的精確
  `perl` 文字替換完成相同小範圍修改，並逐段唯讀核對。新文件最初仍由 `apply_patch` 建立。
- 部分論文 PDF 回傳 403／429：改用原作者 arXiv、CVF HTML、PMLR、ePrint、官方 operator
  文件等 primary-source 入口；未把未取得全文的細節推定為事實。

## 未解事項或風險

- 使用者尚未交付 baseline 資料夾，實際模型、checkpoint、eligible activation、訓練 recipe 與
  兩資料集 checkpoint 關係未知。
- 目標板卡、compiler、precision 與功耗／latency budget 尚未確定，現階段只能報 operator proxy。
- Proposed family 尚未實作 autograd/export/toy recovery，`a,T` 可行範圍、量化誤差與 detection AP
  未驗證；新穎性檢索也不等同法律或專利意見。
- 只有 1 seed、最多 2 seeds 的資源規劃不足以支持強統計結論。
- 本輪沒有任何 accuracy、COCO、BBAT5 或硬體實驗結果；純函數誤差不得誤寫成模型增益。
