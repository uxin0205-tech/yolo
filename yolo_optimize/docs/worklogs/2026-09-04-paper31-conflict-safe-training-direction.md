# 2026-09-04 論文第 3.1 節與衝突安全訓練方向整理

## 變更內容與原因

- 完整閱讀使用者提供論文第3.1.1–3.1.4節，分開記錄模型預算、LR短篩、多階段transfer與
  WST/HOG/Canny的原始做法。
- 建立研究報告
  [`2026-09-04-msfa-section31-training-transfer-innovation.md`](<../research/2026-09-04-msfa-section31-training-transfer-innovation.md>)，
  並以原論文、官方MSFA程式、PCGrad、GradNorm、distillation等第一手來源核對可轉用邊界。
- 稽核Full35 J1/J2/J3 durable gradient logs。負cosine率為32.258%／32.759%／41.176%；J2/J3
  Pose／Detect norm ratio median已低於1，故首輪優先處理方向衝突，而不是再掃LR或只調loss weight。
- 建立[`OPT-TRAIN-CONFLICT-SAFE`](<../../proposals/training-conflict-safe/README.md>)資料夾，包含方向說明、
  [最小計畫](<../../proposals/training-conflict-safe/plan.md>)與
  [終端架構圖](<../../proposals/training-conflict-safe/architecture-report.md>)。
- 首輪固定為`G0-MATCH`／`G1-APC-DETECT`兩臂；只有`g_detect·g_pose<0`時投影Pose shared gradient，
  Detect與兩個task heads保持原樣。
- 將此方向排在person-only head決選後：舊logs是COCO80 `g_detect`，不得冒稱person gradient；新head需重新screen。

## 驗證方式與結果

- 逐段複核研究報告562行及公式、資料入口、source links、實驗矩陣與停止條件。
- 唯讀核對現行`MacroStepEngine.run()`的順序：Detect snapshot、Pose backward、AMP unscale、finite check、
  global clip與single optimizer step；確認存在可插入projection的工程seam。
- 核對歷史負事件`|cosine|`median約0.027935／0.039848／0.041183，支持文件明載「典型修正量不大、
  值得測但不能宣稱必定補回AP」。
- 全目錄38份Markdown的本地連結與code-fence稽核通過：missing links 0、unbalanced fences 0；尾隨空白0。
- 以正式`parent-j2/logs/events.jsonl`加J3 `logs/events.jsonl`重算，J1/J2/J3的n、負事件率、norm ratio與
  負事件幅度全部對上報告。合成向量亦通過負dot歸零、Detect不變與正dot不介入測試。
- `/home/uxin/yolo/.venv/bin/python scripts/audit_repconv_seams.py`通過；
  `scripts/audit_accuracy_regressions.py`仍依設計exit 1並列出既有4個MASF/BinaryQK紅燈，沒有新增項目。
- 沒有啟動GPU training、validation或外部queue；沒有修改production trainer、checkpoint、dataset或label。

## 困難與解法

- 困難：指定PDF禁止一般copy。解法：使用本機`pdftotext`唯讀擷取並逐段對回3.1節，沒有修改原PDF。
- 困難：論文的SAR/DOTA脈絡與本地RGB joint任務不同。解法：只保留「先量測轉移shock、最小介入」原則，
  不搬DOTA、固定LR或永久filter input。
- 困難：初次只讀最終J3 log只能看到J3事件，無法驗證J1/J2。解法：依final package lineage合併唯讀
  `parent-j2`與J3兩份durable logs後重算，沒有修改任何log。
- 其餘無。

## 未解事項與風險

- 非對稱投影尚未實作或訓練，AP收益未知；AdamW/momentum下也沒有嚴格的loss下降保證。
- person head改變後，歷史衝突率失效；必須依新baseline的trigger重新決定是否執行。
- 若post cosine修正但AP不變，應停止本方向；dual-teacher anchor與Sobel/HOG companion只能另案。
- Full35正式證據目前主要是seed 0；升格仍需三組paired seeds。
