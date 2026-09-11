# 歷史入口快照（非目前狀態）

原路徑：`README.md`。原始位元組另存同名 `.txt`；整理日期 2026-09-12。

# YOLO Optimize

## GitHub 統整發布（2026-09-12）

從 BinaryQK 開始的完整優化閱讀順序見 5090 Done 0912 報告入口（本機／歷史參照：`../../../reports/5090-done-0912/README.md`；未隨本次報告發布），並保留全階段詳細總報告（本機／歷史參照：`../../../reports/consolidated-20260911/README.md`；未隨本次報告發布）。本次只發布報告、指標、設定與研究程式；checkpoint bytes、封存副本與資料集影像留在本機。實驗結果不變，沒有新增訓練。

## 最新狀態（2026-09-11）

全部階段已整理為詳細總報告（本機／歷史參照：`../../../reports/consolidated-20260911/README.md`；未隨本次報告發布），含早期融合後、融合前方向 1、重新 combine、activation、KD、推論、超參數與權重血緣。主封存完成 5,817 檔／113.66 GB，含 406 個 checkpoint／模型產物，逐檔 SHA256 相同；最新補充結果與本機 Ultralytics 原始碼另存補充包。見checkpoint 索引（本機／歷史參照：`../../../reports/consolidated-20260911/checkpoints.csv`；未隨本次報告發布）、本機封存（本機／歷史參照：`../../../archives/README.md`；未隨本次報告發布）。

新 one2many＋NMS 推論驗證已完成：bat 明顯提升、ball 退化，不全面採用；原 qSiLU E2 仍為預設，原權重全部保留。沒有新的 GPU queue；class routing／共享 Conv 仍未執行。見推論附錄（本機／歷史參照：`../../../inference/routing_v1/README.md`；未隨本次報告發布）與[工作紀錄](<../../worklogs/2026-09-11-comprehensive-archive-and-routing.md>)。

## 以下是各階段歷史進度，非目前執行狀態

最新覆蓋：Pose-head KD 五輪與本次推論分支重組完整驗證（本機／歷史參照：`../../../inference/pose_branch_v1/README.md`；未隨本次報告發布）均已完成，目前無本研究 GPU queue。保留原框＋KD E2 關鍵點可精確恢復框 AP，但 Pose AP 無提升，候選不取代原 qSiLU E2。下一步研究 one2many／NMS 路徑與實際顯示處理，尚未啟動新訓練。下方背景執行敘述為歷史。見[工作紀錄](<../../worklogs/2026-09-11-inference-branch-research.md>)。

最新覆蓋：使用者取消原生對照，直接跑 Pose head KD 背景 queue（本機／歷史參照：`../../../kd/pose_focus_v1/README.md`；未隨本次報告發布），成功後自動整理結果。排好後停止模型working，不反覆續接等待；共享特徵修正仍待KD結果決定。取消的native產物保留，詳見[工作紀錄](<../../worklogs/2026-09-11-pose-direct-background.md>)。下方配對啟動敘述為歷史。

使用者已核准 Pose head 專項（本機／歷史參照：`../../../kd/pose_focus_v1/README.md`；未隨本次報告發布） 並允許安排共享特徵試驗。原生／head KD 前置全部通過，Detect輸出與非Pose state精確不變；接續兩臂各5輪，AdamW LR1e-5／warmup1／batch16。共享層小範圍對照另開後續分支，不混入本輪。以下「僅規劃未啟動」為核准前歷史。

K0／雙教師 KD 各5輪已完成，沒有新的合格 best_joint。依使用者要求，下一步優先規劃「凍結共享層與 Detect，只更新完整 Pose head」，見 Pose-only 決策與驗證計畫（本機／歷史參照：`../../../kd/dual_task_v1/POSE_ONLY_NEXT_PLAN.md`；未隨本次報告發布）。本次只分析與規劃，未啟動新 GPU job；以下配對啟動敘述為歷史。

雙教師已獲核准，YOLO26L Detect／原 BBAT5 Pose 教師完整驗證均優於學生。KD 校準與兩臂真實 smoke 已通過，接續 K0／双教師空間 KD 五輪配對（本機／歷史參照：`../../../kd/dual_task_v1/PLAN.md`；未隨本次報告發布），採 AdamW；MuSGD 新前置未過校準，不強行採用。見[最新工作紀錄](<../../worklogs/2026-09-11-dual-teacher-musgd-start.md>)。下段教師待決策敘述屬前一階段歷史。

Activation 四臂零樣本、SiLU／qSiLU 各 10 epoch 配對與選定 qSiLU E2 匯出驗證已完成，見 Activation 結果（本機／歷史參照：`../../../activation/bridge_v1/RESULTS.md`；未隨本次報告發布）。qSiLU 有 ball 收益，但原融合 gate 尚有三項未過。直接 FP-QK 教師八項全降，KD 訓練未啟動；正在確定 大 Detect／BBAT5 Pose 教師方案（本機／歷史參照：`../../../kd/dual_task_v1/TEACHER_OPTIONS.md`；未隨本次報告發布）。目前無執行中的本研究 GPU job；後續 GPU 工作維持 600 秒 blocking monitor。見[最新紀錄](<../../worklogs/2026-09-11-activation-result-teacher-options.md>)。

前一階段的 J3、新舊 combine 同口徑比較、Pose head 恢復與 BN 校準結果保留於恢復報告（本機／歷史參照：`../../../combine/bridge_v1/BBAT_RECOVERY_RESULTS.md`；未隨本次報告發布），不代表目前 activation 狀態。所有變更見[工作紀錄索引](<../../worklogs/README.md>)。

## 以下為歷史進度（非目前狀態）

目前 **balanced-j3-v1** 執行中。J2 完成 23 個 epoch 但未全數通過最終 gate；最後 J3 低 LR 微調已驗證並啟動，完成後集中比較 J1／J2／J3 與 MASF 開關。見[最新紀錄](<../../worklogs/2026-09-10-balanced-j2-result-j3.md>)。

目前 **balanced-j2-v1** 執行中。J1完成14epoch，最佳COCO .504954／person .625437／Pose .888378；COCO通過但Pose／bat未過最終gate。已驗證J2來源、train-only梯度校準與backbone9+真實更新，接續低LR適應，尚未activation。見[最新紀錄](<../../worklogs/2026-09-10-balanced-j1-result-j2.md>)。

當前 active run：**balanced-j1-v2**。v1 在訓練前配置輸出 TypeError，已修復並通過實際配置／保存回呼 CPU 回歸，只重啟失敗工作，設定與資料不變。

目前 **balanced-j1-v1 聯合融合訓練中**：依使用者報告實驗4，training-only校準Pose weight為0.045（確認共享梯度比1.28），真實smoke通過後啟動AdamW低LR J1；最終COCO下降0.005仍保留，訓練暫時適應與最終驗收分離。見[啟動紀錄](<../../worklogs/2026-09-10-balanced-j1-start.md>)。下方「正式J1尚未啟動」已是歷史。

已閱讀使用者 combine PDF 實驗 4。merge J0 已結束、正式 J1 尚未啟動；目前做 train-only 任務梯度校準，避免照抄 1：0.25 後仍嚴重失衡，並核對早期停止與最終驗收差異。見[實驗 4 分析](<../../research/2026-09-10-combine-report-experiment4.md>)。

完整 Pose 已完成 42 epoch 平台停止，最佳 Pose AP **0.897997**；目前 `merge-j0-v1` 在已驗證 10% Pose trunk 初始化下做融合前 Pose head 適應。原独立 Detect／Pose 保留，尚未融合驗收。見[最新結果](<../../worklogs/2026-09-10-full-pose-result-merge.md>)。

當前完整 Pose run：`full-pose-gentle-v1`，首輪 E1 觸發精度保護後，以同起點 LR×0.1 受控接續，未更換 AdamW 或放寬 gate。見[最新事件](<../../worklogs/2026-09-10-full-pose-gentle.md>)。

**完整 Pose AdamW 適應已啟動並監測。** head-only 加訓 40 epoch 完成，Pose AP 0.8071 → 0.8525、COCO 不變；接續獨立 Pose backbone／Neck／head 訓練，尚未融合。見[最新工作紀錄](<../../worklogs/2026-09-10-full-pose-adamw-start.md>)，以下 head-only 執行中敘述為歷史。

最新使用者要求是先訓練好**完整 Pose model**，不只 Pose head；目前 head-only 作前置適應，之後獨立 Pose 分段解凍、驗證融合相容性。AdamW 為主，MuSGD 需實證。見[接續政策](<../../worklogs/2026-09-10-full-pose-before-fusion.md>)。

**先完成 Pose head 適應再融合。** P3 bridge J1 首 epoch 已因 COCO 下降而保存停止；已啟動 Pose-only 延長適應（最多 40 epoch、patience 10），固定 trunk／Detect／MASF，持續 blocking monitor。參見[最新診斷與修正](<../../worklogs/2026-09-10-bridge-pose-first-correction.md>)。以下 J1 執行中等狀態屬歷史，activation／方向 2 尚未開始。

使用者已決定保留 **P3 bridge MASF** 並繼續 combine → activation → 方向 2；新分支 bridge_v1（本機／歷史參照：`../../../combine/bridge_v1/README.md`；未隨本次報告發布） 已通過安全載入、CPU 等價、完整初始驗證及真實更新 smoke，正式 J1 已啟動。α 可訓練，完成後比較同 checkpoint 開／關；Pose 不受此 Detect-only 推論開關影響。J0 head 經共享 trunk 全部 568 個狀態張量等價確認後沿用，不重訓。見[重新啟動紀錄](<../../worklogs/2026-09-10-bridge-combine-restart.md>)。以下停止等待與舊執行敘述均為歷史狀態；舊 queue 保持停用，後續 activation／方向 2 尚未啟動。

新增 Head 對照／P3 bridge 的架構與運算量說明（本機／歷史參照：`../../../combine/pose-masf/ARCHITECTURE.md`；未隨本次報告發布）：640 輸入下 P3 MASF 增加 75,777 個未融合參數、0.475136 G Conv MAC；bridge 只改訓練梯度，推論不多一次 MASF。相對原 shared P3 是改接線而非增加一個模組。本次未執行 GPU，停止狀態不變。

使用者更正比較對象為 optimize 新方向重訓模型。七組 P2／P3 Detect 結果來源符合；下方舊 Pose MASF 開／關的提升不適用新方向決策。已唯讀確認 BBAT5 Detect／Pose val 683 張影像及 932 個框完全一致，可沿用既有 box 驗證。新 P2／P3 無 Pose head，沒有其 keypoint AP，不擅自重訓；見[範圍更正](<../../worklogs/2026-09-10-new-direction-bbat-scope.md>)。

**目前已停止，等待使用者決定 MASF。** 依最新要求直接使用現有 P2／P3 權重驗證，結果見COCO＋BBAT5 比較報告（本機／歷史參照：`../../../combine/pose-masf/RESULTS.md`；未隨本次報告發布）。原 Pose 的 MASF 對 ball／bat box 分別 +0.005220／+0.004674，但 bat keypoint -0.001746；P2 Detect 在 BBAT5 則 ball 小降、bat 小升。這次直接比較未重訓。先前無 MASF J0 已完成，新 MASF 正式訓練未開始，後續 queue 停用；不接 J1/J2、activation 或方向 2。下方執行中狀態為歷史紀錄。

最新接續：P2 MASF 最後配對未過門檻，無 MASF 融合選 control E2（overall 0.508330／person 0.627858）。CPU 組裝與完整 COCO／原 Pose baseline 驗證完成；共享初始 Pose AP 明顯下降，因此先固定 Detect／trunk，啟動 J0 Pose head 適應 8 epoch。真實兩 batch 更新與 live／EMA 固定狀態檢查通過。另依使用者新要求建立 Pose 資料集 MASF 對照研究，尚未訓練。見[最新工作紀錄](<../../worklogs/2026-09-10-p2-result-no-masf-fusion.md>)及融合區（本機／歷史參照：`../../../combine/README.md`；未隨本次報告發布）。下方 P2 執行中／等待選擇敘述為歷史紀錄。

使用者追加最後一次P2 MASF實驗，不新增Detect head；失敗則放棄MASF接融合，已解除原候選選擇等待。P2 preflight／128張配對更新／完整初始AP／匯出已通過，正式5epoch配對已啟動；PWL實際範圍固定[-10,0]、20段。見[實驗紀錄](<../../worklogs/2026-09-10-p2-masf-last-trial.md>)。下方完成與等待選擇狀態屬追加前歷史，不代表本次P2已完成。

已建立 combine 準備區（本機／歷史參照：`../../../combine/README.md`；未隨本次報告發布），CPU 稽核確認兩個方向1候選與舊Pose的layer16不相容。等待候選選擇與適配驗收，訓練及queue均未啟動；不繞過graph audit。

融合前方向1的本輪實驗與兩候選完整COCO匯出驗證已完成，仍未證明MASF／HOG／RepConv有足夠額外增益、精度尚未完全回到FP。集中報告與權重入口：方向1結果（本機／歷史參照：`../../../reports/direction1-20260910/README.md`；未隨本次報告發布）。目前本研究GPU工作皆已結束；是否保留MASF作融合起點待確認。後續順序為combine → activation（QSILU優先）→方向2，均尚未啟動。下方「融合後」結果及舊排程屬歷史紀錄，不取代本次狀態。

## 研究分流（2026-09-09）

- 融合前 Full35-B100：方向 1（本機／歷史參照：`../../../studies/pre-fusion-full35-b100/README.md`；未隨本次報告發布）：新啟動的獨立 Detect 研究，來源為架構專案 final 的 full35-b-f10；程式及產物隔離在 studies 內。
- 融合後 Detect＋Pose：第一輪結果（本機／歷史參照：`../../../optimizations/integrated-roadmap/round1-audit.md`；未隨本次報告發布）：既有研究與全部權重保留，以下歷史結果專指融合後模型，不代表新研究狀態。


本目錄整合 YOLO26M Detect＋Pose 的第一輪精度恢復研究與實驗。最新狀態：預設候選已完成或依既定 gate 停止，但沒有新的精度候選通過驗收，仍保留原 J3 `best_joint.pt`。目前沒有 active GPU job；不能把 queue 跑完稱為精度恢復成功。

## 目前結果

- 第一輪證據總表與需求稽核（本機／歷史參照：`../../../optimizations/integrated-roadmap/round1-audit.md`；未隨本次報告發布）：每個方向的實測、停止原因、未解事項與部署邊界。
- 逐 epoch CSV（本機／歷史參照：`../../../optimizations/integrated-roadmap/results/epoch-comparison.csv`；未隨本次報告發布）：6 組訓練、25 個完成 epoch、50 筆 EMA／live 八項 AP。
- 可重建證據索引（本機／歷史參照：`../../../optimizations/integrated-roadmap/results/README.md`；未隨本次報告發布）：來源、產物與重建方式。
- 12 組同圖 GT／BEST／E5 對照（本機／歷史參照：`../../../artifacts/direction1-20260909/pose-error-audit/comparison.html`；未隨本次報告發布）：開發案例，不是獨立 test 或使用者實際影片驗收。
- [BN、MuSGD 與 J2 最新結果](<../../worklogs/2026-09-09-bn-musgd-j2-results.md>)。

原 BEST joint=0.71117474。HOG、RepConv17、MASF bridge 與 heads-only 均未升格；QK 單點 FP score 替換下降，不能當成純 FP teacher 對照。MuSGD 的 train-only 更新校準未符合全部分組門檻，不啟動長訓。J2 的 ball 較好，但 person／bat 退化，保留 J3。

固定 scale 本地 early-return 已通過 CPU Float／BitTrue 整圖等價與完整 BitTrue 驗證，8 項 AP 不變。這是省略冗餘計算的工程修正，不是精度改善；未宣稱硬體加速、純整數部署或完整 export。

## 計畫與執行邊界

方向 1 master plan（本機／歷史參照：`../../../optimizations/integrated-roadmap/direction1-master-plan.md`；未隨本次報告發布） 定義整合順序；optimizer policy（本機／歷史參照：`../../../optimizations/integrated-roadmap/optimizer-policy.md`；未隨本次報告發布） 保存超參數與前置門檻；機器可讀狀態（本機／歷史參照：`../../../optimizations/integrated-roadmap/direction1-plan.json`；未隨本次報告發布） 與最新稽核記錄實際完成情況。歷史階段敘述不作目前 queue。

GPU 工作使用每次最多 600 秒 blocking monitor，正常不讀進度 log、不額外查 GPU，退出或異常事件才處理。只使用主代理，不啟動子代理。所有新訓練 warmup 為 1 epoch；Detect logical batch128，選定 physical32×4，不能說 physical128 已可用。

第二輪與 person-only 仍暫緩。BinaryQK STE／KD 須另立合法梯度、teacher、resume／export 契約；不修改 baseline guard 偷跑。既有量化專案明確延後至模型最後階段，恢復需重新確認，不能自動重啟其 PTQ／QAT。

## 資料與來源

接續來源為唯讀 `/home/uxin/yolo/yolo_combine/final/full35/`。原始 BEST、歷史 checkpoint、logs 與資料均保留；沒有 commit、push 或清理操作。

所有新棒球實驗只能使用不可變 `/home/uxin/yolo/original/pose/derived/bbat5-v1/`；Pose 入口 `configs/pose.yaml`，ball/bat Detect 入口 `configs/detect.yaml`，registry 為 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。本輪 Detect 任務仍是 COCO80，不是 BBAT Detect 或 person-only。不重切、不抽樣、不改 labels；runtime View 不另成資料版本。完整規範見[資料集規則](<../../../../docs/agents/bbat5-datasets.md>)。

## 文件導航

- 優化方向索引（本機／歷史參照：`../../../optimizations/README.md`；未隨本次報告發布）
- [研究索引](<../../research/README.md>)
- [中文工作紀錄](<../../worklogs/README.md>)
- [MASF P3 位置診斷](<../../research/2026-09-01-yolo26-p3-masf-no-gain-diagnosis.md>)
- [RepConv／BinaryQK 研究](<../../research/2026-08-31-repconv-binaryqk.md>)
- [HOG 訓練適配研究](<../../research/2026-09-04-paper31-to-yolo26m-training-adaptation.md>)
- 第二輪提案（未啟用）（本機／歷史參照：`../../../optimizations/round2-innovation/README.md`；未隨本次報告發布）

CPU 重建交付表：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/uxin/yolo/yolo_combine/.venv/bin/python scripts/build_round1_evidence.py
```
