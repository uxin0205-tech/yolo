# 2026-11-08：固定 qSiLU Rep17＋20、清理 Stage8 與啟動特徵圖研究

## 任務與範圍

使用者要求將 **qSiLU 訓練版 `rep17_20` E1** 當作最終 Stage8 結果，整理後自主啟動下一階段並排 GPU queue。後續要求增加訓練輪數。此授權接續先前的 Stage9 待指定狀態；只執行主代理、不派子代理。Stage9 使用 `yolo_actshape/`，階段完成後停止。

## 變更與清理

- 最終 folded checkpoint：`yolo_achitechure/result/stage8-retrain/handoff/rep17_20.pt`，SHA `2454548104ed64768eb9ecdb99d44804ee9d8e1e371e2a508cf052c0911fdd35`。
- 多分支最佳 EMA：`training/rep17_20-best-ema.pt`，SHA `7ddd8bebb86dc24a3feab15edd840587dd56e9afeb2a1994f89084687a1afb34`。不是含 optimizer 的續訓檔。
- 交接只保留上述候選；其餘新舊 Rep／ACB 權重、optimizer resume、initial 與已退出主線的 `achitechure_2/` 直接刪除，無 archive。共刪除 27,268 個檔案項目，釋放約 **4.080 GiB**；逐項 SHA／大小／連結與理由見 `yolo_achitechure/result/stage8-retrain/cleanup-final-20261108.json`。
- `achitechure_2` 必要小型負結果已抽取到 `result/reference-architecture2/`。`achitechure_1` 因 optimize 的實際 imports、final 與 checkpoint 依賴保留；沒有猜測刪除。
- Stage8 原自動建議 `rep17_20_silu`、gate 未通過及舊 AP 均保留為歷史事實。使用者選擇不被改寫成自動門檻通過。

## 動手前參考與本輪差異

已唯讀核對 optimize 的 `experiments/pose_only_rep_v1/{config.json,run_arm.py,RESULTS.md}`、`experiments/activation/bridge_v1/train.py`、`qsilu_pq-short-e10-seed1-v1/{resolved-config.json,summary.json}` 及 combine J3。參考完整配方、parent 與既有結果，SHA 鎖於 `yolo_actshape/protocol.json`；採用及差異見 `reference-review.json`。

沿用 AdamW（.948/.999）、weight decay .00027、warmup 1E／cosine 最終 .5、fresh EMA、FP32 CIoU、clip10、完整 epoch resume。neck LR 1.9e-6、雙 head LR 5e-6。這是新的形狀適應研究，optimize 未驗證相同 final parent 的尺寸矩陣；不冒稱其原結果。

本輪固定 backbone／Attention 與所有 BN running，只訓 neck／雙 head；繼承已 folded Rep，沒有重新展開分支。訓練前向固定 qSiLU／D1，反向沿用既有 Float PWL proxy。physical batch 8，Detect 累積 32 批＝256，Pose 2 批＝16，維持 reference batch64 的 loss scaling。固定 canvas 適應保留 HSV、Detect 翻轉，停用 random affine／crop，避免再混入隨機幾何變因。

## Queue 與預算

先做原 rect640、固定 640×640、512×512、384×768、512×1024、1024×1024 的完整 E0。每組 COCO val5000、BBAT val683，原 evaluator、FP32、E2E、conf .001、IoU .7、max_det300。原 rect640 必須重現選定 parent 的十項 AP（最大差異不超過 1e-5），否則停下查核。

依 E0 各挑一個省運算組與較高解析度組，最多兩組；任一 AP 下降超過 5pp 或綜合分數下降超過 2pp 者不盲目恢復。選定後先做梯度校準及兩個真實 macro smoke，再啟動完整訓練。使用者要求增加輪數後，預算在 GPU 啟動前改為每組 **10E＋半 LR 10E**、最多 **40E**；後段從該組最佳接續，每輪完整驗證，較差後段不覆蓋最佳。嚴重退化／非有限數值先停止診斷，不自動增加第三段。

600 秒監測 GPU、溫度、記憶體及進度；30 秒檢查其他 GPU compute 工作，有衝突只停自己的 worker、待資源釋出從完整 epoch 接續。SIGSEGV 僅允許一次、且必須有來源／配置／SHA 相符的完整 epoch resume；未完成 epoch 不盲目重跑。`artifacts/shape-20261108-v1/STOP` 可停 queue。

## 驗證與結果

- 最終 Stage8 縮減交接包重新執行 `validate.py` 通過：CPU 160／640 全輸出逐位相同，拒絕 SiLU 回退，未使用 GPU。
- Stage9 四個測試通過，涵蓋 16 組長寬／canvas 的 bbox 與 keypoint 座標反算、原標註不變、非法尺寸拒絕、每組最多一候選、禁止 control 加訓與有界重試。
- 五種固定尺寸的 CPU 全模型前向與 qSiLU／D1 契約通過；切換矩形後切回的完整輸出逐位相同。
- 640×640 時 P3／P4／P5 分別 80×80、40×40、20×20；兩個 Attention 都在 20×20。512×512 時為 64×64、32×32、16×16。Hadamard 處理 key channel32，與 token 是否二次冪不同。
- 最大 1024×1024 對應 Attention 32×32、N1024，符合 bias 單軸32及分母 unsigned26 契約；D1 equal-score 數值檢查通過。
- Conv＋P·V both routing 估算：fixed640 47.407616 GMAC；512平方 30.322（約省36.04%）；384×768 34.117（約省28.04%）；512×1024 60.711；1024平方121.691。未計逐元素、資料搬移及 binary QK 成本，非實測 latency。
- 完整 val metadata 掃描5000／683張：COCO 直式1228張，長方形可能犧牲其有效解析度；BBAT 無直式，約2:1 canvas 平均補邊24.3%，平方約28.9%。仍須看實際 AP。
- 真實 Detect／Pose 各2張 384×768 CPU loss 前向有限；完整 membership、原生 collate、固定 BN 設定通過。來源 preflight 核對上游410份程式與本輪來源通過。

實驗狀態以 `yolo_actshape/artifacts/shape-20261108-v1/status.json` 為準，GPU 啟動與實測時間在後續段落記錄。此處 CPU 數據不能代替完整 AP。

## 困難與解法

固定長方形不能直接交給標準 training 的整數 imgsz 入口，因此新增專用 runtime canvas，仍使用原生 LetterBox／Format。預 resize 的 H/W ceil 比例可能不相同，與原生 scalar 座標反算不完全對應；固定 canvas 改由 LetterBox 一次 resize，bbox／keypoint 共用同一 scalar，實際反算測試通過。原 rect640 參考仍使用原資料層，不改歷史 evaluator。

為避免新 best 寫入後、resume commit 前中斷而覆蓋舊 best，採用每 epoch 的 best 檔名；完成新 resume 後才刪除同段被取代的 best。其餘尚無已知數值困難。

## 資料與限制

COCO80 仍使用既有完整來源；BBAT5 只讀 `/home/uxin/yolo/original/pose/derived/bbat5-v1/` 與原 registry，train5964／val683，未重切、抽樣、修改影像或標註。shape 只在 runtime 改變。Attention 共用算子保持在 `yolo_attention`；`yolo_optimize` 完全唯讀。

單 seed、validation 選型，沒有原形狀 control 加訓；有效解析度及補邊共同改變，不能把差異全歸因二次冪。QKV、P·V、projection、Conv 仍浮點，head sigmoid 等仍存在；ONNX／TensorRT 匯出、RTL／HLS、實板皆未驗證，不宣稱硬體加速。Stage9 完成後會整理 AP／成本／小型數據、可重畫圖、可攜重載與清理清單，再停止。

## GPU queue 啟動

2026-11-08 16:07:39（Asia/Taipei）啟動 supervisor PID 3766388，第一個 worker PID 3766394，使用 RTX 5090。第一組 reference_rect640 已實際進行 COCO 完整驗證，當時顯存約 10.1 GiB；之後依條件式 queue 自動接續。來源鎖 436 份檔案，600 秒 monitor 與 30 秒資源檢查已啟用。

補充封裝驗證：獨立程序以 160×320 與 384×768 比對 Stage9 可攜封裝的完整輸出，逐位相同，qSiLU／D1 檢查及 SiLU 拒絕通過；這是 CPU 封裝驗證，未當作矩形 AP。N288 的 D1 equal-score row-sum 最大誤差約 0.001738，符合事前 .003 數值門檻。

最後再刪除 training/ 中四份已失去權重的未採用 sidecar；SHA 與原因補入同一 cleanup-final-20261108.json，保留小型原始 verify 證據。

補充 `yolo_actshape/ANALYSIS.md`，記錄真實各層形狀、二次冪與解析度的區別、長方形對COCO／BBAT補邊的不同影響、未遮罩token不可直接補零，以及剩餘BN／sigmoid／浮點路徑的限制。此為分析文件，不更動執行中的來源鎖。

已將本次 Stage8 最終選擇、Stage9 自主 queue、40E 上限與長訓後退出 working 的偏好補入根層 AGENTS.md；該文件不在運行中的程式來源鎖內，未改寫配方或已鎖定實作。文件連結與交接檔案存在性檢查通過。

## E0 完成與長訓交接

六組完整 E0 已完成。原 rect640 的十項 AP 與 Stage8 選定 parent 完全一致，最大誤差 0。選定兩組矩形進行恢復，原尺寸／fixed640 不加訓；512平方的綜合退化超過2pp篩選門檻，1024平方有多項退化超過5pp，結果保留但不盲目加訓。

| Canvas | COCO AP | person AP | BBAT box AP | BBAT Pose AP |
|---|---:|---:|---:|---:|
| reference_rect640 | 0.514347 | 0.642796 | 0.614371 | 0.901389 |
| fixed640 | 0.515882 | 0.642821 | 0.608328 | 0.899351 |
| square512 | 0.498495 | 0.616532 | 0.590633 | 0.887790 |
| rect384_768 | 0.496890 | 0.619689 | 0.613683 | 0.899531 |
| rect512_1024 | 0.499511 | 0.641844 | 0.615494 | 0.900870 |
| square1024 | 0.445639 | 0.595150 | 0.577684 | 0.846701 |

384×768 梯度校準通過，Pose task weight＝0.1。兩個真實 GPU macro smoke 通過，415 個參數張量實際更新，固定 backbone／Attention／BN 及 EMA 固定張量檢查通過，qSiLU／D1 未改變；PyTorch peak reserved 約 3.49 GiB。

截至 2026-11-08T16:23:50.543520+08:00，rect384_768/adapt10 正在 E1、macro 50/463，EMA 已更新 50 次，AMP overflow retry＝0。目前約 24.8 分鐘／epoch。兩組各10E＋10E的總完成時間粗估11月9日下午至晚間；第二組依成本比例估算，仍待實測，詳見 eta.json。

長訓穩定後依使用者偏好退出 working。背景 supervisor 持續執行600秒監測、30秒資源檢查、完整驗證與條件式降LR；下一組與後段均已授權。最終結果尚未出爐，不將E0或smoke當作訓練後成果。

## 五級特徵圖與輸入尺寸的說明補充

使用者追問模型的五級 feature-map shape，以及40×40為何必須採這個數字。核對本機 YOLO26 backbone 的stride2層（0、1、3、5、7）及本輪CPU trace：五級累積stride為2／4／8／16／32；固定640輸入時依序320／160／80／40／20。現有雙head使用後三級P3／P4／P5。

本輪選定的384×768、512×1024是**輸入canvas的高×寬**。前者對應五級192×384、96×192、48×96、24×48、12×24；後者對應256×512、128×256、64×128、32×64、16×32。後三級及兩個Attention已有實際CPU trace驗證；前兩級依既有stride2配置推導。40不是硬性規定，而是640÷16的結果。

已在Stage9 README明確區分：目前是調整輸入以改變整條特徵圖尺度，尚未實作固定輸入、獨立調整某級特徵圖的架構候選。後者需另核對resize／stride、neck concat與bbox／keypoint對應；不能把目前矩陣宣稱為已完成此研究。本次僅說明與文件補充，沒有改動執行中的來源鎖。查核時GPU queue仍在rect384_768/adapt10，未見錯誤。困難：先前尺寸名稱容易混淆；解法為明列輸入與各級特徵圖映射。

## 硬體需求優先的尺寸分析補充

使用者指出目的是讓硬體容易實作，並觀察目前各級仍是除以2。本次釐清stride2是取樣步進而非數值除法，需分開看邏輯H/W、實體pitch、channel packing、tile與片上容量。只按輸入尺寸／AP選型不足以宣稱硬體最優。

先唯讀查閱optimize的hardware-codebook研究及對應計數依據，核對本輪Attention forward／bias實作；再用純CPU算術比較七組geometry、4/8/16/32空間tile、64-token query塊、N²與容量。另核對optimize的analytical／runtime profiler，明確區分估算與GPU／板端量測，未套用其舊N400或FP16結果。參考SHA收在hardware-shapes-cpu.json。

新增 `yolo_actshape/analysis/hardware_shapes.py`、JSON／CSV與中文分析文件。40平方在8平方tile下完全對齊，在16平方tile下配置容量多44%；證明不能單憑是否二次冪判斷。新增256×512、384×512、512×768的內部三尺度候選，均維持既有bias／token上限，實際CPU相容與D1誤差另行記錄。所有新增檔在獨立analysis目錄，沒有修改執行中的Stage9來源鎖或訓練參數。

已查閱AMD HLS官方memory access／line buffer文件及NVDLA官方32-byte資料對齊格式，連結見分析文件。目標晶片、位寬、片上容量及並行lane尚不明，已以非阻塞方式詢問；先提供平台未定下的估算，不宣稱已選硬體最佳。CPU讀取實際模型確認五個下採樣Conv皆kernel3／stride2／padding1且未啟動CUDA。

困難：硬體條件未定，單看二次冪可能反而增加記憶體與N²；解法為分開列明容量／tile利用率與AP，保留數學形狀不變的分塊方向。GPU既有訓練查核仍正常，E2期間無AMP overflow重試。新增候選尚未做正式AP、GPU訓練、RTL/HLS或實板量測。

新增三個尺寸的CPU實際前向、qSiLU／D1／Rep契約及均勻score誤差均通過。Conv＋PV計數依序為256×512：15.153 GMAC、384×512：22.735 GMAC、512×768：45.508 GMAC（精確值以hardware-shapes-cpu.json為準），非硬體延遲。新增proposal結果尚無AP。

## Queue 完成

採用 reference_rect640；沒有候選通過採用門檻，保留原尺寸。完整結果見 `yolo_actshape/result/stage9/README.md`。獨立 CPU 可攜驗證通過。安全清理清單見 cleanup.json，未啟動下一階段。
