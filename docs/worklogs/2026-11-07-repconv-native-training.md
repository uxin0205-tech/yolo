# 2026-11-07 第八階段：原生 RepConv 訓練、硬體部署驗收

使用者已授權啟動新訓練，以及直接刪除不再使用的舊產物，不封存。此授權接續 11 月 4 日的需求澄清。

## 變更與理由

先唯讀核對 optimize 的 pose_only_rep_v1 配置、models、run_arm、三組實際 resolved-config／summary／smoke，以及上游 training_b 的 EMA。來源 SHA 與逐輪 AP 記錄於 yolo_achitechure/stage8/reference-review.json，先於新實作寫入。

新輪使用原生 RepConv，訓練前向為 SiLU／FP32 exact Softmax，B2 二值 QK、Q10 scale／bias 沿用。後續部署先在相同算子下折疊，再替換 qSiLU／D1；以部署 AP 選型。這項常規算子訓練是本輪延伸，optimize 的實際 Rep 訓練使用 qSiLU／PWL。

三組獨立從 Stage7 E8 出發；選定 Rep 與兩個 heads 可訓，BN running 固定。各 5 輪，持續改善才降 LR 延長 5 輪，總上限 30 輪。新算子重新做 train-only 梯度校準。既有 Stage8 最佳部署模型暫留為可執行比較來源；無用 resume、中間權重直接刪除。

## 驗證進度

已核對參考 SHA 未變及三組 optimize 正式訓練／smoke 完成。CPU、短訓、完整部署 AP 與清理驗證待執行；尚未啟動 GPU。

## 困難與解法

舊載入器將 qSiLU／D1 同時視為訓練與部署限制，需明確分開 manifest 與重建路徑，禁止載入時靜默換算子。

## 未解事項與風險

常規訓練後的算子轉換可能降低部署 AP；每輪雙版本評估、保留最佳部署結果，不用原生 AP 代替。無硬體綜合或板端量測。第九階段未啟動。

## 使用者要求先查核後的修正

使用者在 GPU 啟動前要求「等等，參考 optimize，確認 SiLU 訓練是否改變表現」。立即停止啟動；沒有 GPU 訓練或 AP 作業。查得 optimize 的 Rep 訓練確實保留 qSiLU／Float PWL，原生 RepConv 不是必須用 SiLU。撤下 SiLU／exact Softmax 草案，protocol 設 `gpu_launch_enabled=false`，run 根目錄設 STOP。

已完成 CPU 三組初始／非零分支 folding、FP32／FP16 單元與 160／640 完整輸出、兩算子模式重載；可攜交接 pytest 1 passed，舊最佳包獨立驗證 passed。這些不代表新方案 AP 已驗證。

直接清理舊初始／多分支／resume、非選定部署及撤下草案初始權重，共 1.928 GiB；未封存。舊最佳 Rep17＋20 folded SHA 不變，optimize／正式資料／Stage7／其外部依賴未修改。

困難：先前把「允許常規算子」擴大成全網換回 SiLU，雖已標示是延伸，仍偏離 optimize 的實際 Rep 配方。解法：取消該啟動，保存查核與小型 CPU 證據，建議保留 qSiLU，另行釐清 Float PWL 訓練到 D1 部署。未解事項：新輪配方尚未重新固定，GPU 保持停止，Stage9 未開始。詳見 [本次查核](../research/repconv-optimize-operator-review-2026-11-07.md)。

補充驗證：實際呼叫 queue，確認在 GPU 與來源鎖建立前明確拒絕已撤下草案；根層 README／整合計畫／AGENTS／專項入口與 RUNBOOK 已同步目前停止狀態。

## 最新授權：重新訓練與少量變體

使用者確認可用前述方向、考慮 RepConv 變體，總之重新訓練。解除暫停；先再次核對 optimize 全部來源 SHA 與實際配方，再查 ACNet 作者原始實作。新 protocol 在實作前寫入：Rep17／20／17+20 使用 qSiLU+Float PWL；額外 Rep17+20 僅 Rep activation 用 SiLU；額外 ACB17+20 用 qSiLU。所有部署維持 qSiLU/D1/B2/S1。

每組5E，符合改善條件再5E；SiLU 若有部署落差且整體尚可，允許一次3E硬體算子適應，總上限53E。原資料、BN running固定、選定Rep及兩heads、train-only梯度校準沿用已核對方法。這是預訓權重適應，不是隨機初始化從頭訓練。

CPU／GPU 驗證與啟動結果續記於下；暫無新的正式AP。困難：SiLU僅換Rep位置需明確拆分共享activation alias，部署再恢復；ACB須驗證非方形輸入和stride2邊界，不能只看權重可載入。

## 五組前置驗證完成

CPU五組全部通過：每個候選在相同算子下初始等價、還原硬體算子後與Stage7等價、非零分支folding，FP32/FP16單元使用13×18輸入驗證非方形與stride2邊界，160/640完整輸出、各模式嚴格重載與非法算子拒絕。實際calls：四組qSiLU163次／SiLU0；SiLU組qSiLU161／SiLU2；部署全部qSiLU163／SiLU0。ACB訓練參數28,423,088，折疊後26,453,168，與Rep17+20部署相同。

可攜交接與硬體適應測試4 passed（14.59秒），涵蓋一般Rep、Rep-only SiLU與ACB；未執行新的正式AP。準備在空閒RTX5090以序列queue執行；預估6–10GiB顯存，實測以smoke為準。optimize來源SHA重新驗證無變。無新增困難。

## GPU 啟動與交接

2026-11-07 14:32:56 啟動 supervisor PID 3695639，run 位於 `yolo_achitechure/artifacts/stage8/repconv-variants-20261107/`。第一組完整 E0 硬體AP通過，與原Stage7最大絕對AP差約2.39e-6。train-only校準選Pose weight 0.21；2macro smoke通過，新分支和兩heads實際更新，frozen/EMA通過；peak allocated 5.859 GiB。

目前正式 `train-rep17` 第 1 輪、第 20／463 macro，EMA updates 20，本次記錄AMP overflow retries 0。410項來源鎖核對一致。新程式包含5組候選、Rep-only SiLU alias拆分／部署還原、ACB folding、逐輪雙算子AP、條件式低LR／硬體適應、保護既有最佳與可攜交付。

首輪25E與評估預估15–18小時，約11月8日清晨至上午；若條件式延長，依實際工作量增加。每600秒持續監測、每60秒避讓GPU衝突。長訓已穩定，依使用者偏好退出working，背景queue繼續。尚無新訓練後AP，不能宣稱優於舊模型；本階段完成後停止，Stage9不啟動。無新增困難。

## 20:05 使用者查詢進度

Rep17、Rep20 各完成5E，部署最佳均E1，folding與重載完整AP驗證passed；來源鎖410項未變。兩組均未符合原定持續改善延長條件，沒有追加輪數。目前 `train-rep17_20` 第1輪 200/463，其後仍有Rep-only SiLU與ACB。GPU約95%、7781MiB；queue正常持續，600秒監測最新紀錄存在。

Rep17部署COCO/person/球/球棒/Pose AP為51.4681/64.3073/51.6807/47.4786/90.0236%；Rep20為51.4086/64.3344/51.4966/46.6321/89.9160%。雖部分Detect指標較舊最佳提升，Rep17球Pose相對Stage7降0.2265pp；Rep20球Pose降0.4460pp，另COCO球棒及整體Pose超過0.1pp保護門檻，故兩組尚未採用，不覆蓋舊最佳。

無需中止或改寫執行中配方。若後三組未觸發延長，依實測約30分鐘/輪推估11月8日04:00–06:00完成首輪與交付驗證；延長另計。新SiLU與ACB尚未有正式結果，不能先推論效果。困難：目前存在Detect改善／Pose退步的取捨，繼續既定候選比較；無執行阻礙。

## 21:38 使用者查詢進度

queue 正常，Rep17、Rep20已各完成5E及重載驗證；Rep17+20進到第4/5輪，230/463 macro。前3E最佳仍E1，部署COCO/person/球/球棒/Pose為51.4347/64.2796/51.5782/46.9576/90.1389%。相對舊最佳COCO小幅增加0.0609pp、Pose降低0.0437pp；相對Stage7 parent的球棒box降低0.1401pp，尚未通過0.1pp保護門檻。不能宣稱新輪全面改善。

SiLU與ACB尚未開始，無其正式AP。GPU使用約7769MiB，最新600秒監測正常；當下macro沒有AMP重試。維持已鎖定queue，不改寫正在執行的配方。無執行困難；尚待後續候選判斷。未觸發額外延長時首輪仍估11月8日04:00–06:00完成。

## 自動完成紀錄

2026-11-08T14:33:07+08:00：五組完成，建議 `rep17_20_silu`，最佳 Rep `rep17_20_silu`。folding、重載完整 AP 與可攜 CPU 驗證均通過；詳見 [正式結果](../../yolo_achitechure/result/stage8-retrain/README.md)。Stage9 未啟動。
