# qSiLU 結構格式優先與短 QAT 修訂

日期：2026-09-06

## 變更內容與原因

- 保留目前正在執行、SHA-256 為
  1ae2965674a91b5e75def5c7c3d18d75d2067d0762e669916a3c4548fe10512a
  的 V30 Q0–Q2，不中途改動 matched-sham／QAT 契約。
- 新增 configs/experiments/v33-qsilu-special-formats-first-continuation-v1.yaml，
  只在 Q2 兩個 arm 完整結束且 Q3 parent 通過門檻後接手。
- 148 個 deployment weight paths 仍全部做 CPU profile，但執行與報告順序改成
  optimal Fixed-SD4、exact-scaled ternary、TWN-v3、Paper-TWN v2 優先；
  W4、W5、W6、W7 延後為 uniform backfill。
- 明確加入 LSQ-SD4：它不是 PTQ 格式，而是 Fixed-SD4 位於 recovery band 時，
  以 exact optimal scale 初始化後進入的 learned-scale QAT lane。
- 後續 short-QAT 從最多 8 候選、每 arm 15 epochs，縮成最多 4 候選、
  每 arm 4 epochs、patience 2。這一階段只作 finalist screening，
  正式較長確認需另行審核。

## 執行中途的使用者優先序修正

- 使用者要求先暫停目前耗時的 qSiLU sham，週二前優先產出較小但可解釋的結果；
  同時澄清不是取消後續訓練，原本 PTQ、sensitivity、短 QAT 與 finalist
  confirmation 的完整脈絡仍保留。
- 於 epoch 8 正在執行時對精確 process group 1621525 發送 SIGINT。
  epoch 8 的部分進度不保留；最後完整邊界是 epoch 7，resume 時從 epoch index 8
  重新開始。
- 封存清單為
  artifacts/archives/v30-qsilu-sham-paused-for-tuesday-v1/archive-handoff.json。
  resume last.pt SHA-256 為
  15a9ea76aab2190f864c52f0eb83c725e40ea97d6ef997f3b9a9b4bc951966c4。
- 暫停後已確認相符 process 數量為 0、GPU compute process 數量為 0。
  沒有刪除 checkpoint、metrics 或 Queue artifact。

## 週二快速線與逐層 sensitivity 修正

- 使用者補充 PTQ 必須能回答 layer sensitivity，而不只是比較整個區域。
  因此先完成 qSiLU＋A8 父模型下 148 個 deployment weight layers 的 CPU
  profile；每層都比較 uniform W4、optimal Fixed-SD4、Paper-TWN v2、
  TWN-v3 filterwise 與 exact-scaled ternary，並同時保留 master／deployment
  兩個 view，共 1,480 筆測量。
- CPU profile 位於
  artifacts/reports/v34-qsilu-structured-148-layer-cpu-v1.json，SHA-256 為
  28280630afb4d11ef75cdbd71032da40fa9a8d7cf6a29d78e8e1017ff13dbb77。
  此階段沒有使用 GPU，也沒有訓練；NRMSE／SQNR 只作排序，不宣稱等於 mAP。
- 新增 src/yolo_quantize/tuesday_quick_queue.py 的 fail-closed 逐層選擇：以
  optimal Fixed-SD4 的 deployment-view NRMSE 排序，從 backbone 50 層、neck
  41 層、head 57 層各取最低、上中位數、最高誤差層，共 9 個單層候選。
  這 9 格會用 GPU 完整跑 COCO Detect 與 BBAT5 Pose search validation，實測
  mAP50 與 mAP50-95，藉此檢驗 CPU 誤差能否預測任務敏感度。
- 逐層 9 格之後保留 6 格 qSiLU 結構格式比較：optimal Fixed-SD4、
  exact-scaled ternary、TWN-v3、Paper-TWN v2 的 head-three 路由，以及
  exact-scaled ternary／TWN-v3 的 pose-safe 路由。總計 15 個 PTQ cells，
  全部不訓練；PTQ 完成後才依 dual-metric recovery band 選一格進短期
  LSQ-SD4 paired QAT，並保留後續較長訓練。
- 具體化計畫位於
  artifacts/queues/v34-qsilu-tuesday-quick-v1/generated/v34-qsilu-layer-and-structured-ptq-v1.yaml，
  SHA-256 為
  3b29ad7a36998534615ce6e6d070d842a1f1c6f4668b88fc9562001c556fe469；
  singleton route manifest 與來源 CPU profile 均由 SHA-256 鎖定。

## 驗證方式與結果

- V33 規劃本身為非 GPU 修改；其後依使用者明確要求，安全停止正在執行的 V30
  sham 並保留完整 checkpoint。
- 核對既有 runtime 的 Fixed-SD4 recovery mapping，確認實際 QAT family 已是
  ls_sd4；問題是舊主計畫沒有清楚呈現，而不是演算法完全不存在。
- 核對既有 CPU profile schema，確認每個 path 已能輸出八格式的重建與容量指標；
  GPU promotion 仍由完整 Detect／Pose 指標判斷，未把 NRMSE 當成 mAP。
- preparation 與 Tuesday queue 測試合計 7 passed；ruff check 全部通過。
- 已用 Full35MixedPolicySearchPlan 實際解析具體化後的 15-candidate YAML，
  candidate 數、qSiLU＋A8 activation policy、singleton route 與來源 hash 均通過。
- 封存 SHA、epoch.csv、early_stop.csv 與 process/GPU 空閒狀態均已唯讀驗證。

## V34 PTQ 實測結果

- 15 個 PTQ cells 全部完成，無 failure。原始報告
  artifacts/reports/v34-qsilu-layer-and-structured-ptq-v1.json 的 SHA-256 為
  dfa12f851d0d659a45a80c658c7884f96a63b08dc234caf1cf9d706d05358b83；
  雙指標 re-gate 報告
  artifacts/reports/v34-qsilu-layer-and-structured-ptq-dual-v1.json 的 SHA-256 為
  552633cd3c76affde781d5134411f3975ba4ce374459f615fe3b4ebe4319ce83。
- 下列數值均是「含 qSiLU activation 替換」相對 accepted FP baseline 的最差任務
  total delta，不是只看單一任務，也不是 NRMSE：

| 單層 Fixed-SD4 | worst mAP50 delta | worst mAP50-95 delta | 決策 |
| --- | ---: | ---: | --- |
| backbone CPU-low | -0.012438 | -0.013811 | green |
| backbone CPU-median | -0.989988 | -0.988889 | reject |
| backbone CPU-high | -0.010476 | -0.013152 | green |
| neck CPU-low | -0.011939 | -0.013603 | green |
| neck CPU-median | -0.012329 | -0.014238 | green |
| neck CPU-high | -0.011622 | -0.013131 | green |
| head CPU-low | -0.015709 | -0.015451 | recover |
| head CPU-median | -0.014659 | -0.016440 | green |
| head CPU-high | -0.011576 | -0.013131 | green |

- CPU NRMSE 與任務敏感度不是單調關係：backbone CPU-median 的
  graph.model.10.m.0.attn.proj.conv 單層造成接近 -0.99 的最差下降，但 CPU
  NRMSE 更高的 graph.model.7.conv 仍通過雙門檻。這證明 CPU profile 只能用於
  分層抽樣，不能直接決定部署格式。
- 結構格式組合中，Fixed-SD4 head-three 通過：worst mAP50=-0.014364、
  worst mAP50-95=-0.039130。它涵蓋 8 層、933,888 個權重；FP32 權重約
  3,735,552 bytes，SD4 code＋scale 約 469,824 bytes。exact ternary、
  TWN-v3 與 Paper-TWN v2 的 head-three 全數 reject；兩個 pose-safe 三元路由
  亦 reject，因此不把明顯失敗的三元 PTQ 送入本輪短 QAT。

## LSQ-SD4 短 QAT 排程

- 選擇 Fixed-SD4 head-three 作短 QAT，而不是只有 2,304 權重的 head-low
  recovery 單層；理由是前者已在 mAP50 0.015／mAP50-95 0.04 門檻內，且有
  約 8 倍的實質局部權重儲存縮減，較適合驗證 learned scale 是否能增加精度裕量。
- 具體化 paired 計畫位於
  artifacts/queues/v34-qsilu-tuesday-quick-v1/short-qat-head-three-lsq-sd4-v1/generated/qat-plan.yaml，
  SHA-256 為
  289d3730d94f376ff54ff88cba0e0083e062af9d9f3c85e8c58ddd2285d5dbe3。
  只有上述 8 層套用 LSQ-SD4，其餘 140 deployment layers 維持 float；不混入
  全模型 W8。
- sham 與 QAT 各最多 4 epochs、patience 5、warmup 1；epoch 0 為 scale-only，
  epoch 1 起 full-strength fake quant joint recovery。Detect logical batch 128、
  physical microbatch 16，Pose batch 16；optimizer 沿用 accepted Full35 的
  AdamW 0.1x J3 role LR，augmentation 沿用 accepted pipeline，不加雜訊。
- strict parser、完整 CPU graph/data preflight 均通過，blockers=0。GPU
  forward/backward smoke 通過，256 個 quantizer gradient 均建立，peak allocated
  memory 為 19,908.03 MiB；尚未把 smoke 當成 accuracy 結果。

## V35 多格式逐層 successor

- 依使用者新增需求，在目前 V34 paired QAT 後安排
  src/yolo_quantize/mixed_layer_successor.py；它先只等待 V34 的兩個 arm 完整結束，
  dependency 未完成時不建立 CUDA context、不停止或修改現有訓練。
- 第一段固定同一 qSiLU＋A8 父模型，挑選已由 V34 單層 Fixed-SD4 驗證為 green
  的 backbone、neck、head 代表層，各自比較 W8、W7、W6、W5、W4、
  Fixed-SD4、exact-scaled ternary、TWN-v3 filterwise 與 Paper-TWN v2，合計
  3×9=27 個獨立 PTQ cells。這裡允許非 2 次冪的 W5／W7，不把全網固定成同一格式。
- 第二段從已通過的 Fixed-SD4 head-three 八層政策開始。各區先從獨立結果保留
  hardware、balanced、accuracy 最多三個不重複格式，再依 backbone→neck→head
  累積跑完整雙任務驗證；當前區沒有 green 候選時保留上一個已鎖政策，不把失敗
  格式帶到下游。
- 每格仍要求含 activation 替換後的所有 mAP50 total delta ≥ -0.015，且所有
  mAP50-95 total delta ≥ -0.04；選擇時 CPU NRMSE 只排序，GPU Detect／Pose
  任務指標優先，同精度帶內再選較小容量。
- V35 只在 mixed policy 結束後保留最多一個 final paired-QAT 候選，規劃為 sham
  與 QAT 各 4 epochs、patience 5；格式與路徑必須由屆時實測產生，現在不預填。
  若沒有 green／recover final policy 就不硬啟動訓練。
- V35 相關測試與既有 Tuesday queue 測試合計 9 passed，ruff 通過。27-cell
  independent plan 已由 Full35MixedPolicySearchPlan 嚴格解析，位於
  artifacts/queues/v35-qsilu-mixed-layer-successor-v1/generated/v35-independent-layer-format-search-v1.yaml，
  SHA-256 為
  71b53d73c630d7ff165616be780ee4c8c06feb5cca4a3d656a460e7d04019b0c；
  另以 head-base＋backbone W6＋neck W5／SD4／TWN 的樣例確認 heterogeneous
  cumulative route 無路徑重疊。
- 以 V34 實際 PTQ 報告逐筆驗證 V35 容量計算介面；Fixed-SD4、Paper-TWN、
  TWN-v3 與 exact ternary 記錄均可解析，沒有欄位契約錯誤。
- 已啟動 V35 後繼 queue，狀態為 `waiting_for_dependency`、completed_jobs=0、
  error=null。它只會每 600 秒讀取 V34 dependency 狀態；V34 paired QAT 完成前
  不會取得 GPU，完成後才自動執行獨立層格式搜尋與 backbone→neck→head 累積搜尋。
- V34 `fixed-sd4-head-three` 的 matched sham 與 LSQ-SD4 QAT 已各完成 4 epochs，
  queue 為 complete、completed_jobs=2、error=null。QAT epoch 3 相對 accepted FP
  的 worst total mAP50 delta 為 -0.007406、worst total mAP50-95 delta 為
  -0.009630；COCO Person mAP50／mAP50-95 delta 為 -0.004442／-0.007514，
  通過 -0.015 與 -0.04 雙門檻。
- 同 epoch matched sham 相對 accepted FP 的 worst mAP50／mAP50-95 delta 為
  -0.007922／-0.008535；QAT 對 sham 的逐任務最差額外差值分別為
  +0.000516／-0.001095，COCO Person 為 +0.000021／-0.000322。這表示大部分
  下降來自共同短訓練漂移，而 8 層 LSQ-SD4 本身增加的誤差很小；仍只視為
  4-epoch screening 證據，不宣稱正式 finalist。

## V35 異質逐層搜尋與 final short-QAT

- V35 已自動完成 27 個獨立單層格式格：backbone、neck、head 各選一個代表層，
  並公平比較 W8／W7／W6／W5／W4、Fixed-SD4、exact ternary、TWN-v3 與
  Paper-TWN v2。三區的 W8～W4 與 Fixed-SD4 均為 green；backbone 三種 ternary
  均 reject，neck 三種 ternary 僅 recover，head exact ternary 為 recover 而兩種
  Paper-TWN／TWN-v3 為 reject。因此 shortlist 分別為 backbone
  Fixed-SD4／W8、neck Fixed-SD4／W6、head Fixed-SD4／W4。
- progressive cumulative 搜尋依 backbone→neck→head 鎖定：backbone 代表層
  Fixed-SD4、neck 代表層 W6、head 代表層 W4，並保留先前 8 層 Fixed-SD4
  head-base。最終 `cumulative-head-w4` 共量化 11 層，PTQ worst total mAP50 delta
  為 -0.012497、worst total mAP50-95 delta 為 -0.039564，仍通過 -0.015／-0.04
  雙門檻。V35 status 為 complete_ready_for_one_final_mixed_qat、completed_jobs=4、
  error=null。
- 新增 `src/yolo_quantize/mixed_layer_qat.py` 與
  `tests/test_mixed_layer_qat.py`。建構器逐一驗證 final cumulative YAML、raw report、
  dual report、summary 與 V34 base QAT 的 SHA-256；它把 9 個 Fixed-SD4 路徑升為
  LS-SD4，把 neck/head 路徑分別升為可學習 scale 的 W6／W4，同時保留其餘層
  float。沒有原地修改 V34 或 V35 既有 artifact。
- 相關 mixed-QAT、V34 QAT 與 V35 successor 測試合計 9 passed；ruff check 與
  format check 通過。strict parser round-trip 得到 6 組 assignment、11 個唯一
  path、格式 fixed-sd4／w6／w4；完整 CPU graph/data preflight ready、blockers=0。
  物化 plan SHA-256 為
  `28d5887dd9cc61b1cecbebc1249b9c7aef5c65cd8782dfd9931e34726f0c9a84`。
- GPU smoke 通過：qSiLU+A8 mixed graph 已 materialize，259/259 quantizer parameters
  有梯度，Detect／Pose objective 皆有限值，峰值 19951.24 MiB。已啟動唯一的
  matched sham→mixed QAT queue；每臂最多 4 epochs、patience 5、scale-only epoch 0、
  epoch 1 起 full quant、AdamW 0.1× J3 role LR、Detect logical batch 128／microbatch
  16、Pose batch 16、accepted augmentation、無新增雜訊。目前 status=arm_started、
  current_arm=sham、completed_jobs=0、error=null。
- V35 mixed paired queue 隨後完成 sham 4 epochs 與 QAT 4 epochs，最終
  status=complete、completed_jobs=2、error=null，未觸發 early stop。QAT epoch 3
  是當臂最新 improved joint score；相對 accepted FP 的 worst total mAP50 delta
  為 -0.006746、worst total mAP50-95 delta 為 -0.009365，COCO Person
  mAP50／mAP50-95 delta 為 -0.003636／-0.007732，均通過門檻。
- 同 epoch sham 的 worst total mAP50／mAP50-95 delta 為 -0.005932／-0.007286；
  QAT 對同 epoch sham 的逐任務最差額外差值為 -0.001069／-0.002079，COCO Person
  為 +0.000891／-0.000936。與 sham best-joint epoch 2 比較的最差額外差值也僅
  -0.001096／-0.001106，因此 mixed fake quant 的額外損失遠小於總訓練漂移。
- QAT best-joint checkpoint SHA-256 為
  `c891d8004f90a439f33d5e71054c145ddb8fcf8d2022fd3839a9c7bf1cafb158`。
  11 個選定層合計 5,947,392 個 weights：FP32 為 23,789,568 bytes，量化 code＋
  per-output-channel scale 為 3,570,752 bytes，僅就這 11 層約 6.66× 壓縮、
  減少 84.99%；這不是整個模型的壓縮率，也未計 activation buffer。

## 困難及解法

- 內建 apply_patch 因 bwrap: loopback: Failed RTM_NEWADDR 無法使用。
  改以同一個 apply_patch 程式的互動輸入新增檔案，沒有使用其他寫檔方式。
- 直接改寫 active V30 會造成 plan SHA 漂移並可能中止目前 Q2，因此改用獨立的
  V33 future-stage revision，避免污染正在產生的證據。
- 初始假設 CPU NRMSE 低的層應較安全，但 GPU 結果顯示並非單調；解法是保留
  CPU 148 層完整排名，同時以代表層的完整 Detect／Pose validation 校準，後續
  progressive routing 一律以任務指標為準。
- V35 dependency 完成後，外層 monitor 的 baseline 恰好建立在狀態切換之後，
  因而一個週期內沒有顯示變化；唯讀確認 supervisor session 存活後發現 27-cell
  validation 已正常執行。後續只以 execution-status JSON 監測，不再讀 stdout。

## 未解事項或風險

- V30 現在是可續跑的 incomplete sham，不能宣稱 paired QAT 完成，也不能鎖成 Q3
  parent；週二快速線必須以原始 qSiLU activation parent 或既有 PTQ 證據明確標示。
- 4-epoch QAT 只能作快速 screening，不能取代 finalist 的較長、多 seed 或 formal
  validation。
- 每層 CPU 最佳格式只代表權重重建特性；activation、Add／Concat 與任務輸出耦合
  仍須逐區 GPU PTQ 驗證。
- 9 個 GPU 單層格是由 148 層 CPU 排名抽出的低／中／高代表，不是假稱已對 148 層
  各跑一次完整資料集；完整 148 層逐層比較保留在 CPU profile，GPU 用代表格校準。
- 4-epoch LSQ-SD4 paired QAT 仍須同時跑 matched sham 與 quantized arm；短期結果
  只能篩選，不能用來取代較長、多 seed 的最終結論。
- V35 第一輪是每個 broad segment 一個代表層的九格式實測，不等於 148 層每層都跑
  九次完整 validation；148 層完整數值 profile 已保留，GPU 矩陣依結果再擴展，
  避免一次執行 1,332 次 validation。
- 最終 mixed policy 的 PTQ mAP50-95 距 -0.04 門檻僅約 0.00044，安全餘裕很小；
  因此本輪 paired QAT 是必要的 recovery／穩定性檢查，仍不能替代後續 formal
  full-val、多 seed 與硬體 bit-true 驗證。
