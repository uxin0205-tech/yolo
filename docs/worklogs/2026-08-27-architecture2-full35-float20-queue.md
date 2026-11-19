# 2026-08-27：architecture_2 Full35 Float20 與安全 GPU 佇列

> 更新：本紀錄前半保留 J2 驗收當時的歷史狀態；同日上游已將 J3 升格 accepted，
> 本輪改鎖 J3 並收到 Pose=true 與完整 queue 授權。最新狀態與結果以文末「J3 續行紀錄」為準。

## 任務與範圍

依使用者指定，本輪處理 /home/uxin/yolo/yolo_achitechure/achitechure_2/：

- BBAT5 Detect／Pose 統一引用 /home/uxin/yolo/original/pose/derived/bbat5-v1/。
- COCO80 Detect 使用 /home/uxin/yolo/coco2017/ 與 /home/uxin/yolo/coco2017.yaml。
- parent 固定為 /home/uxin/yolo/yolo_combine/final/full35/ 的 accepted J2 EMA。
- 只比較 C0、C1、C2、C3 固定20% Float，不重做融合或自動選 C_best。
- GPU 已授權排在現行工作後；Pose 是否納入本次 Float20 仍由使用者明確決定。
- 沿用 OOM 經驗：Detect logical train batch 128、Detect/Pose validation batch 16。
- 不執行完整資料長訓練、PTQ、QAT-lite、正式 QAT 或部署量化。

## 變更內容與原因

### Full35 handoff 與候選

- 新增 handoffs/full35-j2-seed0/manifest.json、training-recipe.yaml 與中文 README。
- revision：full35-final-j2-seed0-bd8aad5d。
- checkpoint SHA256：bd8aad5d944e088c6c9f77b5728eed6c0f5ac0509ed96941a05210c184623cb6。
- 以 final factory 重建23層 shared graph，嚴格載入1,238個 EMA tensors。
- 模型契約固定為 COCO80 Detect、BBAT5 ball/bat Pose、kpt_shape=[2,3]、strides 8／16／32。
- Candidate Region 驗證為 graph.model.6、8、13、19。
- C0不改；C1只改 e 0.5→0.375；C2只改 inner_n 2→1；C3只改第一個 inner kernel 3x3→1x1。
- Detect/Pose heads、shared topology、MASF與inherited attention保持不變。
- 新增 Full35Release adapter，集中封裝上游factory、checkpoint、validation materialization與training primitives。
- J3 challenger不作本輪parent，也不與J2結果混用。

### 正式 YAML、runner 與 queue

- 唯一可排隊設定是 configs/runs/full35-float-screen-20.yaml；closed schema是screen-run.schema.yaml。
- batch、fraction、scale、cache、imgsz、epochs、workers、MuSGD、LR groups、loss、augmentation、OOM與queue都在同一YAML。
- 20%由immutable manifest決定，training.fraction固定1.0，不二次抽樣。
- Detect logical128以physical32加accumulation實現；C0完整macro probe若OOM，C0～C3共同改physical16。
- 若啟用Pose，Pose train batch16；Detect/Pose validation都固定16。
- 每候選20 epochs、每epoch93 optimizer steps、合計1,860 steps，不early-stop。
- 每epoch保存model、EMA、optimizer、scheduler、scaler、criteria、RNG、loader與lineage，可exact resume。
- Float20完成後停止；selection_status保持pending_user_decision、c_best保持null，不自動量化。
- Pose兩個YAML gate必須同為true或同為false；pending或不一致都fail closed。
- float20-run只接受持鎖yolo_combine.gpu_queue，並驗證state、child PID、queue_id與lock。
- queue要求無compute process、free VRAM≥30,000 MiB、util≤10%，連續3次、每次60秒。
- preflight要求accepted handoff與C0～C3 CPU證據有效；revision、reload、freeze或source cache漂移就拒絕。
- 2026-08-27檢查時Full35 J3 PID 4052573使用約26,006 MiB、GPU 100%；architecture_2服務未建立。
- 最終再查時J3已結束，GPU free 31,718 MiB、utilization 0%，兩個service均inactive；仍因Pose pending不啟動。
- Pose尚未收到本次run明確true／false，因此未啟動GPU queue。

### Validation、資料與 cache

- 使用Ultralytics 8.4.90 DetectionValidator／PoseValidator與Full35 materialized graph。
- 分開保存mAP50、mAP50-95、per-class AP、P/R/F1、Macro F1與Micro F1。
- C0在screen search-val決定confidence threshold一次，C1～C3固定沿用。
- Micro F1由P/R curves與class supports估算，JSON標記estimated_from_precision_recall_curves_and_supports。
- Pose另存pose-only best與官方Pose+Box combined fitness；未啟用Pose時填not_run。
- canonical BBAT5 formal train／val=5,964／683，search train／val=5,364／600，group leakage=0。
- screen View固定COCO train 23,657、search-val 5,000；BBAT5 train 1,073／410 groups、search-val 600。
- 原始Pose、歷史Detect、canonical images、labels與split均未修改。
- 實際loss smoke發現cache=false仍會建立label index；已移除本輪可重建的：
  - /home/uxin/yolo/coco2017/labels/train2017.cache
  - /home/uxin/yolo/original/pose/derived/bbat5-v1/pose/labels/train.cache
- 新增RuntimeLabelCacheYOLODataset與validator mixin，只寫子專案runtime-cache。
- coco2017/labels/val2017.cache建立於2026-08-12，早於本輪且不屬screen路徑；本輪未修改或刪除。
- 依2026-08-22既有授權，重建完整portable github-dataset；無symlink、權重、cache、run或test。
- 同步更新主README、RUNBOOK、TRAINING_GUIDE、configs／artifacts／results README與CSV／報告範本。

## 驗證方式與結果

- materialized accept-handoff：accepted=true，嚴格載入1,238 tensors。
- C0 transfer：matched 1,238，missing／unexpected／shape mismatch=0。
- C0～C3：640×640 Detect／Pose／both、80×80／40×40／20×20、finite loss／gradient、frozen gradient 0、contract不變、strict reload全部通過。
- 真實batch smoke：Detect/Pose loss有限，分別410／362個gradient tensors；PoseLoss26、RealNVP、RLELoss路徑active。
- status／float20-plan：cpu_preflight.ready=true、passed_candidates=[C0,C1,C2,C3]、source_adjacent_caches_absent=true；唯一blocker是Pose。
- CPU pytest：71 passed、1 skipped。skip是未設定ARCHITECHURE_2_HANDOFF=1的2.35 GB重型重跑；正式materialized驗收已另完成。
- ruff check .：All checks passed。
- config-check：valid=true、spec 2.2.0、SHA256 8297f934fc7946cd0a0d32a4d18b1fa2e87b736e39980977702896238fe9d88e、Ultralytics 8.4.90。
- inspect-handoff所有hash通過；inspection_only輸出accepted=false是預期，不代表materialized驗收失敗。
- validate-pose-data：valid=true、source_untouched=true、6,647 images、4 patches、test=null。
- validate-screening-data：valid=true、COCO overlap=0、BBAT5 leakage=0、formal validation used=false。
- github-dataset：valid=true、6,647 unique images、13,294 image paths、13,294 label paths、symlinks=0、forbidden=0、test=null。
- snapshot tree SHA256：344ab62b9fb12b6f6a65c11abf96a987ec8415e53465bdc66bafcf336300cb4f。
- publication manifest SHA256：9b0a5380510faef8b4d8879cd6fbe292c2f166d1d74a71ed8ef10569f02364e9。

全部模型與自動測試都以CUDA_VISIBLE_DEVICES空值執行；只有nvidia-smi與systemctl唯讀查詢GPU。

## 困難與解法

1. bwrap loopback持續回報RTM_NEWADDR: Operation not permitted，apply_patch也無法讀檔。
   - 解法：經受控權限使用精確standard patch／機械區段替換，並逐次跑ruff與pytest。
   - 影響：.orig／.rej仍待使用者依finish-work明確清單授權刪除。
2. cache=false仍污染label來源。
   - 解法：移除兩個本輪可重建cache，新增train／validator共用runtime-cache adapter與回歸測試。
3. Pose路徑已提供但執行選擇未明。
   - 解法：不把路徑當opt-in；保持pending_user_decision，queue fail closed。
4. 文件宣稱GitHub snapshot存在但實際缺manifest。
   - 解法：先dry-plan與確認目的地不存在，再從immutable bbat5-v1重建並驗證。
5. 根main目前ahead 1、behind 46且有既有dirty／untracked內容。
   - 解法：沒有reset、pull、清理、commit或push，也未碰其他子專案權重與run。

## 未解事項或風險

- 仍需使用者選擇Pose=true joint Float20或Pose=false Detect-only Float20。
- Pose決定前不建立queue；目前沒有Float20 accuracy、GPU latency、VRAM或成本結論。
- 20%只適合初篩；接近或仍改善候選需共同追加seed與完整資料確認。
- C_best、C1～C3量化資格、Q1、Q2L、Q2與完整Float都維持pending_user_decision。
- 若上游正式改選J3，必須建立新handoff revision，不得覆寫本輪。
- 本輪未commit／push或操作GitHub Issue；發布時必須隔離遠端落後與無關dirty範圍。
- patch備份／reject與工具cache待cleanup授權；runtime-cache、CPU preflight、accepted intake與snapshot保留。

## J3 續行紀錄（2026-08-27，本輪目前狀態）

### 變更內容與原因

- 上游 final 已把 J3 升格為 accepted；本輪 parent 改鎖 `full35-final-j3-seed0-d67fb45c`。
- 正式 inference checkpoint 是 `/home/uxin/yolo/yolo_combine/final/full35/weights/combined/inference/best_joint.pt`，SHA256 為 `d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c`。
- J2 不刪除，保留為上游 rollback；architecture_2 的舊 J2 intake／CPU 證據移至本機 history，避免與 J3 結果混用。
- `EXPERIMENT_SPEC.md` 升為 2.3.0，SHA256 為 `572e10ef85037a4520083a0508688b7f88cd6fc7c11541530a1ef43644599216`。
- 新增 `handoffs/full35-j3-seed0/`；因上游新版未附舊式 CPU fresh-process report，改由本專案可重跑的 `full35-fresh-process` 嚴格載入並驗證 detect／pose／both。
- 使用者已明確選擇 `Pose=true`；正式 run 的兩個 Pose gate 均為 true，C0～C3 共用資料、seed、task ratio、optimizer steps 與 validation events。
- 依使用者指定刪除 C01 的 24 個 `.orig` 與 C02 的 4 個 `.rej`；後續 patch 若重建同類備份，會在最終驗證後再次精確清理。C03 cache 保留到最終驗證完成才刪除。

### 驗證方式與結果

- J3 release package verifier：407 files、4,493,133,705 bytes，全部 checksum 通過。
- J3 materialized accept：嚴格載入 1,238 tensors；C0 matched 1,238，missing／unexpected／shape mismatch 均為 0。
- C0～C3 CPU 640×640 detect／pose／both、finite synthetic loss／gradient、freeze、contract 與 strict reload 全部通過。
- 真實 COCO／BBAT5 batch：Detect 與 Pose loss／gradient 有限；`PoseLoss26`、RealNVP、RLELoss 均 active，source-adjacent cache 不存在。
- `config-check` 與 `float20-plan` 通過；pytest 為 71 passed、1 skipped，ruff 全部通過。skip 是 opt-in 的 2.35 GB integration，實際 materialized gates 已另跑。
- GPU queue 第一次通過三次 idle gate後立即失敗，沒有建立 checkpoint；state／log 已移至 `artifacts/queue/history/*attempt1-device-error*`。
- 根因是 memory probe 在 CUDA allocator 初始化前呼叫 `reset_peak_memory_stats(0)`；已移到 runtime/model 配置 GPU 之後並以 `runtime.device` 計量，另加入 CPU 回歸測試。
- 套件頂層改為延遲載入 torch，確保 queue grant 驗證完成後才公開 CUDA；公開 API 相容測試與 queue/device 測試共 16 passed。
- retry1 已再次通過三次 idle gate。C0 完整 macro probe 成功，physical microbatch 32、Pose batch 16，peak allocated 18,161 MiB、peak reserved 19,088 MiB，因此全矩陣固定使用 32，不需 OOM fallback 16。

### 困難與解法

1. 內建 `apply_patch` 持續受主機 bwrap loopback `RTM_NEWADDR` 限制。
   - 解法：經受控權限使用最小 standard patch 或可稽核機械替換，每次以 pytest／ruff 驗證。
2. 第一次 queue 暴露 CUDA allocator 呼叫順序錯誤。
   - 解法：根據 PyTorch `libc10_cuda` 的例外來源定位到未初始化 allocator，修正呼叫順序、加入回歸測試並完整重排，不跳過 idle gate。
3. 上游 J3 final 未沿用舊 J2 的 fresh-process 檔案布局。
   - 解法：不偽造缺檔；建立本地可重跑的 fresh-process CLI 與獨立 J3 handoff 證據。

### 未解事項或風險

- retry1 正在執行完整 C0～C3、20 epochs、Detect＋Pose Float20；完成前不宣稱 accuracy、C_best 或量化結論。
- Float20 完成後依規格停止；C_best、Q1、Q2L、Q2 與完整資料訓練仍等待使用者查看結果後決定。
- 20%是 train-only screen，不能取代後續完整資料與額外 seed 確認。
- 本輪未 commit、push、pull、reset 或清理根 repository 的無關 dirty 內容。


## J3 成本量測與結果匯出續記

### 變更內容與原因

- 新增 `float20-profile`：只有完整 `matrix-complete.json` 存在，且目前程序持有正式
  `yolo_combine.gpu_queue` grant 時，才會逐一嚴格載入 C0～C3 的 `best-joint-screening` EMA。
- 成本契約固定為同一 RTX 5090、batch=1、imgsz=640、AMP，分別量測 Detect／Pose／both 的
  GFLOPs、25 次 warmup、100 次同步 latency，以及 peak allocated／reserved VRAM；不把 PyTorch
  Float/AMP 數字冒充 INT8 deployment latency。
- 新增 `export-float20-results`：完整 matrix 與 profile 後，依實際 validation schema 自動輸出
  AP50、AP50-95、mAP50、mAP50-95、person、ball／bat、Macro／Micro F1、成本、lineage、CSV、
  中文報告、Pareto、圖表與 hash manifest。
- 匯出器固定 `c_best=null`、`selection_status=pending_user_decision`，不在本輪自動決定 C_best 或
  啟動量化；spec／training／candidate／parent hashes、候選順序、Pose、class name 或 F1 method
  漂移時 fail closed。
- README、RUNBOOK 與 results README 已補上 profile 必須再次排隊及正式匯出指令。
- 本段工作沒有改動 BBAT5／COCO 影像、labels、split 或 manifests，也沒有改動正在執行且 hash
  已鎖定的正式 training YAML。

### 驗證方式與結果

- 新增結果匯出單元測試，覆蓋 Detect/Pose AP、ball/bat、Macro/Micro F1、成本映射、class／method
  漂移，以及 profile 的 spec/training hash 與 C0～C3 順序。
- CLI、profiling 與結果匯出相關 CPU 測試：13 passed；ruff 與 Python compile 通過。
- 實際 C0 validation schema 已核對：Detect overall/person、Pose box/keypoints、per-class AP、固定
  threshold、Macro/Micro F1 與 official combined fitness 欄位均與匯出器一致。
- retry1 持續健康；截至本段紀錄時 C0 已完成 epoch 0～8，最新 COCO mAP50-95 約 0.603493、
  BBAT5 keypoint mAP50-95 約 0.981274，尚未完成 C0，因此這些只是中途觀察而非正式結論。

### 困難與解法

1. 內建 patch wrapper 仍因 `bwrap: loopback: Failed RTM_NEWADDR` 無法啟動。
   - 解法：改由同一受控主機環境中的精確 `apply_patch` 指令套用補丁，再逐次執行 ruff／pytest；
     未使用 reset、checkout 或覆蓋整個檔案。
2. 初次新增測試時兩個 unified diff 行數計算錯誤，曾產生 `.rej`，且新測試尾段一度被截斷。
   - 解法：先檢視實際行號，再以最小補丁補齊／修正；13 個目標測試全數通過。新產生的
     `.orig`／`.rej` 納入使用者已授權的 C01／C02 最終精確清理清單。

### 未解事項或風險

- C0～C3 共 80 個 epoch 仍在 queue 中執行；正式報告必須等 matrix、成本 profile 與最終驗證都完成。
- RLE 路徑已證明 active，但個別 training log 的第五個 loss component 可能為 0；不得在同一次
  immutable run 中途改 loss，最後需依完整紀錄如實揭露。
- C03 工具 cache 只能在所有 pytest／lint／資料驗證完成後刪除，否則驗證會重新建立。
- 新增並以 `bash -n` 驗證 fail-closed continuation 腳本；systemd unit
  `yolo-architecture2-full35-j3-float20-postprocess-seed0` 已啟動，目前只等待 retry1 完成。

## Float20 候選里程碑

- C0 已正式完成 20／20 epochs 與 1,860 optimizer macro steps；`complete.json` status 是
  `completed_screening`。
- C0 best detect=epoch 1、best joint=epoch 14、best pose research=epoch 17、best pose official=epoch 6。
- 固定 Pose keypoint F1 threshold 是 0.45545545545545546，來源是 C0 best-joint 的 train-only search-val。
- 四種 full resume 與 inference-only EMA best checkpoints 均存在；C0 有 20 份 validation metrics。
