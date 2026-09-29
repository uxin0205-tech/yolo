# 2026-08-26 Full35 JOINT 啟動、batch128 分片與 AMP 恢復

## 目的

在 canonical Full35 Pose P1→P2→P3、Float／Bit-True 八項獨立 baseline 與正式
preflight 均完成後，啟動 shared-trunk Detect＋Pose 的 J1→J2 AdamW 訓練；同時遵守
使用者指定的 Detect batch size 128、Pose batch size 16，且不得影響其他 GPU 工作。

## 啟動前基準

- 正式 Pose checkpoint：
  `variants/full35/artifacts/pose/p0-full35-p3-b32a4-e100max-seed0/weights/best.pt`。
- Bit-True 八項 baseline：
  `variants/full35/baselines/formal-gate-p3-seed0.json`。
- baseline 整體值：COCO box mAP50-95 `0.5064008856`、COCO person
  mAP50-95 `0.6261855612`、BBAT box mAP50-95 `0.6309636130`、BBAT pose
  mAP50-95 `0.9121605480`。
- 因 JOINT 尚未完成任何 epoch validation，這些仍是獨立模型基準，不是融合結果。

## 遇到的問題、證據與解法

### 1. 正式 trainer 啟動時缺少 `math`

- `resume2` 於 `2026-08-26 11:15:07 +08:00` 在計算
  `macros_per_epoch = math.ceil(...)` 時發生 `NameError`。
- 根因是 `src/yolo_combine/_formal_training_impl.py` 在拆分後漏掉 `import math`；runtime
  確認載入的就是 repository 內該檔，不是舊安裝副本。
- 先新增 CPU-only regression test，確認修正前在真實 `FormalJointTrainingSession.run()`
  seam 失敗，再補上 import；測試由紅轉綠。
- 失敗產物保留於
  `variants/full35/artifacts/fusion/failed-startups/full35-joint-adamw-seed0-20260826T111507-missing-math/`。

### 2. Detect physical forward 128 在 RTX 5090 OOM

- 空卡且無其他 compute process時，第一個 Detect forward 使用 physical128；程序已使用
  `30.58 GiB`，GPU僅餘`353.75 MiB`，再要求`400 MiB`而OOM。
- OOM發生在第一個正式 macro、Detect head forward，尚未產生任何epoch metric；它不是
  融合精度下降，也不能靠增加epoch解決。
- 使用者指定的batch size仍維持為**logical Detect batch 128**。執行層新增明確的
  **physical microbatch 64**：每個logical128由`64×2`組成；每macro仍是兩個logical
  Detect batch加一個Pose16，因此實際依序執行`Detect 4×64 + Pose 1×16`，每次optimizer
  update仍看256張Detect與16張Pose。
- `JointExperimentConfig`、preflight、resolved config與log均分別記錄logical batch、
  physical microbatch及每macro physical forward數，避免再把三者混稱。
- 這不是physical128的逐值等價替代：head BN與native loss normalization按physical64
  forward運作；shared BN running statistics仍依既定契約凍結。此差異必須由正式八項
  accuracy gate驗收。
- OOM產物保留於
  `variants/full35/artifacts/fusion/failed-startups/full35-joint-adamw-seed0-20260826T130052-physical128-oom/`。

### 3. 首個 macro 的 AMP 梯度 overflow

- physical64成功完成forward/backward，但原先generic檢查只回報
  `non-finite gradient before optimizer step`。
- 新增參數級fail-fast診斷：記錄AMP scale、retry次數、非有限參數數、NaN／正負Inf數，
  並把錯誤預覽限制為前32個參數，避免journal無限制膨脹。
- scale `65536` 時Detect head、Pose head與可訓練neck／MASF均出現Inf／NaN；回退8次到
  scale `256` 仍有96個參數非有限。這符合AMP動態範圍overflow，不是單一標籤或局部權重。
- 實作受控恢復：每次overflow都由GradScaler跳過optimizer step、降低scale、清除梯度並
  重算**完全相同**macro；不更新EMA、不前進scheduler、不遺失該批資料。只有全部梯度
  有限才clip、step與更新EMA；Full35 YAML最多允許16次，超過仍fail-fast。
- 第一個正式macro重算12次後在scale `16`成功；後續macro多為0次retry，遇到另一個
  overflow後scale降為`8`並繼續。`amp/scale`與`amp/overflow_retries`已寫入每個macro的
  JSONL／CSV資料。
- 診斷產物依序保留於`failed-startups/`中的
  `...-nonfinite-generic`、`...-scale65536-overflow-detailed`與
  `...-overflow-scale256-retry8`目錄。

## 程式與設定變更

- `src/yolo_combine/_formal_training_impl.py`
  - 補上`import math`。
  - scheduler與epoch runner按每macro四個physical Detect microbatch計算。
  - 將YAML的AMP retry上限傳入`MacroStepEngine`。
- `src/yolo_combine/_joint_config_impl.py`
  - 新增`detect_microbatch_size`、logical-to-physical推導與整除驗證。
  - 新增`amp_max_overflow_retries`及resolved provenance。
- `src/yolo_combine/formal_training.py`
  - Detect loader使用physical microbatch64；logical batch仍由config保留為128。
- `src/yolo_combine/joint_loss.py`
  - 新增參數級non-finite診斷、GradScaler skip/backoff同macro重算及AMP report欄位。
- `src/yolo_combine/_joint_trainer_impl.py`
  - 每macro記錄AMP scale與overflow retry次數。
- `variants/full35/configs/joint.yaml`
  - `detect_train_batch: 128`、`detect_train_microbatch: 64`、
    `detect_batches_per_macro: 2`、`amp_max_overflow_retries: 16`。
- Partial75只同步必要schema，仍`enabled: false`且未執行。
- 新增／更新 regression tests，並將兩個依賴「正式artifact尚不存在」的舊測試改為
  state-independent fixture；不再因P3／baseline真的完成而誤報失敗。

## 驗證方式與結果

- 最小啟動repro修正前：`NameError: name 'math' is not defined`；修正後`1 passed`。
- AMP recovery測試：人工製造一次overflow，確認第一次不step、scale回退、同macro重算後
  才成功；真正不可恢復的Inf仍會列參數名稱並拋錯。
- GPU隱藏完整測試：`113 passed, 3 skipped`；3項均為明確標記的CUDA integration test。
- 正式preflight：`ready=true`，resolved值明確顯示logical128、physical64、每macro四個
  Detect microbatch與AMP retry上限16。
- `resume7`於`2026-08-26 13:40:40 +08:00`啟動正式J1。記錄快照時已完成epoch0的
  macro68；每macro均為Detect`[64,64,64,64]`、Pose`[16]`，三個gradient group均存在。
- GPU快照：本run約使用`22.1 GiB`，尚餘約`9.5 GiB`，無其他compute process。

### J1 epoch 1 正式 validation（artifact `epoch-0000`）

- Float與Bit-True結果近乎一致；以下正式gate以Bit-True為準。
- Bit-True八項`mAP50-95` baseline → joint（delta）：
  - COCO overall：`0.506401 → 0.425072`（`-0.081329`）。
  - COCO person：`0.626186 → 0.559015`（`-0.067170`）。
  - BBAT box overall：`0.630964 → 0.384924`（`-0.246040`）。
  - ball box：`0.510747 → 0.292680`（`-0.218067`）。
  - bat box：`0.751180 → 0.477168`（`-0.274013`）。
  - BBAT pose overall：`0.912161 → 0.729818`（`-0.182343`）。
  - ball pose：`0.876263 → 0.680608`（`-0.195655`）。
  - bat pose：`0.948058 → 0.779027`（`-0.169031`）。
- person通過`-0.08`；COCO overall超標`0.001329`，其餘BBAT四個class-task與兩個overall尚未通過。
- 因hard gate未全通過，只保存`best_detect.pt`、`best_pose.pt`與`last.pt`，未建立`best_joint.pt`；run已正常進入J1 epoch 2。

## 困難與解法

- 困難：內建`apply_patch`因主機`bwrap: loopback: Failed RTM_NEWADDR`無法更新既有檔案。
- 解法：改用標準unified `patch`套用相同最小diff；過程產生的`.orig`／`.rej`均逐一
  稽核並刪除。這是工具環境限制，不是repository程式錯誤。

## 未解事項與風險

- J1 epoch1已顯示初期shared-trunk／Pose head失配且多項未通過gate，但尚不能代表J1／J2最終效果；
  先完成固定5 epochs的J1，再依趨勢決定是否需調整，而不是現在直接增加epoch。
- 第一macro的Pose shared-gradient norm明顯大於Detect；雖已clip且後續loss下降，仍須看
  epoch validation與gradient cosine趨勢，不能只看training loss。
- J1固定5 epochs、J2最多40／patience17；只有J2結束時validation仍持續改善，才討論延長。
  J3目前關閉，不會因任何單項gate失敗而自動啟用。
- physical microbatch64與physical128不逐值等價；最終接受條件仍是五項／八項同口徑
  Bit-True mAP50-95逐項下降不得超過0.08，而不是joint平均分數掩蓋單項退化。
