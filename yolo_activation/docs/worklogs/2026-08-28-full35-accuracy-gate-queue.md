# 2026-08-28 Full35 Activation Gate 與 Serial Queue

## 任務與範圍

接續完成 Full35 activation 實驗，處理 10-epoch SiLU sham control 的結果，修正 activation recovery
使用的 accuracy gate，並把 qSiLU、uniform recovery、region recovery、bounded mixed search 與
conditional finalists 排成單 GPU serial queue。所有訓練與驗證仍使用完整 COCO2017 與 Canonical
BBAT5 v1；沒有建立 30% 資料、重新抽樣、重新切分或改動影像與標註。

## 變更內容與原因

1. `short-recovery-v2-lr01-uniform-silu-seed1` 已正常完成 10 epochs、4,630 macro steps，AMP overflow
   為 0。best joint 位於最後的 epoch 9：COCO `0.496786`、COCO person `0.620225`、BBAT box
   `0.623433`、BBAT pose `0.901985`，joint score `0.708883`。
2. 再稽核預註冊規格後，確認每個候選本來就應從 accepted inference EMA 乾淨載入並重建
   optimizer/scheduler/EMA/RNG；SiLU run 是 stage-matched fresh-optimizer control，而不是 exact
   full-resume。修正前一日紀錄中把這項機制稱為 loader bug 的誤述。
3. 找到真正的 gate 問題：`_RecoveryConfigProxy.preflight()` 原先沿用 COMBINE 的 joint 前 standalone
   baseline 與 `0.08` fusion gate。新增 accepted baseline contract path/hash、嚴格八項 metric 載入，
   runtime 改以 delivered `best_joint` 為基準。
4. 依 SiLU control 的八項自然漂移凍結 activation hard gate 為 `0.015`。control 最差項為 ball box
   約 `-0.012341`，因此保留約 `0.00266` 漂移餘量；此 gate 只負責淘汰，候選仍須另外和同 seed、
   同 budget SiLU control 比較。
5. resolved config 現在明確記錄 fresh initialization 來源，以及 full-resume 只作 lineage／緊急 exact
   resume 參照，避免再次把兩種實驗語意混淆。
6. 新增 `training/full35/experiment-queue.yaml`，靜態排定 19 個 jobs：qSiLU zero-shot、4 個 uniform
   recovery、8 個 region recovery 與 6 個 preregistered mixed policies；另保留 2 個 adaptive slots、
   最多 2 個 finalists、seed 2、延長、PTQ/QAT/board validation 的 conditional tail。
7. 新增 `scripts/full35_queue.py`。runner 強制單 GPU／單 job、依賴與 gate 傳播，verbose stdout/stderr
   寫入 job log，console 只輸出單行 JSON；queue state 原子寫入，並以 file lock 防止重複 runner。

## 驗證方式與結果

- 新增 `tests/test_full35_recovery_contract.py`：修復前因 config 沒有 accepted baseline contract 而呈紅；修復後確認 runtime baseline、`0.015` gate、fresh-control 與 full-resume 角色全部一致。
- `tests/test_full35_queue.py` 確認 19 jobs、唯一 run name、topological dependencies、單 GPU、mixed policy CLI 與 qSiLU 實體 microbatch `16`；另外以 monkeypatch 隔離正式 queue state，避免實驗完成後讓初始狀態測試失效。
- `tests/test_full35_frozen_budget.py` 確認 recipe、policy plan 與 queue 的 gate 值一致，pilot 狀態已從 pending 改為 frozen。
- 完整 suite 為 `51 passed`；另有 4 個既有 ONNX legacy exporter deprecation warnings，沒有功能失敗。Ruff 與格式檢查通過。
- 完整 hash、資料與環境 preflight 為 `ready: true`、`blockers: []`；確認 accepted `best_joint.pt`、完整 COCO2017 與不可變 BBAT5 v1 正式入口。
- qSiLU full Float／BitTrue zero-shot 完成並通過八項 `0.015` gate：COCO `0.495272`、COCO person `0.618594`、BBAT box `0.622102`、BBAT pose `0.897445`，最差 delta `-0.013091`。
- qSiLU short recovery 首次以 Detect physical microbatch `32` 在第一個 batch OOM；改為 `16` 後已通過第一個 macro，GPU 使用約 `18,554 MiB`、overflow `0`，未再 OOM。

## 困難與解法

- `apply_patch` 可新增檔案，但修改既有檔案仍被環境的 bwrap loopback 錯誤阻擋。先嘗試 `apply_patch` 並保留失敗證據後，改用每次舊內容必須唯一命中的精確 Perl 替換；每批修改後立即跑 Ruff、pytest 與 diff check。
- 一開始把 full-resume 未用解讀成 bug；重新讀取正式研究計畫與 README 後，確認 fresh optimizer 是公平比較契約。真正問題是 runtime gate 仍指向 standalone baseline，已在正確 seam 修復。
- policy YAML 一度因文字轉換造成 UTF-8 損壞；完整測試立即攔截，已用正確 UTF-8 重建該小節並重新通過完整 suite。
- qSiLU 的 piecewise quadratic autograd 中間值比 SiLU 多，physical microbatch `32` 使 31.35 GiB GPU 在第一個 Detect batch OOM。新增先紅後綠測試並只把 qSiLU recovery 的實體 microbatch 降至 `16`；logical Detect batch `128`、每 macro `256` 個 Detect samples、optimizer step 與 loss 權重完全不變。失敗 run 的 56 MB 產物移至 `artifacts/runs/full35/quarantine/short-recovery-v2-lr01-uniform-qsilu-pq-seed1-oom-mb32`，可還原、未刪除。

## 目前執行狀態（2026-08-28 查核）

- Queue runner 與 qSiLU train process 均存活；GPU 使用約 `21,824 MiB`。
- qSiLU short recovery 最新查核已進入最後的 epoch 9、global macro step `4353`（epoch 內 `186/463`），AMP overflow `0`；最新完成的 BitTrue 為 epoch 8。
- epoch 1 BitTrue：COCO `0.497053`、BBAT box `0.623826`、BBAT pose `0.900909`。
- Queue state 為 `1 completed / 18 pending / 0 blocked`；訓練中的 job 只有產生最終 epoch 9 metrics 後才會從 pending 改為 completed。
- 快速健康迴圈確認 runner/train process 存活、events 三分鐘內更新，且最新 log 無 OOM、Traceback 或 job failure，結果為 `health=green`；因此不需中斷或重啟。
- epoch 0／1 八項 hard gate 通過，epoch 2／3 暫時未過；epoch 3 最差為 BBAT ball box，相對 accepted baseline 為 `-0.018797`。但 qSiLU 相對同 seed、同 epoch SiLU control 的八項平均差僅 `-0.000733`，最差單項 `-0.001821`，因此目前判定為 fresh recovery 共同漂移而非 qSiLU 崩壞；依凍結規格繼續跑滿 10 epochs，不中途改 gate 或超參數。
- epoch 6 最差仍為 BBAT ball box，相對 accepted baseline `-0.016196`，暫未過 `0.015` gate；但相對同 epoch SiLU control 的八項平均為 `+0.001278`，最差單項僅 `-0.001176`，再次支持共同 recovery 漂移的判斷。
- epoch 7 已重新通過八項 hard gate；最差 BBAT ball box 相對 accepted baseline 改善為 `-0.011551`，相對同 epoch SiLU control 的八項平均為 `+0.000610`。
- epoch 8 最差 BBAT ball box 相對 accepted baseline 為 `-0.015123`，只比 hard gate 多跌 `0.000123`；屬臨界波動，最終判定保留至 epoch 9 完整 BitTrue 驗證。
- 本次困難：第一次唯讀列印 activation 清單時發生巢狀 f-string 引號錯誤；已改用格式化字串重新執行，未影響 GPU 訓練、queue 或產物。

## 2026-08-29 Queue 節點更新

- `short-recovery-uniform-poly-shift` 已依預註冊配方完成完整 COCO2017 + Canonical BBAT5 v1、10 epochs；沒有重新抽樣、切分或改動影像與標註。Detect logical batch 仍為 `128`，physical microbatch `16` 只用於避開已重現的 polynomial activation OOM。
- final BitTrue gate 未通過。相對 accepted baseline 的最差 delta 為 BBAT ball box `-0.030138`；另外 BBAT box `-0.016640` 與 BBAT ball pose `-0.020129` 也超過 `0.015` hard gate。COCO box `-0.005477`、COCO person `-0.002263` 仍在 gate 內。
- 這是 accuracy 結果，不是執行錯誤；因此沒有中途改 gate、補訓或調參。依 frozen dependency graph，8 個 region recovery 與 6 個 mixed policy 共 14 jobs 已自動標記 `blocked`，避免在 uniform prerequisite 未過時浪費 GPU。
- Queue state 更新為 `4 completed / 1 pending / 14 blocked`，runner 已自動啟動唯一仍可執行的 `short-recovery-uniform-poly-quality`；GPU 仍維持單 job serial 執行。
- 驗證方式：核對 final `gate` event（step 9）、`queue-state.json` 的 job gate/deltas 與 runner 的 `job_complete`／`job_start` 事件，三者一致。
- 困難與解法：低 token 監測 wrapper 曾兩次出現單次 JavaScript 語法錯誤；直接重試即恢復，統一 exec session、訓練程序、checkpoint 與 queue state 均未被中斷或修改。
- 未解事項／風險：`poly_quality` 尚未完成；conditional adaptive/finalist tail 必須依所有可執行候選的正式結果重新套用 frozen selection rule。由於 region/mixed 已被 prerequisite gate 阻擋，不得把缺失結果當成通過或自行放寬 gate。
- `poly_quality` 已完成使用者視角 epoch `1/10`（事件 step `0`）的完整 BitTrue gate，八項均通過；最差 delta 為 `-0.007395`，仍在 `-0.015` hard gate 內。之後已自動進入 epoch `2/10`，查核點為 macro `7/463`、AMP overflow `0`。
- `poly_quality` epoch `2/10`（事件 step `1`）完整 BitTrue gate 只有 BBAT ball box 未過，delta `-0.016253`，比 `-0.015` 門檻多跌 `0.001253`；其餘七項均在 gate 內。這是臨界 accuracy 波動而非 runtime failure，未調參或放寬 gate，已依凍結配方進入 epoch `3/10`；查核點 macro `10/463`、AMP overflow `0`。
- `poly_quality` epoch `3/10`（事件 step `2`）完整 BitTrue gate 八項全數通過，最差 delta `-0.012746`；已正常進入 epoch `4/10`，查核點 macro `17/463`、joint loss `1.487602`、AMP overflow `0`。驗證方式為核對完整 gate event 與下一 epoch macro；困難及解法：無。未解事項／風險：仍須完成其餘 7 個 epochs 與 final gate。
- `poly_quality` epoch `4/10`（事件 step `3`）完整 BitTrue gate 八項全數通過，最差 delta `-0.009535`；已正常進入 epoch `5/10`，查核點 macro `14/463`、joint loss `1.626861`、AMP overflow `0`。驗證方式為核對完整 gate event 與下一 epoch macro；困難及解法：無。未解事項／風險：仍須完成其餘 6 個 epochs 與 final gate。
- `poly_quality` epoch `5/10`（事件 step `4`）完整 BitTrue gate 只有 BBAT ball box 未過，delta `-0.019337`；其餘七項均在 gate 內。這是 accuracy 結果而非 runtime failure，未調參或放寬 gate，已依凍結配方進入 epoch `6/10`；查核點 macro `13/463`、joint loss `1.693661`、AMP overflow `0`。驗證方式為核對八項 gate event 與下一 epoch macro；困難及解法：無。未解事項／風險：仍須完成其餘 5 個 epochs 與 final gate。
- `poly_quality` epoch `6/10`（事件 step `5`）完整 BitTrue gate 只有 BBAT ball box 未過，delta `-0.021982`；其餘七項均在 gate 內。這是 accuracy 漂移而非 runtime failure，未調參或放寬 gate，已依凍結配方進入 epoch `7/10`；查核點 macro `18/463`、joint loss `1.661874`、AMP overflow `0`。驗證方式為核對八項 gate event 與下一 epoch macro。困難及解法：低 token 監測 wrapper 單次出現 JavaScript 語法錯誤，重試相同只讀輪詢即恢復，runner、訓練、checkpoint 與 queue 均未受影響。未解事項／風險：仍須完成其餘 4 個 epochs 與 final gate。
- `poly_quality` epoch `7/10`（事件 step `6`）完整 BitTrue gate 只有 BBAT ball box 未過，delta `-0.021354`；其餘七項均在 gate 內。未調參或放寬 gate，已依凍結配方進入 epoch `8/10`；查核點 macro `18/463`、joint loss `1.750367`、AMP overflow `0`。驗證方式為核對八項 gate event 與下一 epoch macro；困難及解法：無。未解事項／風險：仍須完成其餘 3 個 epochs 與 final gate。
- `poly_quality` epoch `8/10`（事件 step `7`）完整 BitTrue gate 只有 BBAT ball box 未過，delta 改善為 `-0.018976`；其餘七項均在 gate 內。未調參或放寬 gate，已依凍結配方進入 epoch `9/10`；查核點 macro `7/463`、joint loss `1.740039`、AMP overflow `0`。驗證方式為核對八項 gate event 與下一 epoch macro；困難及解法：無。未解事項／風險：仍須完成其餘 2 個 epochs 與 final gate。
- `poly_quality` epoch `9/10`（事件 step `8`）完整 BitTrue gate 只有 BBAT ball box 未過，delta 為 `-0.021394`；其餘七項均在 `-0.015` gate 內。未調參或放寬 gate，已依凍結配方進入 epoch `10/10`；GPU process 約使用 `26.1 GB`，AMP overflow `0`。驗證方式為核對八項 gate event、下一 epoch macro 與 GPU process；困難及解法：無。未解事項／風險：仍須完成最後 1 個 epoch 與 final gate。
- `poly_quality` epoch `10/10`（事件 step `9`）完整訓練與 BitTrue final gate 已完成；COCO box `0.496755`、BBAT box `0.617772`、BBAT pose `0.898036`。唯一未過項為 BBAT ball box `0.486467`，相對 accepted baseline delta `-0.020970`；其餘七項均在 `-0.015` gate 內，因此候選 final gate 判定為失敗。靜態 queue 已正常結束為 `5 completed / 0 pending / 14 blocked`；14 個 region／mixed jobs 因 `poly_shift` prerequisite gate 未過而依凍結規則阻擋。驗證方式為交叉核對 epoch-0009 BitTrue metrics、gate event、`queue-state.json` 與 runner 的 `queue_static_complete`。困難及解法：無。未解事項／風險：須依 frozen finalist rule 執行唯一通過 recovery gate 的 qSiLU 與 matched SiLU control，並判定條件式後段可執行範圍。
- Frozen selector 只有 `qsilu_pq` 通過 uniform recovery gate，因此它同時承擔 accuracy-extreme 與 hardware-extreme 角色；Hardswish、`poly_shift`、`poly_quality` 依正式 gate 排除。新增 `finalist-seed1-queue.yaml`，先跑 matched SiLU control、再跑 qSiLU；兩者皆為 20 epochs、seed 1、patience 5、warm-up 3、原 J3 LR，Detect logical batch 128，physical microbatch 分別為 32／16。
- Finalist 可能 early-stop，故 queue runner 新增 `metric_selection: best_joint`，並要求 `activation-experiment.json` completion marker 後才從所有正式 gate 選 `score/best_joint` 最高的已評估 epoch 取 BitTrue gate，避免把中途結果誤判為 job 已完成。驗證方式：queue 目標測試 9 passed、完整測試 57 passed，另有 4 個既有 ONNX deprecation warnings；preflight ready、`git diff --check` 通過、磁碟剩餘約 1.1 TB、GPU 啟動前空閒。困難及解法：既有檔案 `apply_patch` 仍受 loopback sandbox 阻擋，README 的第一次 fallback 又遇到 Perl UTF-8／delimiter 錯誤；未寫入即終止，改用 `-Mutf8 -CSDA` 並跳脫 delimiter 後完成精確替換。未解事項／風險：seed-1 finalist queue 已啟動，仍須完成兩個 jobs、best_joint gates、extension 判定及可執行的後段。
- Matched SiLU finalist epoch `1/20`（事件 step `0`）完成，最差 delta 為 BBAT ball box `-0.023343`，故該 epoch gate 未過；已正常進入 epoch `2/20`，AMP overflow `0`。此 gate 的 `selected` 未含 `best_joint`，因 trainer 是與初始 accepted best 比較，而不是在已評估 epochs 間排名；因此把 queue selector 修正為 completion 後取最大 `score/best_joint`，加入「較高分 epoch 即使沒有 selected 標記仍被選中」測試，9 個 queue 測試通過。困難及解法：修正不需停止訓練。未解事項／風險：仍須完成 matched control 的其餘 epochs 與 qSiLU finalist。
- Matched SiLU finalist epoch `2/20`（事件 step `1`）完成；最差 delta 改善為 `-0.018193`，但 BBAT ball box 與 ball pose 仍未過。`score/best_joint=0.704097` 低於 epoch 1 的 `0.705621`，故目前已評估 best epoch 仍為 epoch 1；已進入 epoch `3/20`，AMP overflow `0`。驗證方式為核對完整 gate event 與下一 epoch macro；困難及解法：無。未解事項／風險：仍須完成其餘 matched control epochs 與 qSiLU finalist。
- Matched SiLU finalist epoch `3/20`（事件 step `2`）完成；BBAT overall box 與 ball box 未過，最差 delta `-0.025310`，`score/best_joint=0.702128`，故 best epoch 仍為 epoch 1。已正常進入 epoch `4/20`，AMP overflow `0`。驗證方式為核對完整 gate event 與下一 epoch macro；困難及解法：無。未解事項／風險：仍須完成其餘 matched control epochs 與 qSiLU finalist。
- Matched SiLU finalist epoch `4/20`（事件 step `3`）完成；BBAT overall box、ball box 與 ball pose 未過，最差 delta `-0.023363`，`score/best_joint=0.702148`，故 best epoch 仍為 epoch 1。已正常進入 epoch `5/20`，AMP overflow `0`。驗證方式為核對完整 gate event 與下一 epoch macro；困難及解法：無。未解事項／風險：仍須完成其餘 matched control epochs 與 qSiLU finalist。
- Matched SiLU finalist epoch `5/20`（事件 step `4`）完成；BBAT overall box、ball box 與 ball pose 未過，最差 delta `-0.021291`，`score/best_joint=0.700654`，故 best epoch 仍為 epoch 1。已正常進入 epoch `6/20`，AMP overflow `0`；`patience=5` 仍由 frozen finalist 配方控制，不在中途修改。驗證方式為核對完整 gate event 與下一 epoch macro；困難及解法：無。未解事項／風險：若後續沒有改善，trainer 可能依 patience 正式 early-stop；仍須完成 matched control 結算與 qSiLU finalist。
- 本輪沒有建立或執行「batch size 16」比較實驗；`16` 僅是 polynomial activation 的訓練 physical microbatch，Detect logical batch 保持 `128`，正式 validation 仍沿用 COMBINE 的 Detect `32`／Pose `16` 配置以維持 baseline 可比性。
- 驗證方式：以 `events.jsonl` 的 epoch、gate 與下一 epoch macro 三類事件交叉查核；結果一致。困難及解法：首次對尚無 gate 的事件做 compact `jq` 時遇到 null 欄位錯誤，改用存在性判斷後完成唯讀查核；訓練未受影響。未解事項／風險：仍須完成其餘 9 個 epochs 與 final gate。

## 2026-08-29 Finalist selector 修復與接續訓練

- Matched SiLU finalist epoch `6/20`（事件 step `5`）八項 gate 通過，最差 delta `-0.012872`、`score/best_joint=0.705348`；trainer 因 `patience=5` 正式 early-stop，completion marker、`selected=best_joint` 與 queue `job_complete` 一致。Runner 隨後自動啟動 `finalist-seed1-qsilu-pq`，未中斷 GPU queue。
- 診斷 queue selector 時，以真實六個 gate event 確認 epoch 1 雖有較高 raw score，但 accuracy gate 失敗；epoch 6 才是可交付的 gate-passing best joint。先新增紅燈測試重現錯選 epoch 1，再把規則修正為「gate 通過優先，其內取最高 score；若整個完成 run 全數未過，才 fallback 到所有 epoch 的最高 score」。修後 static／finalist queue 測試 `10 passed`、`py_compile` 與 `git diff --check` 全部通過。
- 困難與解法：沙箱的 bwrap loopback namespace 持續無法建立，`apply_patch` 與第一次測試在沙箱內失敗；依權限流程在沙箱外用精確替換修正並執行回歸測試。長存活 runner 載入的是啟動時舊程式，但其 trainer-selected 邏輯已正確結算 SiLU；磁碟上的新 selector 已由獨立 `status` 重算為 `1 completed / 1 pending / 0 blocked`。未解事項／風險：qSiLU 若沒有任何 gate-passing epoch，舊 runner 可能只回報 incomplete；屆時可用已修正程式重新結算 completion marker，不需重訓。
- 本輪沒有建立或執行「batch size 16」比較實驗。qSiLU 的 `16` 只是在 physical batch `32` 已重現 OOM 後採用的訓練 microbatch；Detect logical batch 仍為 `128`。正式 validation 沿用 Detect `32`／Pose `16`，validation batch 只控制顯存與吞吐，不作 activation 品質變因。
- 最新低 token 健康查核：qSiLU finalist runner process 存活，epoch 1 已進行至少 `65` 個 macro-step、joint loss `1.508107`、尚無 gate event；沒有改資料集、split、gate 或超參數。

## 2026-08-29 依使用者指示停止 qSiLU finalist

- 變更內容與原因：使用者判定 activation 與 quantization 強耦合，要求先停止 activation 訓練交由量化工作列分析。於 qSiLU seed-1 finalist 完成 epoch 1、進入 epoch 2 macro `106/463` 後，向唯一 queue PTY 發送一次 `SIGINT`；queue 以 exit code `130` 結束，未重啟、未刪除 checkpoint、events 或 validation 產物。
- 驗證方式與結果：停止後 `pgrep` 未找到 `full35_queue.py`、`full35_activation.py train` 或 `train_full35_joint.py`，`nvidia-smi` 亦無 compute process；qSiLU run 沒有 `activation-experiment.json` completion marker，finalist queue 正確維持 `1 completed / 1 pending / 0 blocked`，未把人工中止冒充完成。
- 停止前量測：epoch 1 BitTrue 為 COCO box `0.494248`、BBAT box `0.620522`、BBAT pose `0.897755`；八項 gate 只有 BBAT ball box 未過，delta `-0.015966`，比 `-0.015` 門檻多跌 `0.000966`。此為 provisional 中途結果，不作 finalist winner 或 immutable quantization intake。
- 困難與解法：無執行錯誤；queue traceback 是人工 `KeyboardInterrupt` 的預期結果。未解事項／風險：量化工作列可先使用 accepted SiLU、已完成 qSiLU zero-shot／10-epoch recovery 與函數／Bit-True 證據做耦合分析，但不得把未完成 finalist checkpoint 當成正式父權重；是否恢復 finalist 由量化分析後再決定。

## 2026-08-29 qSiLU physical batch 128 OOM probe

- 變更內容與原因：依使用者要求，用隔離 run `probe-oom-qsilu-pq-mb128-20260829` 實測 Full35、640×640、qSiLU、Detect physical batch `128`，確認不使用梯度累積是否可行；外層設 5 分鐘 timeout，沒有覆蓋正式 run。
- 驗證方式與結果：GPU 啟動前可用 `31,734 MiB`。Probe 在第一個 Detect forward、`ShiftPiecewiseQuadraticSiLU.forward` 的 `h1 = a1 * u2 + b1 * u + c1` 直接 `torch.OutOfMemoryError`，尚未進 backward；當時程序使用 `30.66 GiB`，PyTorch allocated `29.60 GiB`、reserved-but-unallocated `466.12 MiB`、只剩 `326.31 MiB`，下一個張量仍需 `400 MiB`。
- 結束查核：probe process 已退出，GPU 恢復可用 `31,737 MiB`；沒有 `activation-experiment.json` completion marker。隔離產物約 `56M`，保留作 OOM 證據，未刪除。
- 困難與解法：無額外執行故障；OOM 即本測試要驗證的結果。未解事項／風險：這證明目前 eager qSiLU 實作與 31.35 GiB 單 GPU 無法 physical batch 128；未來 fused kernel／custom autograd 可能降低記憶體，但必須另行實測，不能由本結果推定可達 128。

## 2026-08-29 qSiLU physical batch 64／32 OOM probes

- 變更內容與原因：依使用者要求，在與 batch 128 相同的 Full35、640×640、qSiLU、short-recovery 條件下，以唯一 run name 分別實測 physical batch `64` 與 `32`；兩者均設 5 分鐘安全 timeout，不恢復正式 queue。
- Batch 64 結果：第一個 Detect forward 在 qSiLU tail selection `torch.where` OOM，尚未進 backward；程序使用 `30.59 GiB`、只剩 `400.31 MiB`，下一個張量需 `400 MiB`。
- Batch 32 結果：第一個 Detect forward 已走到 Detect one-to-one head，但在 qSiLU `h2 = a2 * u2 + b2 * u + c2` OOM，仍未進 backward；程序使用 `30.92 GiB`、只剩 `62.31 MiB`，下一個張量需 `100 MiB`。這重現先前正式 qSiLU batch 32 的 OOM。
- 驗證方式與結果：兩個 probe process 均已退出，GPU 恢復可用 `31,737 MiB`；兩個 run 都沒有 completion marker，各約 `56M`，保留作失敗證據，未刪除。困難與解法：無額外執行故障；未解事項／風險：目前單 GPU 安全值仍是 physical batch `16`，logical batch `128` 必須靠梯度累積維持。

## 未解事項與風險

- qSiLU full zero-shot 與 short recovery 均已完成並通過 final gate；Hardswish 與 poly_shift 已完成但 final gate 未過。
- 靜態 queue 現為 4 completed、1 pending、14 blocked；runner 正在 serial 執行最後可執行的 poly_quality recovery。
- poly_shift prerequisite 未過後，region/mixed 結果不會產生；adaptive slots 不可用缺失結果補造。poly_quality 完成後，只能對已有正式 gate 證據的候選套用 frozen finalist rule。
- Target FPGA/ASIC profile 尚未提供；queue 只能完成 accuracy／graph 實驗，不能代替 synthesis、
  latency、energy 與 resource validation。
