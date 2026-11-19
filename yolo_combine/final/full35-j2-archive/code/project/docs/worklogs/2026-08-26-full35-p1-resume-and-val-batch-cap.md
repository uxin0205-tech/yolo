# 2026-08-26 Full35 P1 原生續跑與驗證 batch 上限

## 目的

檢查 GPU 是否空閒，釐清凌晨自動管線的停止原因，並在不改變正式 train physical
batch128、模型架構、資料集與 optimizer 的前提下，安全接續 canonical BBAT5 Pose P1。

## 凌晨管線結果

- 30% 診斷 P1 已完成2 epochs；它只作流程與收斂方向檢查，不可作正式 baseline。
- 正式 P1 原始執行完成 epoch1–6；epoch7 train loop 已跑完，但 validation OOM，因此
  `results.csv`沒有 epoch7，`last.pt`保存的是 epoch6 結束狀態。
- 原始 `last.pt` 稽核結果：checkpoint `epoch=5`（zero-based）、EMA、optimizer、AMP
  scaler皆存在，`updates=282`，設定仍是17 epochs、physical batch128。
- 中斷前 epoch6：Box mAP50-95=0.28609、Pose mAP50-95=0.55128。

## OOM 原因與修正

Ultralytics 8.4.90 的原生 trainer 會把非 OBB/semantic validation batch 設為
`train_batch * 2`。因此正式 P1 實際是 train batch128、Pose validation batch256；train
本身可跑，OOM發生在 validation，而不是 physical train batch128。

新增以下 fail-closed 修正：

1. `MaterializedPoseTrainer.get_dataloader()`只在`mode="val"`時使用明確的
   `validation_batch_size`；train仍收到原本128。
2. CLI新增`--validation-batch`，正式 Full35 目前固定16，與 JOINT Pose validation
   設定一致。這不改影像、標註、模型或 AP 計算口徑。
3. CLI新增`--resume-checkpoint`；續跑時使用 checkpoint 的 EMA、optimizer、scaler、
   epoch與 progressive-loss狀態，不以 best checkpoint另開新 optimizer。
4. materialized Full35 graph會對 checkpoint EMA 做 state name與shape完整比對，再
   `strict=True`載入；缺少、額外或shape不符都立即報錯。
5. 新增`pose-stage-complete.json`。管線不再只看到`best.pt`、`last.pt`、`results.csv`
   就誤判 interrupted run 已完成；只有 trainer 正常返回後才寫 completion marker。
6. resume輸入的SHA-256在`last.pt`被後續 epoch覆寫前先擷取，避免 provenance 指向
   已變更的檔案內容。

## 驗證

- 針對 validation cap、train batch不變、EMA strict load與不相容 graph fail-fast：
  14項相關測試全數通過。
- 完整 repository CPU 測試：`111 passed`，耗時16.06秒。
- 真實 Full35 checkpoint CPU相容檢查：998個state keys、23,607,725 parameters；
  trunk transfer完整且 resume EMA strict load成功。
- 2026-08-26 09:13 CST 啟動前 GPU：RTX5090 free31,759 MiB、utilization 0%、
  compute process 0。
- 續跑 log 明確顯示：`Resuming ... from epoch 7 to 17 total epochs`。
- train batch128 執行時 GPU 約22.9 GiB；validation batch16已連續完成，沒有再 OOM。
- P1最後完成17/17 epochs：Box mAP50-95=0.41296、Pose mAP50-95=0.69012，
  高於中斷前 epoch6；`pose-stage-complete.json`已建立。

## 持久執行與接續

- P1 resume service：`yolo-full35-p1-resume-seed0.service`。
- 後續 watcher：`yolo-full35-seed0-pipeline-resume1.service`。
- watcher使用`systemctl --wait`等待 P1 正常結束；成功才會執行既有 Full35 pipeline，
  接著跑 P2→P3→Float/Bit-True standalone baseline→preflight→JOINT J1/J2→final
  validation。P1失敗時不會誤跑後續。
- 各 GPU stage仍先檢查空閒條件，不會與另一個 GPU 工作同時啟動。

## 困難與解法

1. 困難：已存在 partial P1 目錄，直接重跑可能覆寫或被 Ultralytics自動改名。
   解法：使用同一個`last.pt`原生 resume，並以 completion marker區分 interrupted與完成。
2. 困難：custom Full35 checkpoint需要`yolo_attention` pickle module。
   解法：沿用 variant既有 source activation與 exact tiled XNOR安裝入口後再載入。
3. 困難：單靠checkpoint檔案存在無法判斷 stage 是否正常結束。
   解法：只有正常返回才原子流程地建立 completion evidence，pipeline將它列為必要檔。

## 未解事項與風險

- P2/P3 memory gate後續證實128/64不可行，已另記於
  [P2/P3顯存診斷](2026-08-26-p2-p3-memory-gates.md)。
- P3、八項 baseline與 JOINT 尚未完成，因此0.08 hard gate目前仍不能判定。
- user-level transient service不跨主機重開機；主機重啟後需依 checkpoint重新排程。

除上述風險外，無其他未解困難。
