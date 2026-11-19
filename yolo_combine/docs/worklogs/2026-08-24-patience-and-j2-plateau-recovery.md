# 2026-08-24 Patience 與 J2 plateau recovery

## 目標與決策

依使用者最新決策，正式 patience 鎖定為：

- standalone Pose P1：10（最多17 epochs）。
- standalone Pose P2：12（最多22 epochs）。
- standalone Pose P3：20（最多100 epochs，維持既有設定）。
- JOINT J1：0，固定跑滿5 epochs。
- JOINT J2：17，最多40 epochs。
- optional J3：5，維持預設停用。

P1/P2 已有逐 epoch cosine learning-rate schedule，而且 stage 較短，因此只修改原生
Ultralytics early-stop patience，不再疊一套 plateau scheduler，也不在 run 中途改
MuSGD momentum。

J2 新增一次性的 deterministic recovery：以每 epoch 的 Bit-True unconditional joint
score 為監控值；只有高於歷史最佳至少 `1e-4` 才算改善。連續8個 validation epoch
未改善時，下一 epoch 起將所有 semantic optimizer group 的當期 cosine LR 乘0.5；
整個 J2 最多一次。stale counter 到17便早停。AdamW beta／MuSGD momentum與 optimizer
種類全程不變，避免同一 run 同時改太多控制變因。

## 變更內容與原因

1. 新增 `src/yolo_combine/plateau.py`：封裝 policy、decision、stale counter、一次性
   recovery 與嚴格 state restore。這讓正式 training loop 只需呼叫 `observe_metric()`，
   不需知道內部狀態機細節。
2. 擴充 `StageWarmupCosineScheduler`：原 cosine LR 仍是 owner，plateau只套用持久的
   multiplier；scheduler schema升為2，checkpoint會保存最佳分數、stale epochs、降 LR
   次數與 policy。
3. 正式 J2 loop 在 Bit-True validation與 gate後觀測 joint score，先寫 log並保存完整
   checkpoint，之後才判斷 early stop；exact resume不會重設 stale counter或重複降 LR。
4. 新增 `plateau` event。JSONL、`plateau.csv`、TensorBoard（若可用）與
   `plateau-curves.png`會包含 score、best、stale、recovery、LR multiplier與 stop旗標。
5. Full35與預設停用的 Partial75各自 YAML 都同步相同 policy；Partial75仍未啟用或執行。
6. P1/P2 public stage、兩份 variant training YAML、README、正式設計與 runbook同步更新。

這個模組邊界採用 codebase-design 的 deep-module原則：狀態機與 resume invariant藏在
單一 scheduler seam後面，formal loop只保留「觀測一次、記錄、保存、停止」四個動作。

## 驗證方式與結果

全程設定 `CUDA_VISIBLE_DEVICES=''`，未存取 GPU，也未啟動訓練或 validation run。

1. 定向測試：

   ```text
   13 passed in 0.57s
   ```

   覆蓋第8個 stale epoch降 LR、只降一次、改善後 stale歸零、17 epoch早停、
   `min_delta=1e-4`、momentum禁止中途調整、scheduler state round-trip、P1/P2/J2設定與
   Full35/Partial75一致性。

2. 全部非 integration CPU 回歸（CPU threads限制為2、低優先序）：

   ```text
   71 passed, 24 deselected in 4.19s
   ```

3. stage-policy integration（同樣隱藏 CUDA、CPU 2 threads）：3 passed in 0.84s。

4. Ruff定向靜態檢查：通過，無輸出。

## 困難與解法

- 困難：本次環境的 `apply_patch` update模式因
  `bwrap: loopback: Failed RTM_NEWADDR` 無法讀取既有檔案。
- 解法：新增檔案仍用 `apply_patch`；既有檔案先保留 `/tmp` 備份，必要的小型更新改用
  GNU unified patch。工具產生的 `.orig`／`.rej`已移到 `/tmp`，repository未留下雜檔。
- 其他困難：無。

## 未解事項與風險

- 尚未以正式 GPU J2 metrics觀察8/17門檻是否太敏感；目前值是可重現、可設定的首版。
- plateau監控的是 unconditional joint score；八項 `delta >= -0.08` hard gate仍是獨立
  veto，不會因降 LR或平均分數而被繞過。
- 本次不會產生新的 mAP/AP；正式數值仍須先完成 canonical Pose P1→P2→P3、八項
  standalone baseline，再執行 J1→J2。
- physical batch128與完整訓練時間仍須等 GPU空閒後做 memory gate；本次沒有改變這項
  前置風險。
