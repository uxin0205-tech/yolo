# 2026-08-26 Full35 seed0 GPU 串接佇列

## 目的

依使用者指示，在GPU空閒後自動開始，而且不只執行30%短測。建立一個持久、fail-fast的
seed0 pipeline；Codex回合結束後仍由user-level systemd持續監看。

## 空閒判定與資源保護

- GPU：index 0，RTX 5090，總顯存32,607 MiB。
- 每60秒只呼叫一次`nvidia-smi`，watcher不載入Torch或CUDA。
- 必須同時符合：
  - compute process數為0；
  - free memory至少30,000 MiB；
  - GPU utilization不超過10%；
  - 連續3次poll都符合。
- pipeline每個後續GPU stage前會再要求連續2次符合，避免其他工作在stage切換時已佔用GPU。
- 不會終止、暫停或調整其他使用者程序。
- 使用`flock`防止同一queue重複啟動；狀態以atomic JSON更新。

排程建立前，GPU仍由MambaPose PID 3170826使用約21,280 MiB，free約10,470 MiB，因此
只會進入waiting，不會立即訓練。

## 串接順序

1. Full35 canonical BBAT5 30% P1：physical batch128、2 epochs、seed0、full val683。
2. 正式100% P1：17 epochs、patience10、batch128。
3. 正式100% P2：最多22 epochs、patience12，從P1 best開始。
4. 正式100% P3：最多100 epochs、patience20，從P2 best開始。
5. 獨立Full35 Detect與獨立Pose P3的Float＋Bit-True八項baseline。
6. 正式preflight；只有ready=true才會繼續。
7. JOINT J1→J2 AdamW seed0；J1固定5 epochs，J2最多40、patience17與第8次停滯後一次
   LR×0.5 recovery。
8. 對`best_joint.pt`再跑一次Float＋Bit-True final validation。

未自動加入J3、MuSGD challenger、seed1或Partial75；這些必須在seed0結果完成後再依hard
gate、時間與使用者決策執行。

## Fail-fast規則

- 每個stage必須return code 0。
- `best.pt`／`last.pt`必須存在且不是異常小檔。
- `results.csv`、baseline JSON與JOINT validation CSV/JSON的所有數值必須finite。
- 下一個Pose stage只能使用上一stage的正式best checkpoint。
- 任何partial output directory會阻止自動resume，避免Ultralytics自動改名後誤接checkpoint。
- physical batch128若OOM，整條pipeline立即停止；不會自動降batch、accumulate、imgsz或改
  optimizer。
- baseline或preflight不合法會停止，不會繞過正式blocker。

## 實作與設定

- `src/yolo_combine/gpu_queue.py`：GPU probe、穩定空閒判定、lock、durable state、單次launch。
- `scripts/queue_gpu_job.py`：通用CLI入口。
- `scripts/run_full35_seed0_pipeline.py`：上述8個Full35 steps、stage間GPU重新確認、產物驗證與
  安全resume。
- `variants/full35/configs/joint.yaml`已預先填入可預測的正式P3與baseline未來路徑；目前檔案
  尚不存在，所以preflight仍會阻擋，直到pipeline完成它們。

預定systemd unit：

```text
yolo-full35-seed0-pipeline.service
```

已於2026-08-26 00:17:59 CST建立，invocation ID為
`9eebcecf30a74b51b6643948e735e6a9`；初始Main PID 3357125。實際systemd properties：
`Nice=10`、`IOSchedulingClass=idle`、`KillMode=control-group`。

durable狀態與log：

```text
variants/full35/artifacts/queue/full35-seed0-pipeline/
  queue-status.json
  pipeline-status.json
  pipeline.log
  queue.lock
```

## 驗證

- `tests/test_gpu_queue.py`驗證：free memory、utilization、compute-process三條件；stable polls；
  單次launch；invalid policy拒絕。
- 相關測試與pipeline `--plan`在CUDA隱藏下完成，16 passed。
- `--plan`列出完整8步、命令、checkpoint依賴與預期產物，沒有啟動GPU。
- Full35 source／dataset `audit --verify-hashes`通過：missing 0、hash mismatch 0、canonical
  registry=`bbat5-v1`。
- 完整repository測試以CUDA隱藏、CPU22–23、最低CPU/I/O優先序執行：105 passed、3個
  CUDA-only tests skipped，耗時12.53秒。
- 建立service後的第一個snapshot：free=10,471 MiB、utilization=82%、compute process為
  MambaPose PID3170826／21,280 MiB，`idle_streak=0/3`、status=`waiting`。沒有建立P1 run
  directory，也沒有本專案CUDA process。
- `joint.py preflight`目前仍正確列出兩個blocker：預定P3 checkpoint不存在、八項baseline
  尚未提供。

## 困難與解法

1. 困難：使用者要求不只短測，但JOINT需要P3與八項baseline才能ready。
   解法：把所有正式前置串成依賴鏈，baseline完成後再執行原本的preflight；不直接繞過。
2. 困難：不同stage間GPU可能被其他工作取得。
   解法：每個GPU stage前重新做兩次穩定空閒判定。
3. 困難：長pipeline可能因system/session結束失去父程序。
   解法：使用user-level systemd transient service，而不是依賴Codex tool session或terminal。

## 未解事項與風險

- batch128尚無空卡實證，第一步仍可能OOM；這是刻意的memory/stability gate。
- pipeline可能運行很久；systemd只負責持久執行，不代表精度一定達標。
- user systemd目前active且有6個login sessions，但`Linger=no`；Codex回合或單一terminal結束
  不影響service，若主機重開機或所有uxin sessions全數登出則transient unit不會自行跨開機
  恢復，需重新建立佇列。
- 30%結果只作健全性檢查；即使有AP/mAP也不會寫入正式baseline。
- JOINT seed0完成後仍要依八項delta≤0.08 hard gate檢視，才能決定seed1、MuSGD、J3或
  Partial75。

除上述風險外，無其他未解困難。
