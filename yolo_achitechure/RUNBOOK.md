# Stage8 執行手冊

本輪已於 2026-11-08 14:33 完成；五組最終部署完整 AP、folding 與獨立可攜驗證通過。現用 handoff 依使用者選擇為 rep17_20 E1（qSiLU 訓練版），Stage9 待指定。以下保留本輪執行與恢復方法供追溯，不表示有進行中的 queue。先讀 `stage8/protocol.json` 與 `reference-review.json`；腳本使用專案 `.venv`，不更新外部套件。

狀態入口：

```bash
bash yolo_achitechure/run_stage8.sh status
```

原啟動／已完成 epoch 續跑入口（本輪已完成，不需重啟）：

```bash
bash yolo_achitechure/run_stage8.sh run
```

正式root為 `artifacts/stage8/repconv-variants-20261107/`。source.json鎖住來源；啟動後不可改protocol與程式。`supervisor.json`、各job的progress.json與worker.log記錄進度。STOP檔停止本輪；只有使用者解除暫停後才移除。flock防止同一root重複queue。

順序為每候選E0雙版本完整AP、train-only任務梯度校準、2macro短訓、5E、條件式半LR5E，SiLU符合條件才加3E硬體適應，最後fold／重載完整AP與獨立可攜包驗證。選定Rep/ACB及兩heads要有實際梯度更新，固定tensors與EMA必須一致。部署選型不使用SiLU訓練版AP代替。

每600秒主動監測；每60秒檢查其他GPU程序，有衝突則停止自己的worker、等待資源，從已完成epoch續跑。異常立即停止並留下原因。完整Stage8完成後停下，Stage9不自動啟動。

2026-11-08：SiLU 兩次 CUDA SIGSEGV 後，由 `operations/stage8_queue_recovery.py` 從完整 E4 resume 恢復 E5。外層狀態位於本輪 root 的 `recovery-20261108-0925/state.json`；原 `supervisor.json` 與各 job 的 `progress.json` 仍是訓練進度來源。外層只允許額外一次自動重試，且須為 train worker SIGSEGV、無 STOP、來源／config／parent／resume SHA 全部通過。有其他錯誤或沒有完整 epoch 時停止，不重設配方。此機制覆蓋後續 ACB，但不是 CUDA 根因修復。

外層及原 queue 都在執行時不要另啟一份。停止仍使用本輪 root 的 `STOP`；外層不移除 STOP。`PYTHONFAULTHANDLER=1` 與 `PYTHONUNBUFFERED=1` 供即時錯誤紀錄，沒有更動模型或鎖定實驗程式。詳見 [診斷與查核](../docs/worklogs/2026-11-08-stage8-results-comparison.md)。

最終查核：外層一次自動重試已於 09:40:53 用於 SiLU E5，之後 SiLU 5E、3E 硬體適應與選定 E1 的重載驗證完成。ACB 5E 與最終驗證也完成，未觀察到同類 SIGSEGV。原 supervisor 與外層 state 的最終 status 均為 complete；state 內較早的 job／queue_status 是最後一次進度快照，以 terminal status 和 last_supervisor 為準，不改寫原始事件證據。

交付已位於 `result/stage8-retrain/`，`handoff/` 已指向此輪。可攜hand off提供CPU驗證，redraw.py只用小型CSV／JSON重畫圖，不需GPU或checkpoint。多分支EMA／optimizer resume／folded部署檔分開。
