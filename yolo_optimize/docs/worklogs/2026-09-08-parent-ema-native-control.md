# 2026-09-08：Parent EMA 原生對照（E1–E5 已完成，E5 安全暫停）

## 完成結果

E1–E5 全部完成訓練、Float／BitTrue 驗證與 checkpoint 保存。E5 EMA joint=0.7101160556，Ball Box AP=0.5023759176，較 parent 下降 0.0050611186，依既定門檻保存後暫停；沒有升格新 BEST。21:42:48 監測器確認程序退出與 GPU 釋放。原 supervisor 隨對話中斷失效，不能宣稱取得退出代碼；正式 summary 為 `paused_for_analysis`。完整數字、CPU 追加分析與下一個 LR 單變因候選見[新工作紀錄](<2026-09-08-native5-and-quarter-lr.md>)與[native5 報告](<../../proposals/integrated-roadmap/native5-results.md>)。下文保留啟動與修補歷史，不是目前活躍狀態。

## 最新狀態：明示前綴採用已啟動

2026-09-08 20:59:05（Asia/Taipei）啟動新 run `artifacts/direction1-20260908/native-parent-ema-control-adopted`，child PID 3617439。實際 GPU 前置已通過 SHA、設定、scope、AMP、state、optimizer manifest、scheduler／criterion 與八項 E1 AP 核對，已載入 E1、从 E2 開始。以下原 20:22 run 失敗紀錄保留為歷史，不是目前活躍程序。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/uxin/yolo/yolo_combine/.venv/bin/python -u scripts/supervise_recovery.py --variant native --physical-batch 32 --ema-age-mode parent --validate-live --adopt-ema-prefix artifacts/direction1-20260908/ema-age-paired --output artifacts/direction1-20260908/native-parent-ema-control-adopted
```

接續點為 global macro 463、EMA updates 27060、scheduler current_step 463、horizon 10；總長度仍 E1–E5，只有 E2–E5 在新程序訓練。恢復 RNG、AdamW moments、AMP scaler 與已 advance 一次的 criterion，不重設 warmup、不再 advance E1。Selector 由已驗證的 E1 continued BitTrue 八 AP 重新 observe，不把診斷的 `automatic_promotion=False` 冒當 selector state。新 run 另存具有明示 provenance 的 E1 與 run-local best，原始快照與失敗 run 不改。

### 本次修補與驗證

- `validation_boundary` 只恢復無 parameter／buffer／child 的 exact `nn.SiLU` training flag，直接賦值並記錄 alias，不遞迴 `.train()`；其他 tensor／requires_grad／mode 改變仍拒絕。真實 seam RED 2.54 秒 → GREEN 2.39 秒；原始 CPU repro 1.12 秒通過，live／EMA 全部零漂移。新增 9 項 guard 測試通過。
- `control_prefix.py` 僅支援明示 native／parent／physical32 前綴採用。5 項 CPU 測試通過（0.822 秒），含真實 full snapshot restore 後下一個 AdamW step 與不中斷對照 bitwise 相同，以及 source／horizon／step／selector／optimizer manifest 錯配拒絕。Snapshot metadata 對 JSON 摘要採 JSON 語義比對（整數 mapping key／tuple 正規化），不放寬 optimizer payload 或模型狀態檢查。
- 正式迴圈在每 epoch 完成後、驗證前先存 `epoch-N-validation-pending.pt`；不產生新 BEST，驗證失敗仍保有完整 optimizer 邊界。前綴 E2–E5 範圍、live 不影響 selector／EMA pause、先存後驗證與真實 pending snapshot 保存恢復共 3 項相關測試通過（1.28 秒；其中 2 項新增）。兩個 CLI help 與計畫 JSON 格式檢查通過。
- 本專項累計 68 項唯一 CPU tests：training contract 22、HOG 12、資料保護 8、training safety 8、paired EMA 4、validation boundary 9、control prefix 5。未重跑無關全套。

本次額外困難：CPU 不能直接驗證 GPU AMP 設定及還原 CUDA RNG，故沒有假稱純 CPU 完整還原 GPU 狀態；實際 GPU 嚴格前置已通過。公開 resume shim 已保留 optimizer 整數 state keys，沒有 optimizer-key bug，不做無必要修補。其他困難無新增。仍每 600 秒監測 GPU／磁碟，提早退出立即處理；尚未取得 E2–E5 精度結論。

## 變更原因與範圍

EMA age 配對診斷已完成；本輪接續原定 native5 對照，暫不並行啟動 HOG。這次 parent-age run 的 E1 已完成全部 463 macros，EMA Float／BitTrue 與 live BitTrue validation 都完成；在 epoch checkpoint 儲存前，`validation_boundary` 因 mode drift 報錯並以 exit code 1 結束。這是驗證邊界保護退出，不是 AP pause，也沒有可用的 epoch checkpoint。

這不是從診斷候選偷偷續跑：重新從原 J3 best_joint inference EMA 與原 paired full-resume criterion 建立 AdamW，EMA 起始 updates=26597。正式可訓練 scope、BN、LR、資料、種子、macro 與 warmup 都維持既定設定；GPU 工作已停止。

## 實作與驗證

`training.py` 增加 `ema_age_mode='fresh'|'parent'`（預設仍 fresh，保留歷史／benchmark 行為），parent mode 嚴格讀取已配對 snapshot 頂層 ema_updates。增加 `validate_live=False` 選項；本輪設 true，除 EMA Float／BitTrue 八項 AP，另在獨立目錄驗證 live BitTrue。

Live 指標只用於健康診斷，不混入 EMA checkpoint selector，也不因已知 E1 live 回退就反覆中斷。原本 EMA 每項 AP 對 parent 的 0.005 暫停門檻、finite／hardware guards 完整保留。驗證期間保存／恢復 RNG，並檢查來源 state、training mode、trainability 不變，避免影響 epoch-end snapshot。

保留既有設定與 52 項唯一 CPU tests 的歷史紀錄；本紀錄不宣稱本次新增測試已通過，也不重跑無關全套。原有 EMA age 旗標仍由訓練入口及 supervisor 轉發。

## 固定設定

- Native 上限 5 epochs，與未來 HOG 的前 5 epochs 共用 10-epoch cosine horizon。
- Warmup=1 epoch，factor 0.1→1，最終 horizon factor=0.5。
- Fresh AdamW：Neck LR=1e-5、Detect/Pose heads LR=2.5e-5；betas=(0.948,0.999)、weight decay=0.00027、clip=10、AMP。
- Physical Detect batch=32，logical=128（32×4）；每 optimizer macro=Detect 256+Pose 16，reference=64，task weights=1/0.25。
- Backbone、attention、既有 MASF 凍結；shared BN running statistics 凍結，head BN train；不改 BinaryQK 固定值。
- COCO80 與 canonical BBAT5 全量資料、完整 assignment／labels 不變；只使用既有 cache-isolated runtime View。
- E2E criterion 延續 Detect updates=51/horizon120、Pose updates=59/horizon128，不因 optimizer fresh 而重設。
- EMA：decay=0.9999、tau=2000、起始 updates=26597；固定 state exact-copy，AMP retry 回滾 BN/RNG，HOG 未啟用。

## 執行與監測

2026-09-08 20:22:40（Asia/Taipei）啟動，child PID 3595331；20:33:50 以 exit code 1 結束；run：`artifacts/direction1-20260908/native-parent-ema-control`。E1 的 463／463 macros、EMA Float／BitTrue 與 live BitTrue validation 均已完成，之後在 checkpoint save 前由 `validation_boundary` 報告 mode drift；沒有 epoch checkpoint，不能把它寫成 AP pause。GPU 工作已停止。啟動前 nvidia-smi compute 清單為空，方向 1 計畫 JSON 格式檢查通過。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/uxin/yolo/yolo_combine/.venv/bin/python -u scripts/supervise_recovery.py --variant native --physical-batch 32 --ema-age-mode parent --validate-live --output artifacts/direction1-20260908/native-parent-ema-control
```

GPU／磁碟與進度每 600 秒監測，提前失敗／完成由 supervisor 10 秒 poll 偵測；等待時不另開 GPU 工作。日誌在 `artifacts/direction1-20260908/logs/native-parent-ema-control.log`，監測檔同目錄 `.monitor.jsonl`，逐 macro 及逐 epoch 結果保存在 run 內。

supervisor 的 600 秒樣本時間為 2026-09-08 20:32:40（Asia/Taipei），GPU 為 `52°C`、使用率 `64%`、顯存 `9357 MiB`、功耗 `280.72 W`；隨後 child 退出且 GPU 工作停止。

## 困難與解法

GPU／資料前置檢查無阻礙。既有工具 bwrap 故障採明示完整補丁、經審核授權的方式處理，未繞過審核。EMA 可能掩蓋 live 回退，因此同時記錄 live AP；它不自動代表新 baseline 獲得有意義的提升。

CPU 以約 `1.12 s` 重現同一 `validation_boundary` 失敗：Pose `target.eval()` 經由 `Conv.default_act` singleton 改變 live 唯一的無狀態 `SiLU` training flag；該 module 有 24 個 alias path。tensor、BN 與 gradient 均未變，只有 live 的 mode flag 由 `true` 變 `false`，因此 guard 正確拒絕保存。最小修正目前進行中，只恢復 exact 的無狀態 `SiLU` mode，其他 guard 保留。

與既有 run 對照時，`13905` 個 step／metric／value 與 `463` 個 macro 事件除 `time`／`seconds` 外完全相同；這只說明事件與數值軌跡一致，不宣稱未保存的失敗 run tensor 或 diagnostic snapshot 已逐 tensor 比對。

## 未解事項

目前 run 已退出，沒有完整 5-epoch 結論，也沒有 epoch checkpoint。已準備明確採用已完成的 `artifacts/direction1-20260908/ema-age-paired/checkpoints/continued-epoch-0001.pt`（SHA-256：`44d78885d839fa7e2a8626b9ee0785ef7b530a5d09c05678fc973102562a395d`）續跑 E2–E5，但尚未啟動；需先完成上述最小 mode 修正。不自動把 run-local best 升格成原 BEST；HOG μ 尚未校準。MASF、BinaryQK、RepConv、MuSGD 與第二輪不在本次原生對照中。

使用者指定的視覺失敗案例尚未收到；目前工具無法讀取 weekly 配額，不能宣稱已確認 73% remaining 門檻。完成或觸發安全暫停後立即分析，不擅自增加預算或變更資料。
