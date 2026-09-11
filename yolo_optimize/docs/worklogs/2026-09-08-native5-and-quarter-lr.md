# 2026-09-08：Native5 結果、CPU 分析與 LR×0.25 候選

## 完成結果

`native-parent-ema-control-adopted` 完成 E1–E5；E1 明示採用診斷完整快照，E2–E5 在新程序訓練。E5 EMA Ball Box AP=0.5023759176，比原 BEST 下降 0.0050611186，觸發預設 safety gate，保存後暫停。E5 EMA／live joint 分別 0.7101160556／0.7053870730；原 BEST 0.7111747390。原始權重不替換，詳細表格、架構與推導見[native5 報告](<../../proposals/integrated-roadmap/native5-results.md>)。

21:42:48 的退出事件確認 GPU 回到 443 MiB／0% 使用率。原 supervisor 因對話中斷退出，後續 observer 不是訓練的 parent，退出代碼不可觀察；只依完整 summary、checkpoint、progress 與 `/proc` 退出作判斷，不假造 exit code。

## 變更與原因

以真實 `severe_regressions` 函式重播 E5 已保存八 AP，0.85 秒穩定回傳 exit 1，明確重現 Ball Box delta −0.0050611186。這是精度安全線紅燈，不是 CPU 測試故障。

新增 CPU-only `scripts/analyze_native_control.py`，讀五個完整 snapshot 的 live／EMA 與原 BEST、既有 2315 個 macro 紀錄，不重新推論或改資料。五 epoch 的固定 state 全部零變更；E5 Pose one-to-one／Detect one-to-one／Neck 相對參數變動分別 1.760%／0.460%／0.582%。24 個既有 gradient samples 的 cosine 中位數 −0.012162，負比例 62.5%；只有稀疏弱衝突證據，不足以判定為主因。Head BN 也漂移，但舊 BN 換回對照未解決 AP 回退。這些結果只排序假設，不宣稱因果已證明。

新增 `base_lr_scale`，只允許 1 或 0.25；用新的 stage value 建 optimizer，不原地修改全域 SCOPE，既有 frozen／trainable scope 不變。新候選只縮小 base Neck／head LR，不改 HOG auxiliary LR（本候選根本未啟用 HOG）。另加可選 `pause_on_live_regression`，只影響保存後是否停止，不讓 live 參與 EMA selector。預設舊行為保留；新 LR／live policy 禁止套用舊 EMA diagnostic E1 prefix。

監督器改為只在 400 秒到期或 child 退出時讀完整指標、GPU、磁碟。間隔內只 `child.poll()` 偵測退出，不每分鐘讀取進度。新的 supervisor 自己也在獨立 session 執行，避免對話中斷時遺失監測。沒有使用子代理。

## 驗證

受影響 4 項 CPU tests 通過（1.41 秒），其中 2 項新增：真實 builder／scheduler 的完整 4630-step LR 比例與 scope 不變，以及 live gate 開啟後先保存再停止。累計唯一 CPU tests=70；沒有重跑無關全套。兩個 CLI help 通過。CPU 狀態分析成功輸出 `artifacts/direction1-20260908/native5-cpu-analysis.json`，不代表 AP 已修復。

啟動前 GPU compute 清單為空，原 BEST SHA-256 仍為 `d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c`。啟動時 resolved config 明確核對 LR、EMA、warmup、horizon、live safety 與 5-epoch 上限。

## LR×0.25 候選：完成 E2 後安全暫停

21:57:10（Asia/Taipei）啟動，supervisor PID 3648797、training PID 3648800；run `artifacts/direction1-20260908/native-quarter-lr-control`。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/uxin/yolo/yolo_combine/.venv/bin/python -u scripts/supervise_recovery.py --variant native --physical-batch 32 --ema-age-mode parent --validate-live --base-lr-scale 0.25 --pause-on-live-regression --output artifacts/direction1-20260908/native-quarter-lr-control
```

Neck LR 2.5e-6，Detect／Pose heads 6.25e-6，fresh AdamW、betas=(0.948,0.999)、weight decay 0.00027；warmup1、horizon10、physical32／logical128、Pose16、task weights1／0.25、parent EMA updates26597 及 BN 原策略不變。只從原 BEST 重建，不接退化 E5。最多 5 epochs，EMA 或 live 任一必要 AP 比 parent 下降超過 0.005 就保存再停。與既有對照只比較共同完成的 epoch。

監督輸出：`artifacts/direction1-20260908/logs/native-quarter-lr-control.supervisor.log`；完整監測 `.monitor.jsonl` 每 400 秒一筆，提前退出會立即新增事件。

## LR 與 BN 最終結果

LR×0.25 於 22:19:20 保存後以 exit code 0 安全暫停，只完成 E1–E2。E2 EMA joint=0.7105572037；live Ball Box=0.4914646391。Live BBAT Box／Ball Box／Ball Pose 相對原 BEST 分別下降 0.009120／0.015972／0.005987，觸發預設停止線，未跑 E3。EMA 達線不代表 live 已修復，也不把較早停止的結果稱為同預算勝出。

新增 `diagnose_native_recovery.py --live-snapshot`，在同一低 LR E2 live 參數上只換回 252 個 parent head BN buffers，以完整 683 張 canonical BBAT5 val 評分。介入前後參數 SHA 相同：`a2c0a198c9a8a1a4c624dd0aecc2fc14257a5b00099ffeeddaa56b638648cec5`。Ball Box=0.4954847264，改善 0.004020，但仍低於原 BEST 0.011952；BBAT Box=0.6221657988，亦未過線。評分 3.15 秒，exit code 2 是精度 gate 未過，不是程式崩潰。沒有重跑 COCO，不計算或拼湊 joint。

補查 parent 的 runtime batch plan 與 overrides，實際 physical batch=32，並非 nominal 64；Detect／Pose mosaic=0，fliplr=0.5／0，與本輪一致。因此沒有 batch 或 augmentation 不一致的證據。LR 與 BN 都有局部影響，但不足以證明唯一根因；正式 HOG 未啟動。

## 交付核對與保留清單

依 `finish-work` 同步根 README、優化索引、整合 README／plan、master plan、optimizer policy 與 machine-readable plan，移除「尚未選 PSEL／實驗未啟動／LR 執行中」的當前狀態誤述；歷史流程保持標示。`active_run=null`、`training_enabled=false`、完整監測間隔 400 秒。

最後 CPU 唯讀檢查通過：兩個 summary 均為 `paused_for_analysis`，合計 7 個 epoch 保存檔存在且各大於 100 MB，7 筆 epoch 的已記錄 EMA／live 指標為有限值；plan JSON 可解析，4 支相關 scripts 通過 AST 語法檢查。這不是重新執行 AP 評估或完整 snapshot 恢復測試。先前累計 70 項唯一 CPU tests 通過，不重跑無關測試。根 README 與工作紀錄索引連結已核對。

以下路徑皆位於 `artifacts/direction1-20260908/`，全部保留，不刪除、不搬移、不提交或上傳：

| 產物 | 約略容量 | 處置與原因 |
| --- | ---: | --- |
| native-parent-ema-control-adopted | 6.0 GB | Keep：5 epoch 權重、恢復狀態與指標證據 |
| native-quarter-lr-control | 3.7 GB | Keep：低 LR 的 2 epoch 與停止原因 |
| ema-age-paired | 1.5 GB | Keep：E1 採用來源與 EMA 成對證據 |
| logs | 7.8 MB | Keep：400 秒監測與程序退出紀錄 |
| diagnostics | 144 KB | Keep：BN-only 與歷史診斷結果 |

原 BEST 與資料集不修改；沒有新增已驗收配方。後續 HOG 可從原 PSEL 獨立比較，不接退化 checkpoint，也不要求原生對照必須先超越原 BEST。

## 困難與解法

BN probe 編輯時曾將參數不變量檢查插入 if／elif 之間，在啟動前閱讀檢查發現並修正；正式執行成功驗證參數不變，沒有產生該錯誤版本的實驗結果。產物受 Git ignore 保護，初次 `rg --files` 未列出；改用 `--no-ignore` 唯讀核對，並非檔案缺失。

原因尚未唯一確定，採單一 LR 變因與 live safety 限制成本，不同時加 HOG／改 BN／改 task loss weight。CPU replay 只能重現 gate，不能取代 GPU AP 因果實驗。先前工具 bwrap 問題依明示完整補丁的授權流程處理，未繞過權限。其他困難：無。

## 未解事項

LR 候選與 BN-only 評分已完成，皆未提供足以替換原 BEST 的結果。根因尚未唯一確定；HOG μ 尚未校準、正式 HOG 未啟動。MASF、BinaryQK、RepConv、MuSGD、第二輪與 person-only 未在本候選啟用。使用者視覺失敗案例仍未提供；73% 配額門檻已取消，採單一主代理與 400 秒完整監測。
