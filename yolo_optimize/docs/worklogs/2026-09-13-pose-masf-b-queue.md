# 2026-09-13：B 組完整訓練與分析佇列

## 使用者授權與範圍

使用者要求把 B 組訓練及後續分析整個排好。僅執行 B：固定 native QK E2，Pose head＋独立 Pose P3 MASF；A 加訓取消，不恢復原 Attention E3 或 scale_bias，不改正式 default、不自動 Git 發布。

## 實際依賴順序

1. GPU 空閒及共享 lock → smoke：兩次有效 batch128 的真實 BBAT loss／AMP 更新，驗證 α 首步啟動、context 次步梯度與固定 live／EMA。
2. smoke 通過 → 正式 B5：5 epoch、warmup1，AdamW；每回合完整5964張、47次更新、尾端76張。原 E2、移接候選與其他模型不覆寫。
3. 每回合保存完整 snapshot，再做 COCO5000／BBAT683 BitTrue 驗證；驗證失敗先補驗，不跳過失敗回合繼續訓練。固定 state 或 COCO 不符即停；BBAT 任一 AP50–95 下降超過5pp保留權重後停查因。
4. 正式完成 → E5 独立 strict 重載 BitTrue／Float／alpha-off；best Pose 回合不同於E5時另做 BitTrue 重驗與alpha-off。
5. GPU 分析完成 → CPU逐回合續訓、EMA、export、SHA、235總更新與固定 state稽核，產出完整CSV、訓練曲線、中文報告及工程門檻判斷。

## 變更與原因

新增 finalize_b.py、check_queue_b.py 與 queue-plan.json；補強 training_b.py 的先補驗後續訓、驗證 RNG 隔離、匯出比對與訓練／消融安全門檻區分。queue_b.py 新增原始碼 SHA 固定與最終 CPU 工作。這些修改只在本次實驗目錄內，不改舊 runner／外部套件。

## 驗證方式與結果

原 B CPU preflight 已通過，未重跑。此次 `CUDA_VISIBLE_DEVICES='' .../python -B check_queue_b.py` 通過5項排程回歸：模擬驗證失敗只補驗、不改既有export、完成不重驗、例外時RNG還原、訓練大退化攔截但alpha-off可記錄，以及恢復分支在新epoch前補驗（詳情在 queue-preflight-v1.json）。4個執行Python AST與2份JSON格式通過。

真實 GPU smoke 與正式訓練結果以 artifacts/queue-b-v1/state.json／events.jsonl、各summary為準，啟動佇列不等於完成GPU驗證。

## 監測與故障邊界

背景程式自行以600秒為監測間隔；正常時不讀log、不輸出週期進度。外部GPU工作存在時等待，不終止它。每次子程序等待片段最多60秒，十段由程式處理，不需要模型輪詢。

ERROR 停止相依工作、不盲目重試；STALLED 在30分鐘無log更新或macro心跳時發事件，不自動殺程序。事件供主代理後續診斷，不假稱背景程式能自動思考或保證喚醒已結束對話的模型。完成job自動接續已排依賴。

## 分析決策及未解事項

主比較固定B5−E2；best只是補充，不事後換掉主要比較。門檻為COCO差異≤1e-8、overall Pose AP至少+0.002、其餘BBAT AP下降不超過0.001；未全部通過則不建議自動採用。無A5無法分離MASF與額外訓練收益；alpha-off只是已訓模型的分支依賴，不是無MASF重訓對照。無新target硬體或energy實測，不因結果不好加回合／KD／P2。

困難：尚無新的執行錯誤；既有恢復流程的風險已透過例外注入測試核對並修正。尚未驗證GPU的真實loss／AMP／顯存／精度，須等待排程結果。未刪除任何既有權重、結果或cache。

## 實際啟動與 smoke 驗證

佇列 PID 2481161 已啟動；GPU 當時空閒，2026-09-13 19:04:50（Asia/Taipei）開始 smoke，19:05:04 完成並自動接續正式 B5（train PID 2481451）。smoke 使用256張 canonical train 正常批次，共2次累積更新；固定 live／EMA 通過，context 首步梯度為0、第二步最大梯度 2.685578e-06，峰值 allocated 6.433 GiB。未把 smoke 當正式 epoch，B5 重新從固定 parent／RNG／optimizer開始。正常監测交由佇列每600秒處理；尚未完成正式 AP 驗證。

## ALL_DONE

5 epoch、独立驗證、alpha-off 與全部權重 CPU 稽核完成。結果：通過工程參考門檻，未自動升版。無執行錯誤；統計與硬體限制見最終報告。
