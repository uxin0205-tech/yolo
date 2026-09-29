# 2026-09-09：RepConv layer17 正式接線與連續安靜監測

## 範圍與原因

依使用者要求不在單一 job 後結束全流程，接續既定 RepConv layer17 實驗。HOG 已拒絕，不疊 HOG／MASF 位置／QK 變更；使用原 BEST。同原生對照的 AdamW、Neck 1e-5、Detect／Pose heads 2.5e-5、physical32／logical128、Pose16、warmup1、horizon10、parent EMA age26597，最多 5 epochs，沿用必要 EMA AP parent−0.005 安全線。Live 只診斷，與原 native5 契約一致。

## 修改

`repconv.py` 加入 layer17 安裝與 eval 副本融合；新分支初始化隔離 RNG，保留原 BN eps／momentum。trainer 新增 `rep17`：在原 BEST／criterion 正確載入後 graft，before-graft parent digest 另存；完整 snapshot contract 明記 layer17 reparameterization，live／EMA 都保留雙分支。評分與 inference export 才深拷貝融合成原 Conv state 格式；不改來源、不消耗训练 RNG。5-epoch 政策與 native 相同，沒有更改原 source bundle 或 hardware guard。

新增 `quiet_recovery_supervisor.py` 與 `start_quiet_recovery.py`。正式程序與 supervisor 放在獨立 session，正常不讀 log、不查 GPU、無進度取樣；每次最多等待 600 秒。退出才讀 summary，区分 JOB_DONE／ERROR，不把單臂安全停止當成全流程 ALL_DONE。純等待模式不聲稱能自動辨識仍存活且沒有明確事件的 STALLED。

## 驗證與結果

相關 CPU tests：25 passed，1.63 秒，包含既有 training contract 與 RepConv 轉換；沒有擴大到無關全套。

`verify_rep17_integration.py` 在真實 Full35／原 BEST 的 CPU 圖完成：160×160、Detect／Pose 原圖與零分支候選初始輸出 torch.equal；eval 副本、Float／BitTrue materialization 各自兩任務整圖 parity 通過 atol=1e-4、rtol=1e-4。新分支合成更新非零；正式 full snapshot 保存、重新以 `_build(rep17)` 建圖、公開 `load_training_snapshot` 恢復後，下一步 optimizer／EMA 結果與未中斷路徑逐 tensor 相同。合成權重只作 correctness，不是可部署或 AP 證據。產物 `artifacts/direction1-20260908/rep17-integration-cpu/` 保留。

`rep17-gpu-smoke` 於 00:41:22–00:41:44 執行一個真實 training macro，JOB_DONE、exit0、summary passed。新分支確有參數更新；既有固定 state／BN／AMP 重試與單步 EMA／optimizer 檢查均通過；更新全部丟棄，不作正式訓練前綴。詳細指標見該 run summary。

## 正式接續

預定新 run `artifacts/direction1-20260908/rep17-parent-ema-control`，只在上述檢查通過後啟動，從原 BEST 重建。既有 native5 是比較對照，僅比較共同 epoch；不把改過架構的 base state hash 與原圖誤判為同一 tensor manifest，而用 graft 前 parent digest 核對來源。新候選不預先宣稱精度有效。

下一個事件後分析完整 summary；若未過線就保留原 BEST，依可執行前置條件接續其他方向，不自動放寬 gate 或加 epochs。使用者允許基於證據另開必要加訓，不代表每個退化候選都要延長。

## 困難與未解事項

CPU／GPU 必要 correctness 檢查均通過；正式 RepConv AP 尚待實驗，不宣稱硬體 latency／energy 收益。工具 sandbox 問題依既有明示補丁授權解決。其他困難：無。沒有刪除、commit、push 或改動 canonical dataset／原 final。
