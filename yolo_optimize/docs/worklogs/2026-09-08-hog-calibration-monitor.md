# 2026-09-08：HOG 接續、校準修正與 400 秒監測

## 使用者指示與範圍

使用者要求直接從 HOG 往後執行；個別候選停止後，先分析再自主接續符合前置條件的下一個實驗，不把單臂精度停止線擴大為全流程停止。維持單一主代理、400 秒完整監測、提前退出立即處理。第二輪與 person-only 不納入。

## 問題與推導

`hog-parent-ema-control` 於 23:01:39 啟動，23:01:59 以 exit 1 退出。只做 2 個 training calibration macros，optimizer steps=0，沒有正式 epoch。原 μ=0.1981686883，Detect 梯度比 0.01839281、Pose 0.06106620；Detect 小於原定 0.02 下限。不是 GPU 故障，也不是精度評分失敗。

依 `diagnosing-bugs` 以實測平方梯度重播原 `calibration_coefficient`，CPU 0.86 秒得到相同 Detect 比例並 assertion 失敗。根因是先求整體 5%，再檢查各任務範圍，沒有把兩者一起作限制。若 r_t=sqrt(H_t/N_t)，既定條件為 0.02/r_t ≤ μ ≤ 0.10/r_t。這次兩個區間有交集；選最接近原整體目標的可行 μ 約 0.2155，Detect 約 2%、Pose 約 6.64%、整體約 5.44%。這不是精度恢復的證據。

## 修改與驗證

`training.py` 校準加入任務共同可行區間投影；原 2–10% 不變，共用單一固定 μ，不加每圖 selector；零／非有限梯度或無交集仍停止。記錄 unconstrained μ、實際整體比例與 policy 版本。所有校準更新丟棄，正式訓練從原 PSEL 重建。沒有改原 final、dataset、BN、LR、MASF 或 BinaryQK。

新增 `test_hog_calibration_band.py`，先確認新契約未實作時 5 項失敗，再修正。執行相關 calibration／probe tests：11 passed、17 deselected，0.99 秒；沒有重跑無關全套。原 CPU 重播是實際失敗路徑；新測試驗證記錄的真實雙任務範例、原目標可行時不變、無交集與無效值停止。

## 接續實驗

新 run：`artifacts/direction1-20260908/hog-parent-ema-band-v1`，從原 `best_joint` inference EMA 初始化，延續 parent EMA age；AdamW、base LR scale=1、Neck 1e-5、heads 2.5e-5、aux 3e-4；warmup 1、scheduler horizon 10、最多 10 epochs、patience 4。physical batch 32，logical 128，Pose 16。每 epoch 保存／評分，live 只診斷，不混入 EMA selector；沿用原 native5 的 EMA 精度停止線。只比較共同完成的 epoch，不將不等訓練預算稱為純 HOG 收益。

監測由獨立 session 的本地 supervisor 每 400 秒記錄 GPU、磁碟及進度，間隔內只偵測程序退出。23:35:59 啟動（supervisor 3692668／training 3692671）。GPU 校準已通過：μ=0.2154849493323042，Detect 比例 0.020000000000000004、Pose 0.06640225148588978、整體 0.054369070885695626。這是原始 GPU 校準情境的實際重跑，不只 CPU 模擬。

23:42:39 首次 400 秒完整檢查：E1 macro315／463、GPU 63°C、使用率 59%、顯存 12099 MiB、磁碟可用約 1.06 TB；該 macro loss 有限、AMP 重試 0。μ=0 與 aux 無梯度符合既定 E1-off／E2-ramp／E3–8-hold／E9–10-off 排程，並非啟用失敗。尚無完整 epoch AP 結論。

## 困難、風險與保留

23:49:20 第二次完整檢查（elapsed 800.11 秒）：E2 macro97／463，GPU 62°C／90%／12191 MiB，該步 AMP 重試 0；μ=0.0456102，`hog_auxiliary=true`，Detect／Pose 兩任務均有有效 HOG target。E1 已保存，EMA joint=0.7112260043801761，与原生 E1 相同，符合 E1-off 的設計；不是 HOG 收益。E1 live 若干 AP 仍低於 parent 0.005，維持已登錄 diagnostic-only，不事後混入 EMA 選模或改變與 native5 的停止契約。監督程序持續運作，尚無 HOG 最終精度結論。

工具 sandbox 的 bwrap 問題沿既有完整補丁授權流程處理。校準只用 training prefix，不能保證整個訓練過程的梯度比例固定；不使用 validation 調 μ。修正預防方式是把全域目標與任務界限寫成同一校準契約，而非遇到此案例就放寬下限。其他困難：無。所有舊 run 與原 BEST 保留；無刪除、commit 或 push。
