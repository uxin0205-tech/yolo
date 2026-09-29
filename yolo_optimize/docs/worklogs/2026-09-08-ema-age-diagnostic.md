# 2026-09-08：EMA age 單變因診斷（已完成）

## 授權與變更原因

使用者同意追加 EMA 對照，並要求自主準備、進入 GPU 訓練後每 600 秒休眠觀測。上一輪原生 E1 AP 下降，BN／參數混合對照已完成；本輪只隔離 EMA age，不同時調整 LR、BN scope 或加入 HOG。

原方案兩組各 1 epoch，經唯讀確認 loss、optimizer、BN、criterion、scheduler 均不讀 EMA 後，改為單一 live trajectory 同步維護兩份 EMA。這是相同軌跡上的精確配對，不是兩次獨立重複實驗；省去重跑同一訓練軌跡，兩份 EMA 仍各觀察完整一個 epoch。此執行調整已向使用者說明。

新增 `src/yolo_optimize/ema_diagnostic.py`、`tests/test_ema_diagnostic.py`、`scripts/run_ema_diagnostic.py`。`scripts/supervise_recovery.py` 新增受限的 `--variant ema-age`，沿用原監測與輸出邊界，不支援任意外部命令。

## 實驗契約

- 同一原始 J3 best_joint inference EMA，strict paired full-resume 配對確認，原檔不變。
- fresh EMA 初始 updates=0；continued EMA 初值只讀 paired snapshot 頂層 `ema_updates=26597`。兩者 decay=0.9999、tau=2000、固定 state exact-copy 相同。
- 成功 optimizer step 才同時更新兩份 EMA；observer 不是 nn.Module，不加入可訓練參數。
- 原生訓練固定 1 epoch；LR 仍取 10-epoch horizon 的第一個 epoch，warmup=1，fresh AdamW，Neck LR=1e-5、兩 heads LR=2.5e-5，其餘沿用上輪。
- Physical Detect batch=32；logical=128（32×4），每 macro 為 Detect 256+Pose 16。COCO80 與 canonical BBAT5 完整資料不變。
- 不改 backbone／attention／MASF freeze、shared BN 統計與 head BN train 規則；criterion 延續 Detect updates=51／Pose=59，不重設。
- 先完成固定 epoch，保存同一 RNG 邊界的兩個完整 snapshots，再驗證 live、fresh EMA、continued EMA 的 Float／BitTrue 八項 AP。驗證不得修改被評分 state。
- 不自動升格 BEST，不把 EMA 平滑改善誤認成 live 學習改善；不納入第二輪或 person-only。

## 準備與驗證

唯讀 preflight：RTX 5090 32607 MiB，已用 443 MiB、無 compute apps；磁碟約 999 GiB 可用。原 BEST inference 106825541 bytes，paired full-resume 425495487 bytes 均存在；runtime COCO80／BBAT5 YAML 與原 canonical registry 存在。

新增 4 項 CPU 測試通過（0.456 秒）：3 個成功 macros 的雙 EMA 與兩個同種子單 EMA 參考，其 live／EMA state 完全相同；一次 overflow 後 optimizer／observer 只更新一次；固定 state bit-exact；拒絕缺失、bool、負值與非整數 parent age。Observer 建構不消耗 RNG、不進入 trainable model。未重跑上一輪已通過的 45 項測試。

兩個 CLI 的 `--help` 載入成功。新入口重用既有 training runner／資料唯讀 guard／snapshot API，不改原始 final。

## 執行與監測

2026-09-08 19:59:26（Asia/Taipei）啟動，child PID 3582970；run：`artifacts/direction1-20260908/ema-age-paired`。Supervisor 每 600 秒查 GPU／磁碟與進度；10 秒 poll 偵測提前結束，60 秒輸出進度摘要。Agent 等待期間只做短時間可中斷等待，不另外啟動 GPU 工作。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/uxin/yolo/yolo_combine/.venv/bin/python -u scripts/supervise_recovery.py --variant ema-age --physical-batch 32 --output artifacts/direction1-20260908/ema-age-paired
```

日誌：`artifacts/direction1-20260908/logs/ema-age-paired.log`；監測：同目錄 `ema-age-paired.monitor.jsonl`；逐步資料：run 下 `progress.jsonl`、`resolved-config.json`、`summary.json`。

## 困難與解法

來源資料／GPU preflight 無阻礙。預設沙盒既有 bwrap 故障仍以受限範圍的必要權限提升處理，沒有擴張原始來源寫入權限。單獨兩次 GPU 訓練會引入非決定性差異且重複計算，因此使用已經 CPU 等價驗證的共享 trajectory 對照。

## 未解事項

本次 `complete`，AP 結果已取得，沒有升格新 BEST。既有 E1 僅作歷史參照，不冒充修補後對照。視覺失敗案例仍未收到；目前工具無法讀取 /status weekly 配額，不能聲稱已確認 73% remaining 門檻。

## 完成結果與決策

2026-09-08 20:12:36（Asia/Taipei）supervisor 確認 child exit=0；child 耗時 784.967475 秒（約 13.08 分鐘）。成功 463 macros，AMP retries 共 7 次（macro 0/6/51 為 4/2/1），final scale=512；兩份 EMA final ages=463/27060，固定 state 與驗證前後 state digest 檢查通過。結束後 nvidia-smi compute 清單為空。

BitTrue joint：parent 0.7111747389752653、live 0.7071994219042768、fresh EMA 0.7072467239000011、continued EMA 0.7112260043801761。Continued 比 fresh 高 0.00397928048017504，但只比 parent 高 0.00005126540491084963，不宣稱有意義／顯著增益。其 Ball Box 仍低 parent 0.0005105164。

EMA age 在同一 live trajectory 上確實改變評分結果，卻不改 live 學習。Continued 的 parent 係數乘積 0.9540919068，約保留 95.41% 起點；fresh 的 log(product)=-1163.0168805，直接 exp 會 underflow，約 8.09443e-506。這支持「平滑維持原基準」的解釋，不把它誤稱為 live 優化已成功。

600 秒 GPU 監測於 20:09:26 記錄，elapsed=600.629520 秒、GPU 50°C、17337/32607 MiB、65.87 W，當時處於 validation 切換，瞬時利用率 0%。程序後續正常完成，不以單點利用率判斷掛起。

結果已由 `scripts/report_ema_diagnostic.py` 生成並在 terminal 顯示，完整八項 AP、公式與來源見 [EMA age 結果報告](<../../proposals/integrated-roadmap/ema-age-diagnostic-results.md>)。

後續原定 native5/HOG10 採共同 parent EMA age，兩者從原 BEST 重新建 fresh optimizer，不從本次候選偷偷續訓；另逐 epoch 記錄 live BitTrue AP，selector 與安全暫停仍使用 EMA 八項 AP。先執行原生對照，HOG 不並行、μ 尚未校準。本次結果不足以取代原 BEST。

補充困難：文件同步曾因未提供確切補丁而被權限審核拒絕；直接補丁工具另遇 bwrap 讀檔故障。後續將實際目標與完整補丁交給審核，獲准後才套用相同內容。沒有繞過審核或擴張來源寫入範圍，GPU 工作未受影響。
