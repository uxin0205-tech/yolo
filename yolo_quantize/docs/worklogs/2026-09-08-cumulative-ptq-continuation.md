# 六組 QAT 完成後的有限累積 PTQ 實作

## 變更與原因

14:50（Asia/Taipei）收到 `decision_required,current_index=6,completed_jobs=638,error=null`，六組 QAT 完成。CPU 彙整第六組：第 1 回合未過，第 2–5 回合全過；第 5 回合最差總下降 1.016／1.147 百分點。完整報告保留 PTQ→QAT 各指標，不能相減不同最差指標當恢復量。

依使用者要求整批後接續，並限制次數：新增 runner、十項 CPU 回歸測試與機讀計畫。十區各一個候選，最後同前綴 W6/W5 兩個替代，最多 12 PTQ；後續 QAT 總共最多再兩組、各 5 epochs。使用既有 V36，六組 QAT 提名格式、不拼權重，沒有重訓 baseline。

每步全 16 指標過門檻且 codes+scales 估算較小才接受；失敗保留上一個已量化前綴，負結果留下。NaN、缺項、雜湊漂移、覆蓋錯誤則 fail closed。前綴不可覆寫、設定不可靜默覆寫、續跑需要明確旗標、已完成報告續跑前驗證 hash、舊 child 仍存在時拒絕重啟。沿用既有 mixed-policy evaluator，不修改訓練程式。

已更新 README、CURRENT_PLAN、scripts／artifacts 導航與新 queue 說明。舊 machine-readable rolling 契約保留為血緣，新 plan 明確限制新增訓練上限，不回寫舊結果。

## 驗證

- 六組實際 QAT graph、40 PTQ、592 probe 同 parent／EMA 身分稽核通過。
- 新測試 10 passed；CPU 全回歸 351 passed（50.81 秒），禁止 CUDA、不生成測試快取。
- 新兩檔 ruff check、format 通過；不宣稱原全域 lint 的 6 項／18 檔既有問題已修。
- CPU prepare-only 真實 schema smoke 通過，輸出 11 stages、max 12 candidates、max 2 additional QAT；未執行 GPU。
- 六組結束事件後一次 nvidia-smi 查詢無 compute process，不停止其他人的程序。

## 困難、解法及風險

1. 既有 PTQ evaluator 的內建 selection 仍是 mAP50-only；新 runner 不用它晉級，改讀 all_search_metrics 的 8+8 指標。
2. probe 的 aggregate NRMSE 混入後處理候選及特徵範圍；本次只從既有 raw boxes/scores/kpts 提名，保留原 probe，不假裝重新測量。這仍是小樣本代理，不保證 mAP。
3. 不能把不同 QAT 結果拼接，也不重啟新 baseline；以同 V36 及累積路由重新評估，後续短 QAT 另做。
4. 幾個初始定位檔名不存在，以 `rg --files` 與實際檔案定位修正；沒有因此重寫舊程式。

其他困難無。成本只含權重碼與 FP32 尺度，不含 bias、protected、alignment 等；不是硬體量測。後續 formal、export 重驗與至多兩組 QAT 尚未完成。GPU 啟動與監測狀態另補記。

## 實際啟動

新報告／入口 61 個本機連結全部可解析。已執行 `scripts/run_cumulative_ptq_queue.py --execute-reviewed-plan`，狀態檔確認 `running_cumulative_ptq,current_index=0,current_candidate=backbone_early,current_arm=ptq,completed_jobs=0,error=null`。舊六組 queue 未修改或重啟；新 supervisor 自動逐區接續，監測改用 `artifacts/queues/full-model-cumulative-0908/execution-status.json`。未新增 QAT 或 formal。

## 累積完成事件 1

Monitor 回傳 index=2、backbone_attention_safe、completed_jobs=2、error=null。只讀 cumulative-selection：parent codes+scales 20,290,368 bytes；early 替換通過 16 gates，最差總下降 1.3598／1.4318 百分點，估算 19,995,456 bytes。Deep 累積替換最差下降 1.5839／1.5821 百分點，mAP50 超標，未接受；保留 early 前綴，queue 已自動接 attention。Deep 若接受可再降至 18,815,808 bytes，負結果保留供本批後少量 QAT 恢復判斷，不把 PTQ 未通過當永久淘汰。

驗證為已完成報告及 runner 的 16 指標 gate 摘要，不讀 active log、不改訓練或 queue。困難無；恢復是否有效尚未測量，繼續 600 秒 monitor。

## 累積完成事件 2

Monitor 回傳 index=4、neck_attention_safe、completed_jobs=4、error=null。Backbone attention 追加 SD4 的最差總下降為 2.7039／4.1101 百分點，未通過雙門檻，維持 early 前綴。接續 neck 的 SD4 累積替換通過全 16 指標，最差總下降 1.4055／1.5230 百分點；現已接受 early＋neck，估算 codes+scales 為 19,405,632 bytes，較 parent 減少 884,736 bytes。此數值不是實測檔案大小或硬體加速。

驗證方式為讀取已完成 cumulative-selection 四階段結果，未讀 active training log、未操作 GPU、未增加訓練。依使用者詢問再次釐清 592 probes、40 分區 PTQ 與現行累積 PTQ 均不訓練；既有六組短 QAT 已完成，後續最多新增兩組各 5 epochs，不做每層全格式 QAT 窮舉。困難無；單層代理與累積精度不一致是實驗負結果，不代表程序錯誤。回到新狀態檔的 600 秒 shell monitor。

## 累積完成事件 3

Monitor 回傳 index=7、pose_one2one_tower、completed_jobs=7、error=null。Neck attention 追加 SD4 最差總下降 1.8659／2.2221 百分點，mAP50 超標而未接受；MASF 三路加入 filterwise TWN 後通過全 16 指標，最差總下降 1.4027／1.5529 百分點，估算 codes+scales 為 19,354,304 bytes；detect tower 追加 SD4 最差總下降 1.8921／2.0363 百分點，未接受。

目前累積新路由為 early 與 neck 的 SD4，以及 MASF 三路的 filterwise TWN；其他部署權重保持 V36 已量化方案，沒有退回浮點。驗證只讀已完成階段的門檻／成本摘要；困難無，未做 source inspection、active log 或 GPU 查詢。後續 Pose、predictor 與 uniform 替代仍待測，回到 600 秒 shell monitor。

## 終點與使用者新決定

後续完成全部 11 階段／12 候選，error=null；supervisor 與 monitor 均自然結束。期間使用者依老師建議要求量化延後，因此不執行原先考慮的額外兩組 QAT。完整結果、CPU 稽核與交接決策集中於[階段收尾工作紀錄](2026-09-08-quantization-phase-handoff.md)及[收尾報告](../reports/2026-09-08-quantization-phase-handoff.md)。本篇較早的「接續 QAT／monitor」為當時紀錄，不再是目前授權；沒有刪除原始結果或權重。
