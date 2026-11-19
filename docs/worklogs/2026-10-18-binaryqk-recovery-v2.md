# 2026-10-18：依 optimize 重建初篩／恢復，直接刪除錯誤產物並啟動長訓

## 任務與範圍

使用者要求繼續依既定計畫執行，且錯誤內容直接刪除、不用封存。延續第 1 階段，不另啟動其他階段；單一主代理。實作前先唯讀核對 optimize 的 A0 恢復／梯度修正，並沿模型血緣追查上游 H-SCR、W-DIR 與本機 Ultralytics 實際行為。

## 變更內容與原因

1. 先保存 [事前方法核對](../../yolo_binaryqk/artifacts/stage1/reference-review-v2.json)，含 11 份來源 SHA、原始結果、採用項目、差異與理由。每個候選啟動前核對來源，正式隊列另保存研究程式 SHA，執行中變更程式會拒絕後續候選默默沿用。
2. 移除唯一 trainer 中的錯誤 Q/K-only／`5e-7` 配方，改為三分支各 10 epochs Attention 適應，選一組後接最多 40 epochs 全模型恢復、patience 5。三組各自從同一 native parent 出發；不繼承舊 B1 的額外訓練。
3. 參照初次恢復採 AdamW `5e-5`、batch 16、logical batch 64、betas `(0.937,0.999)`、weight decay `0.0005`、3 epoch warmup、linear lrf 0.01。保留 native batch-sum；固定 4 microbatches、尾批保留。初篩／恢復 criterion horizon 分別 10／40。
4. 保留精確 XNOR 前向與 signed-dot／clipped STE，更新整個 Attention 而非只 Q/K。全模型 BN statistics 可更新；EMA 跟進可訓練權重及 BN buffers，固定係數及凍結 affine 逐位不變。
5. AMP FP16 訓練、FP32 score／Softmax／parameters／evaluation；參照 optimize 的 overflow retry，還原 BN 並重算相同 macro，不跳過資料。GPU 短測未發生 overflow。
6. 與來源差異已列入 protocol：本輪維持校準固定 scale、不學 gamma；全部 LR 從 0 暖機；固定累積；mosaic 等關閉；官方 COCO 選型並另存 internal。沒有宣稱逐步完全重現舊 W-DIR。
7. Checkpoint v2 加入 immediate parent 路徑／SHA 與 phase；resume 含 live／EMA／optimizer／scaler，核對完整契約。恢復重建 optimizer、EMA、criterion；未提交 epoch 的孤立 best 可在合法 resume 後清除重跑。
8. Supervisor 改為三組初篩→只接 winner 恢復→第 1 階段交付。最高初篩 AP 若低於實測 parent 官方 AP 的 90%，停止診斷。完成自動生成八項 AP CSV／JSON、曲線 SVG、結果報告及 handoff，不進入第 2 階段。

完整現行設定見 [stage1-protocol](../../yolo_binaryqk/docs/stage1-protocol.md)。使用的 `yolo`／`yolo-training` 技能由主代理執行，依本機版本核對，不使用雲端服務或上傳資料。

## 清理內容與結果

依使用者明確授權，先確認舊 run 沒有 active process，再直接刪除舊 `formal-20261018`（包括 B1 best／resume）、舊 trainer smoke、啟動／profile／verification 產物及舊圖，共 **44 個檔案，allocated 532,443,136 bytes（約 508 MiB）**。沒有 archive／backup；小型精度、失敗原因、原路徑與 SHA 見 [刪除清單](../../yolo_binaryqk/artifacts/stage1/cleanup-invalid-v1.json)。

新版 GPU 短測完成 strict reload、續訓及 phase 交接後，其 8 份可重建臨時 checkpoint 也直接移除，釋放 allocated **1,242,083,328 bytes（約 1.16 GiB）**；保留小型測試結果、recipe 與 target 圖，見 [短測清理](../../yolo_binaryqk/artifacts/stage1/cleanup-smoke-v2.json)。這些短測不是研究 parent，不能拿其不完整 epoch 結果作正式 AP。

受保護的既有正式模型、optimize 使用的外部 CSV、原始 parent、BBAT5 v1、Pose 與仍被載入的 `yolo_p2/ultralytics` 不在刪除範圍。同步修正舊文件的產物連結，保留必要歷史事實，沒有留下失效的可執行舊配方。

## 驗證方式與結果

- CPU：**44 passed in 3.35s**；Ruff 全部通過。涵蓋二值與梯度契約、Attention／全模型 scope、BN／EMA、不同 phase 排程、recipe／scaler resume、GPU 占用／STOP／續接、只恢復 winner、來源鎖及不需 checkpoint 的 JSON 重畫。
- 真實特徵：沿用原 128 張 train 校準影像，兩處各觀測 128 次，所有數值有限。固定尺度與逐影像幅度參考的 P·V 相對 RMSE 差異小；本輪均值約 0.43～0.69，二值替換仍有明顯誤差，不能以此宣稱 AP 過關。完整 [score 診斷](../../yolo_binaryqk/artifacts/stage1/score-diagnostics-v2.json)。
- GPU：B1／B2／B3 初篩及 B3 full recovery，各 512 張、8 optimizer steps；四組全部 Q/K 更新非零、固定 tensor 通過、best strict reload 及 optimizer／scaler resume 通過，0 AMP retries。
- 初篩 peak allocated 約 3.82／3.82／3.85 GiB，全模型約 9.04 GiB；初篩各約 9.54～9.57 秒訓練觀測，全模型約 13.48 秒。短測含啟動暖機，正式 ETA 須用實際長訓速度更新。
- 已目視 B3 `train-targets.jpg`，影像與框對應合理，繪圖使用獨立 labels 副本，不影響 loss。
- 四項 GPU 短測隊列約 170 秒結束；通過後才啟動正式隊列。完整 [短測摘要](../../yolo_binaryqk/artifacts/stage1/recovery-v2-smoke-summary.json) 與 [驗證摘要](../../yolo_binaryqk/artifacts/stage1/verification-v2.json)。

資料仍是 `/home/uxin/yolo/coco2017`，COCO80 Detect train 118,287／val 5,000，runtime View 完整保留 membership／labels；未抽換、重切或修改來源。BBAT5 v1 完全未動。

## 正式啟動與監測

2026-10-18 **12:46:01（Asia/Taipei）**啟動，supervisor PID `687207`，B1 child PID `687213`。GPU preflight 無其他 compute process。每 600 秒記錄 GPU／進度，候選結束與錯誤以 5 秒輪詢立即處理；完成第 1 階段停止。

- [啟動命令、來源 SHA 與兩段 recipe](../../yolo_binaryqk/artifacts/stage1/formal-launch-v2.json)
- [目前隊列狀態](../../yolo_binaryqk/artifacts/stage1/formal-20261018-v2/supervisor.json)
- [監測紀錄](../../yolo_binaryqk/artifacts/stage1/formal-20261018-v2/monitor.jsonl)

12:50 交接：B1 已進入第 1 epoch 並穩定前進，近期約 **128 張／秒**，Q/K 梯度非零，未發生 AMP retry。以此速度估初篩每 epoch 約 16 分鐘，三組初篩約 10 月 18 日 21:00 前後完成；以短測 scope 成本比例推估，若恢復跑滿 40 epochs，約 **10 月 19 日中午前後**完成，保守預留至下午。若 patience 觸發則較早；其他 GPU 工作、分支／恢復速度與資料 I/O 均可能延後。具體假設與當時數據見 [ETA 快照](../../yolo_binaryqk/artifacts/stage1/formal-20261018-v2/eta.json)，不是保證或持續自動更新的預測。

退出 working 前完成文件收尾：13 份文件的 146 個本機連結有效，11 份方法來源 SHA 均不變，背景隊列的研究程式 SHA 與啟動時一致。正式訓練仍在執行，尚無訓練完成 AP 或 winner。

## 困難與解法

1. 前次誤將已訓 A0 的後期微調配方用在初次二值化。本次實作前追查 optimize 血緣與實際 freeze／BN／loss 行為，再重建，沒有只把 LR 放大。
2. 舊配方移除後原文件會指向已刪檔案。改連到小型刪除／精度證據，並明示舊 checkpoint 已刪除，不建立空殼或封存目錄。
3. 本機 Ultralytics 位於 `yolo_p2/ultralytics`，首次查找多加一層目錄未找到；改查實際載入來源，未刪除該外部依賴。
4. Sandbox 的 loopback 限制仍存在，使用工具核准的工作區指令完成，未修改系統網路。

## 未解事項或風險

正式 AP 與 winner 尚未產生，CPU／GPU 短測不代表精度已恢復或會到 0.51。固定 scale 與訓練適應仍須由正式結果判斷；單 seed、僅恢復初篩 winner 與來源差異會在報告揭露。尚無 RTL／HLS／板端量測。

使用者已授權長訓穩定後退出 working；退出後背景程序繼續記錄，不宣稱會在聊天自動推播。其他 GPU 工作可能延後候選啟動與 ETA。不 commit、不發布、不覆蓋 optimize 正式模型。

## 14:43 進度查詢

依使用者要求讀取 supervisor、B1 status／result、監測紀錄與 GPU 狀態；僅更新紀錄，未改動執行中的研究程式或訓練設定。

B1 已完成 7／10 epochs，正在第 8 epoch。第 7 epoch 在完整 5,000 張驗證集取得目前最佳官方 COCO AP50–95 **0.503361**、internal AP50–95 **0.497097**；初始二值替換的官方 AP 為 0.401422。原始 parent 的官方 AP 為 0.524686、internal AP 為 0.518002，兩種 evaluator 分開比較。B2、B3 與 winner 的全模型恢復尚未開始。

最近完整 epoch 含驗證約 16.5 分鐘。據此估 B1 約今日 15:35 完成，三組初篩約 21:00～22:00 完成；含後續最多 40 epochs 恢復，第 1 階段仍暫估 10 月 19 日中午至下午結束，實際時間受 early stopping、GPU 占用與恢復速度影響。完成第 1 階段後停止，不啟動第 2 階段。

GPU 記憶體使用 6,107／32,607 MiB，查詢當下使用率 73%；背景每 600 秒監測持續寫入。Q/K 梯度非零且有限。前 7 epochs 共發生 2 次 AMP overflow retry，已按既定規則降 scale 並重算相同 macro，沒有跳過資料或中斷訓練。新增困難：無；未解事項：正式 winner 與恢復後精度尚待完整隊列結果。

## 23:34 完成查詢與交付

正式隊列已於 23:10 完成，GPU 已釋放。B1／B2／B3 最佳官方 AP 為 0.503966／0.509889／0.506355，選定 B2；全模型恢復 5 epochs 未改善，保留初篩最佳並停止。完整分析、權重核對與文件同步見 [完成紀錄](2026-10-18-binaryqk-stage1-results.md)。下方與前文的進行中狀態及 ETA 為當時快照；第 2 階段未啟動。

## 17:45 進度查詢

依使用者要求讀取 supervisor、兩組 status／result、最近監測及 GPU 狀態；僅更新工作紀錄，未修改訓練程式與設定。

B1 已完成 10／10 epochs，最佳為第 10 epoch：官方 COCO AP50–95 **0.503966**、internal **0.497746**。B2 正在第 8／10 epoch，已處理 98,560／118,287 張（約 83%）；目前最佳為第 6 epoch：官方 **0.509608**、internal **0.502978**。以上各 epoch 均使用完整 5,000 張驗證集。B2 暫時領先 B1；B3 尚未開始，尚不能確定 winner。原始 parent 官方 **0.524686**、internal **0.518002**，仍須分別比較相同 evaluator。

Supervisor 與 B2 child 持續執行，每 600 秒監測正常，Q/K 梯度有限且非零。GPU 記憶體使用 6,109／32,607 MiB。B1 累計 5 次、B2 已完成 epochs 累計 3 次 AMP retry，均自動重算恢復，沒有失敗或跳過資料。新增困難：無。

近期每個完整 epoch 約 16.7 分鐘，B2 預估今日 18:20～18:30 完成，三組初篩約 21:00～22:00 完成；接續只對 winner 執行全模型恢復，整個第 1 階段仍暫估 10 月 19 日中午至下午完成。未解事項：B3 結果、winner 及全模型恢復精度尚待後續；ETA 受 early stopping 與 GPU 資源影響。第 1 階段完成後停止，不開始第 2 階段。
