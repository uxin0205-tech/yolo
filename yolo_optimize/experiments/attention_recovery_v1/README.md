# 移除 BinaryQK 補償設計後重新訓練

最新狀態（2026-09-13 16:52）：**E2 已完成驗證、保存完整 checkpoint 並安全暫停**，結果見 `artifacts/queue-v1/pause-outcome-v1.json`。第一順位 [Pose MASF 比較與分析](<../pose_masf_priority_v1/RESULTS.md>)現已完成；原 Attention 工作仍保持暫停。

先前指示（2026-09-13 16:33）：**已安排目前 E2 完成驗證與存檔後暫停**，後續 queue 已停止，scale_bias 不啟動。尚未宣稱已暫停；以 `artifacts/queue-v1/pause-outcome-v1.json` 為準。詳見[暫停工作紀錄](<../../docs/worklogs/2026-09-13-attention-epoch-pause.md>)。

先前狀態更新（2026-09-13）：CPU 前置驗證、完整 E0 與 GPU smoke 已通過；正式 native_qk 正在訓練。依使用者最新要求，取消額外 binary_control 正式訓練與其後驗證，只保留 native_qk、scale_bias 兩組；已完成前置測試不刪除。最新狀態以 artifacts/queue-v1/events.jsonl 為準。2026-09-13 使用者明確更正：不是只在推論時切換 FP-QK，而是移除 BinaryQK 相關設計、恢復原本 Attention 後重訓。

## 固定來源與保留部分

從目前 qSiLU E2 的已訓練 Detect＋Pose 權重起步，來源及 SHA 見[目前模型白話報告](<../../reports/current-model/README.md>)。舊權重不覆寫。固定 COCO80 與 BBAT5 v1 正式 train／val，不重切資料。保留目前 MASF 位置、qSiLU 與兩個任務 head，避免一次改變太多設計。

## 預定移除與恢復

兩處 Attention：`graph.model.10.m.0.attn`、`graph.model.22.m.0.1.attn`。

- 移除 BinaryQK 的 sign／XNOR 相似度及 Hadamard 第二分支。
- 移除二值分支固定尺度、混合係數與額外的相對位置偏置。
- 恢復原生浮點 `QᵀK/√d`，保留原生 V、pe 位置卷積與 proj。
- 將已訓練 Q/K/V Conv、BN 依每個 head 的 Q/K/V 次序映回原生 fused QKV，不直接串接錯誤通道，不丟棄 Pose 的已訓練參數。
- Softmax 已確認保留 PWL [-10,0]／20 段。此路線稱「原生 QK＋PWL」，不稱完全原生 Attention。

## 最小實驗順序與驗收

1. 重建與 CPU 驗證：檢查移除項目為零、QKV 投影與映回前對照一致、前向有限值、Q/K 有梯度。除明確移除部分外，保存參數映射與來源 SHA。
2. E0：新架構尚未訓練的完整 COCO val5000／BBAT val683 評估。它是恢復訓練起點，不是訓練結果。
3. GPU 一個完整更新的 smoke：檢查 loss、梯度、AMP、顯存、共享與 head 的可訓練範圍。輸出獨立 smoke 目錄，不能混入正式 epoch。
4. 恢復訓練：重建 optimizer，warmup 1 epoch；先以 AdamW 穩定架構切換。暫定最多 20 epoch、patience 5，至少完成 warmup 與 5 個正式 epoch 才以平台判斷；詳細分組 LR 依 E0 與梯度／顯存測試後鎖定。這是初步預算，不是已生效配置。
5. 每 epoch 同口徑完整驗證，分列 COCO overall／person、BBAT overall／ball／bat 框與關鍵點；保留目前選用版本與 E0 兩個比較基準。不只看 joint 平均，也不因單項改善就稱全面恢復。
6. 選定權重獨立重新載入驗證，再做 MASF α 開／關。同 checkpoint 開關只能測依賴性；若要聲稱 MASF 的訓練收益，另做同預算無 MASF 配對，不能偷換比較定義。
7. 如精度有可用收益，再量測 Params、Model size、MAC／FLOPs、CPU／GPU latency、峰值記憶體及能耗；目標板未量測不能推定。若平台仍無恢復，再討論是否改用原浮點 backbone 初始化，不預先全面重訓所有階段。

不直接沿用先前受硬體限制的 Attention LR 5e-8，因為那是保護二值模型的小幅調整，未必適合恢復浮點 Q/K。訓練時不得又由舊 `convert_yolo26_model` 或 materialization 重建回 BinaryQK；這是實作必須明確驗證的邊界。

## 產物與監測

後續配置、架構映射、E0、訓練、獨立驗證各自保存於本資料夾，僅新增版本。不刪除既有模型。不因本計畫自動 commit 或上傳。GPU 工作依使用者要求使用靜默阻塞 queue 與 600 秒監測契約，完成／錯誤時才處理。

舊[同權重切換測試](<../inference/component_switch_v1/README.md>)只作診斷，不取代本研究。

固定 scale／bias 的保留路線與替代結構見[先前研究計畫](<../../docs/research/2026-09-13-attention-scale-bias-alternatives.md>)。其中配對 control 已被最新指示取消；實際執行以本頁與 queue-plan.json 為準。

## 本次實際執行契約

正式兩組：`native_qk`、`scale_bias`。同一 qSiLU E2 起點、seed 1、20 epoch 上限、patience 5（最快 E6 停止）、warmup 1。順序為 native_qk → 獨立驗證／MASF 開關 → scale_bias → 獨立驗證／MASF 開關 → 彙總。兩組皆已通過兩個真實 GPU macro。`binary_control` 只保留已完成的診斷資料，不再安排正式訓練。

兩組皆開放 Q/K 更新；scale_bias 保留 BinaryQK，使用已驗證的 exact-forward surrogate。scale_bias 以 task loss 校準 16 個真正生效的係數，前向量化為固定 m/1024，範圍 [1/1024,1]；不是改無效 gamma，也不是每張圖選尺度。這次先採直接任務監督，不額外安排 train 全量 FP-score 擬合，避免把已適應二值的 FP score 當成理想教師。

native_qk 實際還原 fused QKV，移除 sign/XNOR、Hadamard、固定二值分支尺度與額外 relative bias，保留 pe／proj、PWL [-10,0]20 段。投影用 AMP，QK score／PWL 用 FP32；1/√32 是固定常數，沒有逐圖 sqrt。這是精度恢復參考，不冒充 INT8 上板驗收。

scale_bias 的 relative bias 保持可訓練；native_qk 移除額外 relative bias。兩組均固定 MASF 參數與其 BN，保留 qSiLU。共享 BN running 固定、affine 可訓練。

最終以既有正式 qSiLU E2 作比較基準，不需要新的 binary_control 結果。沒有同預算對照，不能把 scale／bias 修改收益與額外訓練收益完全分離；只回答兩組新模型的實際精度是否優於現有模型。

AdamW，betas=(0.948,0.999)，WD0.00027，clip10，cosine final0.5；backbone3.8e-7、neck1.9e-6、attention1e-5、Detect／Pose head5e-6；scale_bias 的尺度群 LR2e-4、不做 WD，MASF LR0。Detect physical16、logical128，每 macro256 Detect＋16 Pose，Pose physical16。資料固定完整 COCO train118287／val5000、BBAT5 v1 train5964／val683，沒有新 split 或抽樣。

每 epoch 保持原 combine 的 Float→BitTrue 全量驗證，選擇以 BitTrue 為準。best_joint gate 仍相對原 qSiLU E2，八項最大下降0.015；不代表所有指標有提升。災難性停止是相對該臂 E0 任一項下降超過0.05，先保存 checkpoint 再停，不把新架構初始差距誤判成訓練崩潰。

原有 loader 不接受 YAML 直接寫 j3；本實驗循既有 J3 adapter，以合法入口讀取後顯式構造唯一 j3 階段。**真正生效設定是各 run 的 resolved-config.json**，不要把 YAML 載入標籤誤讀成重跑舊 J1。

入口：`run_queue.py`（可跳過已成功工作、正式 run 可從 last.pt 續跑）；`preflight.py`、`validate_recovery.py e0`、`smoke.py <arm>`、`training.py <arm>`、`validate_recovery.py final --arm <arm>`。queue 每600秒為等待監測邊界，60秒內部阻塞片段不讀 log、不查 GPU、不輸出正常進度；錯誤／完成才輸出事件。

原正式 checkpoint 不覆寫。每組獨立保存 inference 與完整續訓 checkpoint；最後會比較 MASF 開／關。未通過接受 gate 不自動升版。

## 取消對照與不中斷接管

2026-09-13 15:47（Asia/Taipei），只更換 queue 管理程序，原 native_qk 訓練 PID 與起始時間不變，沒有停止或重跑 GPU 訓練。新版以 pidfd 等待既有程序退出，隨後自動接續兩組計畫。`queue-plan.json` 是後續待辦範圍；每個新 job 前重新讀取，不將取消工作偽裝為已訓練結果。
