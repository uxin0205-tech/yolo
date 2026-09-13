# 移除 BinaryQK 補償設計後重新訓練

狀態：PWL 契約已確認；依使用者最新要求先完成報告發布，尚未啟動訓練。2026-09-13 使用者明確更正：不是只在推論時切換 FP-QK，而是移除 BinaryQK 相關設計、恢復原本 Attention 後重訓。

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

固定 scale／bias 的保留路線、替代結構與配對 control 的安排見[最新研究計畫](<../../docs/research/2026-09-13-attention-scale-bias-alternatives.md>)。本目錄不因報告完成就代表訓練已排隊。
