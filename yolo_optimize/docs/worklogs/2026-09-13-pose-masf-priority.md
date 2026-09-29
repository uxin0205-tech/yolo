# 2026-09-13：Pose 端 MASF 比較排入 GPU 第一順位

## 變更內容與原因

使用者要求 Pose 端也要有 MASF 並比較，GPU queue 優先級最高。新增 experiments/pose_masf_priority_v1，保留原 Detect-only MASF 作相同權重基準，將已訓練 Detect MASF 複製到獨立 Pose P3 分支；不改共享 Neck、Detect 或原正式模型。

先固定已安全暫停的 native QK E2 推論副本，避免之後 last.pt 更新使比較來源改變。副本 SHA 與原檔一致，完整 source-pin 保存來源；原檔未刪除或覆寫。這是新實驗的共同起點，不是把 E2 升為正式 best。

新增 CPU preflight、三項 GPU 推論比較及結果彙總入口，包含 baseline／Pose MASF 開／Pose α=0。新增 gpu-priority-plan.json；原 Attention queue 在這項比較未完成前直接 PRIORITY_HOLD。新 runner 使用原 queue lock，既有 native_qk 續訓與 scale_bias 不能插隊。

## 驗證方式與結果

CPU 前置全部通過：Detect 輸出精確不變；Pose α=0 精確重現 baseline；Pose α 開啟改變輸出；Float／BitTrue 重建保留 PWL [-10,0] 20 段與新 Pose MASF。one2one 的 boxes／scores／kpts 合成目標對新 MASF α 的梯度為 -3.2881507873535156，且共享輸入保持 detach；不是用 raw feature 人為製造梯度證據。

基準參數 26,528,689，新增 75,777（約 0.286%）。Python AST 與 compare.py --help 載入檢查通過。執行 run_priority.py 無 --execute 確認只顯示 READY；執行原 run_queue.py 確認顯示 PRIORITY_HOLD，兩者均未使用 GPU。

讀取暫停事件確認原訓練已在 16:52:37（Asia/Taipei）完成 E2 後 PAUSED，progress next_epoch=2、global_macro_step=926；完整續訓 SHA 與推論 metadata 核對成功。尚未產生新 Pose MASF 的 GPU AP 結果，不宣稱提升。

## 困難與解法

新要求的「排 GPU 最前」沒有明確解除上一個暫停指示，因此先完成排程、CPU 準備及安全隔離，並以非阻塞問題詢問要立即啟動或保持暫停。沒有擅自重啟原訓練。

既有 Pose forward 會 detach one2one 輸入，若直接把 MASF 放在外面，該部署分支不會監督它。沿用已驗證 Detect bridge 的形式，讓 MASF 作用於 detached P3，再以梯度比例橋接；one2many 保留正常路徑。當前比例僅用於 CPU 可訓練性檢查，正式 Pose 訓練前仍須真實 loss 校準。其他困難：無。

## 未解事項與風險

目前只完成 CPU 與第一順位排程，GPU 是否立即啟動待使用者答覆，原 Attention pause-request 保留。第一階段只測「既有 Detect MASF 移接 Pose」的推論效果，不能取代等預算的 Pose 專項訓練；後續訓練尚未開始或加入自動執行。新增分支增加參數與計算，未量測延遲、MAC／FLOPs、能耗或目標硬體。資料固定 COCO80 與 canonical BBAT5 v1，沒有新 split 或抽樣。未新增 Git commit／push。

## 使用者授權執行補記

使用者明確要求將原 Pose 與 Pose P3 MASF 比較連同分析完成。2026-09-13 17:08（Asia/Taipei）確認 GPU 無既有計算程序後啟動 run_priority.py --execute；baseline → pose_masf → pose_alpha_zero → 報告。原 Attention 訓練保持暫停，不恢復 scale_bias。採 finish-work 規範核對產物與報告，未授權 Git 發布或刪除，因此不 commit／push、不清除舊產物。

## 完成與分析補記

三項 GPU 比較均成功完成：COCO val 5,000／BBAT val 683 使用相同 E2 起點，原基準精確重現，Pose α=0 全指標還原，新增 Pose MASF 不改 COCO。AP50–95：overall Pose 0.8869435963 → 0.8869351829；ball Pose 0.8490478328 → 0.8495178401；bat Pose 0.9248393599 → 0.9243525257。框 AP 亦未回升，不以單項增益宣稱整體改善。

新增一次全量 BBAT val 特徵診斷（非抽樣、非調參）：實際殘差 L2 相對原 P3 的聚合值為 2.1409%，逐圖中位數 2.1682%。hook 的全部 BBAT 指標與候選一致，warmup 不計入 683 張。這確認 MASF 生效，但不能把特徵變動直接當作 AP 因果證明。Direct Detect→Pose 移接尚未經共同訓練，是報告明確標記的解釋假設。

本機 benchmark：Core Ultra 9 285K／RTX 5090、batch 1、640×640、FP32、BitTrue PWL、一次共享 trunk＋雙 head。CPU 中位數 668.649 → 681.408 ms；GPU 中位數 21.639 → 22.036 ms。單次順序量測，未證明小差異顯著。有限運算集合 MAC 47.923712 → 48.398848 G，新增 0.475136 GMAC 的卷積公式獨立核對；未量測專用板或能耗。

量測初次在 CPU→GPU 時發生裝置不一致：DualHeadPrediction 不是原生 Detect，外層 _apply 沒有搬移兩個 head 的非 buffer anchors／strides。只修 benchmark，明確移動 stride／anchors／strides、清除 shape，warmup 後斷言裝置一致，再重跑該失敗工作成功。精度結果沒有重跑，外部套件與原權重未修改。其他新困難：無。

最終 CPU 稽核通過：所有指標有限、所有 COCO 指標不變、α=0 全指標還原、候選 export state 逐項一致、strict 重載輸出一致、兩處 PWL 契約、全部引用權重 SHA、原 E2 full-resume SHA 與新增 MAC 公式。共保存三份比較 summary、benchmark、全量殘差診斷及 final-audit，報告與 CSV 可依 build_report.py 重建。

依 finish-work 規範把實測、估算、未訓練與未上板項目集中於 RESULTS.md；盤點保留所有權重、cache、驗證產物與來源，不執行刪除。結論：暫不替換原 Pose；Pose 專項 5 epoch 等預算訓練僅為後續建議，尚未執行。原 Attention／scale_bias 繼續暫停，沒有 commit／push。

## 最後交付檢查

9 個 Python 檔 AST、27 個報告／入口連結、8 列精度 CSV 的差值公式、9 列成本 CSV、文件空白與完成狀態全部通過。新 runner 顯示 COMPLETED，原 runner 顯示 PAUSED；最後一次 nvidia-smi 查詢沒有 GPU 計算程序。清理盤點為 4 組 Keep、0 組刪除候選，未刪除檔案。未執行與本次無關的完整倉庫測試、未做多 seed 重訓或目標硬體／能耗量測。最終完整結果入口：experiments/pose_masf_priority_v1/RESULTS.md。
