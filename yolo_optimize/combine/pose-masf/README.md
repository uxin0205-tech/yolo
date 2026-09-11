# Pose 資料集 MASF 專項

- [結果與來源](<RESULTS.md>)
- [Head 對照／P3 bridge：架構圖與運算量](<ARCHITECTURE.md>)

**目前已完成並停止：** 使用者改為優先已訓練權重；直接驗證結果見 [RESULTS.md](<RESULTS.md>)。七個 P2／P3 Detect 權重在完整 Pose validation 上量測 box AP，原正式 Pose 量測 MASF 開／關 box／keypoint AP；沒有新增 MASF 正式訓練。訓練草稿與失敗 smoke 全部保留但停用，等待使用者決定，不執行下方原成對訓練計畫。

使用者於 2026-09-10 新增：使用 combine 的訓練配置與指標，檢查 MASF 對 ball／bat 的效果。與無 MASF 融合主線分離，不覆寫舊 Pose 或 COCO 試驗。

## 最小必要實驗

1. 目前無 MASF J0 `j0-no-masf-v1` 作 control；不重新訓練相同 control。原獨立 Pose baseline 與共享初始值均已完整重驗。
2. MASF candidate 從與 control 完全相同的 J0 初始 trunk／Pose head 出發，不從 control E8 出發。只在 Pose head 的 P3 輸入加入 MASF，alpha=0，不改 Detect／共享節點；保留原生 one2one detach。相同訓練 seed、完整資料、8 epoch、Pose head 與 BN 政策；MASF 新參數沿用 Pose head LR 2e-4。原參數 RNG 不受新模組初始化影響。先做 CPU 等價與真實 alpha／context 非零更新驗證。
3. 逐 epoch 比較 ball／bat 各自 box AP50-95、keypoint AP50-95，以及整體 box／keypoint AP；以 BitTrue 選擇、Float 作診斷。相對提升與相對原獨立 Pose 的恢復幅度分開呈現。
4. 兩組每 epoch 都完整驗證 COCO overall／person 與 Pose。呈現同 epoch 配對、各自 best、原獨立 Pose baseline 與既有 COCO P2 MASF 結果；不能把不同插入位置的結論混為一談。
5. 依使用者最新指示，此比較與必要驗證完成後立即停止，等待是否採用 MASF 的決定。不接續 J1/J2、activation 或方向 2。

資料只用 `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml`，train 5964／val 683，不重切或改標註。沒有新增 Detect head，也不啟動 activation 或方向 2。

狀態：已建立規格，尚未開始 MASF Pose 成對訓練。沒有宣稱 MASF 已改善 ball／bat。
