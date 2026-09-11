# MASF 名稱、架構與運算量說明

## 變更與原因

回應使用者對 RESULTS 的原 Pose 消融、Head 對照 E8、P3 bridge E8 與原架構／運算量疑問。新增 `combine/pose-masf/ARCHITECTURE.md`，以 ASCII 接線圖區分原 B100 shared P3、無 MASF control、P3 Detect-only bridge，並說明訓練與推論差異。

## 驗證方法與結果

唯讀核對實際 `masf_p3.py`、`masf_task_bridge.py`、`continue_masf_head.py`、`train_masf_task_bridge.py`、MASF 原始模組與三組 E8 summary。確認 Head 是更新範圍，不是新增 head；E8 是本研究回合，不是整體總訓練回合。bridge 只有訓練 one-to-one 梯度路徑改變，推論單次 MASF。固定梯度係數 0.012076444778011642，不是 QK scale 或 PWL 係數。

CPU 算術：P3 80×80×256，三個卷積共 475,136,000 MAC；含 BN affine 與 alpha 的未融合參數 75,777。P2 160×160×256 為 1,900,544,000 Conv MAC。元素操作另列，不宣稱整網 FLOPs／latency。原 shared P3 與 P3 bridge 推論的 MASF 卷積運算量相同；alpha=0 的原 Pose 消融仍計算 context，不能當作省算。

另更正比較解讀：Head control 與 bridge 是整體方案對照；純 bridge 對照為 `masf-head-fork-v1`。E8 bridge 比原生 fork 小幅改善，但仍低於無 MASF control 的 COCO overall／person。

## 困難、限制與未解事項

困難：無。未執行新 GPU、checkpoint 反序列化、整网 profile 或硬體測速，不提供缺乏同口徑依據的整網百分比／延遲推估。BN folding、memory traffic 與 BinaryQK 硬體成本須另行計算。原 Pose 消融保留歷史旁證，不重新納入新方向決策。停止狀態維持，不恢復任何 queue；無刪除、commit 或 push。
