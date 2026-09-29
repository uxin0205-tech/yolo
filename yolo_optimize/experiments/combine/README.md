# 階段 2：Detect＋Pose 融合

> 本目錄已實際移入 `experiments/`。內部 run 名與模型內容保留；操作請見[整理後指南](<../../docs/OPERATIONS.md>)。

| 區域 | 用途／狀態 |
| --- | --- |
| [bridge_v1](<bridge_v1/README.md>) | 使用者選定 P3 bridge 的新融合主線，已完成融合與恢复試驗 |
| [pose-masf](<pose-masf/RESULTS.md>) | 現有 MASF 權重的 COCO／BBAT 比較及架構說明 |
| artifacts/、full35/、configs/ | 初期無 MASF J0 與準備流程的原始產物，歷史保留 |
| monitor.py | 共用 600 秒子工作監測工具；本次沒有新 queue |

詳細[新舊 combine 與恢復結果](<bridge_v1/BBAT_RECOVERY_RESULTS.md>)。目前新模型较保護 COCO，但 bat 等 BBAT 指標仍有缺口；沒有原嚴格 gate 全過的新模型。

使用者實驗 4 PDF 已移至[文件參考區](<../../docs/references/README.md>)。其他研究程式、設定、checkpoint 與 artifacts 路徑不改，避免破壞來源載入。舊入口與初期排程見[歷史快照](<../../docs/history/README.md>)，不要直接啟動舊 run_pose_pair.py。
