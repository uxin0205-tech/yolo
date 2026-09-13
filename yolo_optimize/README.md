# YOLO Optimize：完整研究與交付

[先讀這份：目前到底用哪些技術？架構圖與移除 BinaryQK 重訓說明](<reports/current-model/README.md>)

[教授版完整研究報告：從 BinaryQK、MASF、融合、Activation、KD 到推論與失敗實驗](<reports/professor-overview/README.md>)

從 BinaryQK 精度恢復開始，依序研究 HOG、RepConv、MASF、Detect＋Pose 融合、activation、KD 與推論。**正式結論以詳細報告為準；候選檔名的 best 不等於全指標最佳。**

- [全階段詳細報告：架構、數據、推導、超參數與限制](<reports/final/README.md>)
- [各階段閱讀順序](<reports/publication/README.md>)
- [Checkpoint 導航與 SHA-256](<reports/checkpoints/README.md>)
- [新目錄地圖、舊新路徑與使用方式](<docs/WORKSPACE.md>)

## 實體分類

| 資料夾 | 內容 |
| --- | --- |
| [experiments](<experiments/README.md>) | 全部階段程式、原始產物、模型、共用模組與測試 |
| [reports](<reports/README.md>) | 完整總報告、方向 1 細節、模型索引、GitHub 發布紀錄 |
| [proposals](<proposals/README.md>) | 原始假說與尚未完成的優化方向，不冒充實測成果 |
| [docs](<docs/README.md>) | 研究推導、參考 PDF、工作紀錄與歷史原文 |
| [tools](<tools/README.md>) | 封存、整理、文件發布工具；不自動訓練 |
| archives（本機保存：`archives/README.md`；本次未上傳） | 已驗證的固定保存副本，不作日常工作目錄 |

## 目前結論

新研究預設為 qSiLU P3 bridge E2：BitTrue COCO AP 0.503885、person AP 0.625887、BBAT 框 AP 0.618008、Pose AP 0.891329。它比舊 combine 更保護 COCO，但 bat 仍有缺口；HOG、RepConv、KD 與推論候選並未產生全面勝出的替代模型。

MASF 保留 P3 bridge 是使用者選定的研究路線，配對消融沒有證明其獨立增準。PWL 固定 `[-10,0]`、20 段；qSiLU 不是逐圖動態 scale。沒有因整理重跑 GPU 或建立新 queue。

## 資料與保存

COCO80 使用 `/home/uxin/yolo/coco2017.yaml`；BBAT5 固定使用 `/home/uxin/yolo/original/pose/derived/bbat5-v1/` 的正式 Task View，不重新切分或更動影像／labels。見[資料集規範](<../docs/agents/bbat5-datasets.md>)。

全部權重已隨實驗資料夾搬到新位置，內容不變；舊路徑不留根層捷徑。[工作紀錄](<docs/worklogs/README.md>)與[遷移清單](<docs/history/layout-v2-20260912/manifest.json>)保存過程。GitHub 更新使用 commit 名稱 `5090 Done 0912`，發布報告、數值證據、設定及程式，不把本機大型權重、封存包、runtime 資料或 PDF 混進一般 Git。

歷史發布紀錄：整理版 `8786b58f` 曾因發布確認不足受阻；其後使用者明確確認，已於 `4f4131e` 一併發布成功。先前受阻證據保留於歷史交付狀態（本機保存：`reports/publication/PUBLISHED.md`；本次未上傳），成功收據見[2026-09-12 發布紀錄](<reports/performance/PUBLISHED-4f4131e.md>)。

## 各階段精度與效能

已完成 26 組代表／配對的 CPU／GPU 成本比較，詳見[九項指標報告](<reports/performance/README.md>)與[精確 CSV](<reports/performance/comparison.csv>)。包含 Params、模型大小、MAC／FLOPs subtotal、峰值記憶體、CPU／GPU 延遲及 GPU 能量；目標板卡尚未指定，target latency／energy 明記未量測。

2026-09-13：先完成白話報告與發布；新增 scale／bias CPU 稽核，現行權重不變。PWL [-10,0]／20 段已確認。

2026-09-13 後續已執行原生 QK＋PWL Attention 恢復；E2 已完成完整驗證與 checkpoint 保存後依指示暫停，scale_bias 未接續。見[實驗與狀態入口](<experiments/attention_recovery_v1/README.md>)。

2026-09-13 最新：[Pose 端 MASF 完整比較與分析](<experiments/pose_masf_priority_v1/RESULTS.md>)已完成：直接移接未帶來整體 Pose AP 收益，COCO 不變；權重、全量驗證、成本量測與來源稽核均保存。

2026-09-13：[Pose MASF 專項重訓推導與完整架構圖](<experiments/pose_masf_training_v1/README.md>)已整理；A 組取消，只保留 B 組 5 epoch。B 程式與 CPU 檢查完成，GPU 與等待佇列尚未啟動。
