# 實驗工作區

[Attention 恢復研究計畫](<attention_recovery_v1/README.md>)：native QK E2 已完成並暫停；新 scale/bias 與後續 Rep 三組已完成，見 [0914 完整更新](<../reports/update-0914/README.md>)。

此處是實際程式、原始結果與 checkpoint 所在位置，不是連到舊根目錄的捷徑。各階段保留內部目錄關係，避免把不同 parent 的 run 混合。

| 閱讀順序 | 目錄 | 內容與結果入口 |
| --- | --- | --- |
| 歷史獨立支線 | [artifacts](<artifacts/README.md>)＋[scripts](<scripts/README.md>) | 早期融合後 BEST、EMA、HOG、RepConv 與診斷 |
| 1 | [studies](<studies/README.md>) | 融合前 full35-B100：BinaryQK／HOG／RepConv／MASF |
| 2 | [combine](<combine/README.md>) | BBAT 對照、Pose 適應、共享融合與恢復 |
| 3 | [activation](<activation/README.md>) | SiLU／qSiLU 配對與部署重建 |
| 4 | [kd](<kd/README.md>) | 雙任務 KD、Pose-head KD |
| 5 | [inference](<inference/README.md>) | 分支重組、one2many＋NMS |
| 共用 | [src](<src/README.md>)／[tests](<tests/README.md>) | 共用模組與 CPU 契約測試 |

輸入是外部唯讀來源與本階段選定 parent；輸出放所屬階段的 artifacts，不再寫到 optimize 根層。完整報告在 [reports/final](<../reports/final/README.md>)，提案在 [proposals](<../proposals/README.md>)。

使用方式及依賴見[操作指南](<../docs/OPERATIONS.md>)。歷史 runner／queue 已完成或停止，不因搬移而重啟。Git 只發布文字指標、必要設定與程式，不發布 checkpoint、cache 或 runtime 影像／labels；本機全部保留。

[benchmark：跨階段推論成本實驗](<benchmark/README.md>)，不重新訓練。

[Pose MASF 已完成移接分析](<pose_masf_priority_v1/RESULTS.md>)與 [B 組專項架構／CPU 檢查](<pose_masf_training_v1/README.md>)：A 已取消，B 的 5E／驗證／分析已完成。

[BinaryQK 後接 Rep 三組](<post_binary_rep_v1/README.md>)：Rep17／Rep20／Rep17＋20 全部完成，未過採用閘。
