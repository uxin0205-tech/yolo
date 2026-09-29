# YOLO Optimize：研究、結果與權重索引

## 最新：0914 本輪已完成

[先讀 0914 更新總入口](<reports/update-0914/README.md>)：MASF B5 → BinaryQK scale/bias E5 → Rep17／Rep20／Rep17＋20，包含完整八項 AP、架構圖、超參數、硬體界線與 checkpoint 索引。三組於台北時間 2026-09-14 04:49:57 全部完成，均未過採用門檻，未自動替換原模型。

[目前正式選用模型](<reports/current-model/README.md>)仍為既有 qSiLU E2，不把新實驗中的 best 或最後 E5 混稱為正式版本。新候選含 Detect／Pose 兩套 MASF；舊正式版本只有 Detect P3 MASF。

## 文件与資料夾

| 入口 | 用途 |
| --- | --- |
| [本輪完整數據與分析](<experiments/post_binary_rep_v1/STATUS-20260914.md>) | 六階段比較、結果解讀與未量測項目 |
| [scale/bias 硬體核對](<experiments/post_binary_rep_v1/SCALE-BIAS-HARDWARE.md>) | 固定常數／查表與尚未完成的純整數部分 |
| [全階段總報告](<reports/final/README.md>) | 從 BinaryQK、HOG、MASF、融合、activation、KD 到推論的歷史 |
| [各階段閱讀順序](<reports/publication/README.md>) | 歷史報告導覽 |
| [九項效能指標歷史報告](<reports/performance/README.md>) | 已量測舊模型成本，不冒充新三組實測 |
| [Checkpoint 索引](<reports/checkpoints/README.md>) | 本機權重位置與 SHA；0914 新增 53 份稽核 |
| [experiments](<experiments/README.md>) | 程式、設定、原始結果與本機權重 |
| [proposals](<proposals/README.md>) | 原始假說與尚未執行方向 |
| [目錄地圖](<docs/WORKSPACE.md>) | 各目錄用途與歷史遷移 |
| [工作紀錄](<docs/worklogs/README.md>) | 中文修改、驗證、困難與風險 |
| 封存索引（本機保存：`archives/README.md`；本次未上傳） | 歷史保留副本，非目前訓練入口 |

## 資料規範與發布

COCO80 固定使用 `/home/uxin/yolo/coco2017.yaml`；BBAT5 固定 `/home/uxin/yolo/original/pose/derived/bbat5-v1/`，不重切、不抽換影像與 labels。詳見[共用資料集規範](<../docs/agents/bbat5-datasets.md>)。

Git 只發布報告、圖、程式、設定与必要數值證據；checkpoint、runtime 資料集、cache、大 log／PDF 全留本機，不刪除。0914 commit 使用使用者指定的 `5090 Uodate 0914`，發布範圍與驗證見 [0914 入口](<reports/update-0914/README.md>)。
