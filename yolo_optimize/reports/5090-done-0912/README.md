# 5090 Done 0912：從 BinaryQK 到推論的完整優化報告

這是 2026-09-12 的 GitHub 發布入口，將從 BinaryQK 開始的研究脈絡、實際優化、失敗原因、訓練配方與最終推論結果集中整理。提交名稱固定為 `5090 Done 0912`。

先讀[全階段詳細總報告](<../consolidated-20260911/README.md>)，再按下表查看各階段證據。所有分數須區分模型、資料、backend、EMA／live、parent 與階段 gate；不是把每項最高分拼成同一個模型。

## 研究順序

| 順序 | 研究內容與報告 | 已確認的結論 |
| --- | --- | --- |
| 0 | [早期 BinaryQK 研究](<../../../yolo_binaryqk/README.md>)／[YOLO26 PWL Attention](<../../../yolo_attention_final/README.md>) | 前者是 YOLO11m 歷史研究，不能把其 26 組消融直接當成後來 YOLO26M 的結果 |
| 1 | [BinaryQK 精度恢復](<../../optimizations/binaryqk-accuracy-recovery/README.md>)／[固定 scale 與 codebook](<../../optimizations/binaryqk-scale-codebook/README.md>) | 區分動態與固定尺度、硬體代價；沒有把未實施的 8-scale selector 說成現行方案 |
| 2 | [正式 attention 梯度斷點](<../../docs/worklogs/2026-09-09-attention-gradient-recovery.md>)／[融合前方向 1](<../direction1-20260910/README.md>) | 布林 XNOR／整數 reduction 切斷 Q/K 梯度；保留精確前向的 surrogate 可修訓練通道，但未補回全部 FP 缺口 |
| 3 | [早期融合後完整稽核](<../../optimizations/integrated-roadmap/round1-audit.md>)／[EMA 診斷](<../../optimizations/integrated-roadmap/ema-age-diagnostic-results.md>) | 早期融合後與後來融合前 parent 不同；EMA、BN、LR 有影響，但沒有單一充分解法 |
| 4 | [HOG 結果](<../../docs/worklogs/2026-09-09-hog-result-ball-bat.md>)／[RepConv17 結果](<../../docs/worklogs/2026-09-09-prefusion-rep17-result-masf-off.md>) | HOG、單點 RepConv 未達增準門檻；不全面替換或無限加訓 |
| 5 | [MASF P3／P2 結果](<../../combine/pose-masf/RESULTS.md>)／[架構與成本](<../../combine/pose-masf/ARCHITECTURE.md>) | shared／fork／bridge 接線不同；P2 不加 head 仍未達門檻；使用者選 P3 bridge 繼續研究，不是因為已證明它全面最佳 |
| 6 | [重新 combine 與 BBAT 恢復](<../../combine/bridge_v1/BBAT_RECOVERY_RESULTS.md>) | 新共享模型較保護 COCO，但 ball／bat 尤其 bat 與舊 combine 仍有缺口 |
| 7 | [Activation 配對](<../../activation/bridge_v1/RESULTS.md>) | qSiLU E2 有 ball 收益；Hardswish／PolyShift 未擴訓；GPU 慢不等於硬體設計失效，也不代表板端已加速 |
| 8 | [雙教師 KD](<../../kd/dual_task_v1/PLAN.md>)／[Pose-head KD](<../../kd/pose_focus_v1/README.md>) | 任務分流教師避免混資料標籤；兩輪都沒有新的合格 best_joint |
| 9 | [原框＋KD 關鍵點重組](<../../inference/pose_branch_v1/README.md>) | 框精確保留，但 KD 關鍵點收益沒有保住，不升版 |
| 10 | [one2many＋NMS 推論](<../../inference/routing_v1/README.md>) | bat Pose +0.03244，但 ball Pose −0.01620；不全面切換，class routing 尚未驗證 |

## 目前模型與重點結果

新主線仍保留 qSiLU P3 bridge E2：BitTrue COCO 0.503885、person 0.625887、BBAT 框 0.618008、Pose 0.891329。這不是全局最優宣告，也沒有通過最初所有獨立模型融合門檻。旧 combine 在 bat 上仍較好；沒有用新的階段 baseline 掩蓋差距。

PWL 已確認為 `[-10,0]`、20 段、固定表格；qSiLU 也不是逐圖動態 scale。完整訓練設定、原本／修改後架構圖、MAC 推導與失敗修正都在總報告，精確數值可讀[當前比較 CSV](<../consolidated-20260911/current-model-comparison.csv>)。

## 保存與發布範圍

本機已保存 406 個 checkpoint／模型產物，主封存總檔案量 113.66 GB，各副本與來源 SHA-256 一致。可查[checkpoint 索引](<../consolidated-20260911/checkpoints.csv>)、[run 索引](<../consolidated-20260911/runs.csv>)、[保存稽核](<../consolidated-20260911/PRESERVATION.json>)。這是同磁碟保存，不是異機備份；索引存在不代表模型 bytes 已上 GitHub。

本次 GitHub 發布報告、CSV／JSON 指標、設定與必要研究程式；不發布 `.pt`／`.onnx`／engine、113 GB 封存副本、runtime 資料集影像／labels、cache、訓練逐步大 log 或使用者／第三方 PDF。原始本機檔案全部保留，不刪除。歷史報告內的本機限定連結會明確標記，已發布的檔案使用 GitHub 相對連結；發布轉換清單附於 `publication-manifest.json`。

程式作為方法追溯資料提供，仍有自訂來源、絕對路徑與外部權重依賴，不宣稱 clone 後可直接重跑全部實驗。現有基準與完整 GPU 驗證重用已完成證據，本次不訓練、不重新跑正常 job。

## 下一步與未完成

優先研究固定 class routing：ball 用 one2one、bat 用 one2many，但要實測 AP 及雙分支成本，不能直接拼接最佳指標。共享 Conv 小範圍適應、區域／候選一致性 KD、定點 reciprocal 保留為條件式方向。沒有新背景 queue；person-only、PTQ／QAT 不因此自動重啟。

資料固定 COCO80 與 canonical BBAT5 v1；多 seed、獨立 test、使用者實際影片最終驗收、板端延遲／能耗仍未完成。完整工作脈絡見[中文工作紀錄索引](<../../docs/worklogs/README.md>)。
