# 工作紀錄索引

本目錄保存 `yolo_optimize` 的中文修改、研究與驗證紀錄。

## 索引

- [2026-09-12：全階段精度與部署成本比較](<2026-09-12-stage-performance-comparison.md>)

- [2026-09-12：實體資料夾重整、詳細報告與發布](<2026-09-12-physical-layout-and-publication.md>)

- [2026-09-12：全工作區目錄整理與產物保護驗證](<2026-09-12-workspace-organization.md>)

- [2026-09-12：從 BinaryQK 到推論的 GitHub 報告發布](<2026-09-12-github-report-publication.md>)

- [2026-09-11：全階段報告、checkpoint 封存與推論接續](<2026-09-11-comprehensive-archive-and-routing.md>)

- [2026-09-11：推論分支重組研究與完整驗證](<2026-09-11-inference-branch-research.md>)

- [2026-09-11：取消原生對照，直接 KD 背景執行](<2026-09-11-pose-direct-background.md>)

- [2026-09-11：Pose head 專項啟動與共享特徵試驗授權](<2026-09-11-pose-focus-start.md>)

- [2026-09-11：KD 完成後的 Pose-only 分析與下一步規劃](<2026-09-11-pose-only-next-plan.md>)

- [2026-09-11：核准雙教師 KD、教師驗證與 MuSGD 前置](<2026-09-11-dual-teacher-musgd-start.md>)

- [2026-09-11：Activation 配對完成與大 Detect／BBAT5 Pose 教師前置](<2026-09-11-activation-result-teacher-options.md>)

- [2026-09-11：依最新指示先 Activation，再雙資料集 KD](<2026-09-11-activation-before-dual-task-kd.md>)

- [2026-09-11：J3 完成與新舊 combine 的 BBAT 差距診斷](<2026-09-11-j3-result-bbat-diagnosis.md>)

- [2026-09-11：使用者追加舊 combine 的 ball／bat 差距分析與恢復優先序](<2026-09-11-old-combine-bbat-priority.md>)

- [2026-09-11：J3 第 9 輪進度查詢與驗收狀態](<2026-09-11-j3-status.md>)

- [2026-09-10：J2 完成但尚未驗收、J3 最後低 LR 整體微調](<2026-09-10-balanced-j2-result-j3.md>)

- [2026-09-10：J1完成、COCO過門檻但bat待恢復，J2低LR接續](<2026-09-10-balanced-j1-result-j2.md>)

- [2026-09-10：任務梯度校準0.045與最終／訓練安全分離，J1正式啟動](<2026-09-10-balanced-j1-start.md>)

- [2026-09-10：閱讀使用者 combine PDF 實驗 4，核對梯度與停止門檻](<2026-09-10-combine-pdf-experiment4.md>)

- [2026-09-10：完整 Pose 達 0.8980、共享相容性與融合 J0 啟動](<2026-09-10-full-pose-result-merge.md>)

- [2026-09-10：完整 Pose 首輪停止與同起點 LR×0.1 受控嘗試](<2026-09-10-full-pose-gentle.md>)

- [2026-09-10：Pose head 40 epoch 結果與完整 Pose AdamW 正式啟動](<2026-09-10-full-pose-adamw-start.md>)

- [2026-09-10：完整 Pose 模型先訓練、AdamW 主線與融合相容性](<2026-09-10-full-pose-before-fusion.md>)

- [2026-09-10：J1 退化診斷、先完成 Pose head 適應再融合](<2026-09-10-bridge-pose-first-correction.md>)

- [2026-09-10：P3 bridge MASF 選定、α 開關驗證與融合 J1 重新啟動](<2026-09-10-bridge-combine-restart.md>)

- [2026-09-10：MASF 論文與 BBAT5 小框 CPU 統計](<2026-09-10-masf-small-object-review.md>)

- [2026-09-10：BinaryQK／HOG／MASF 收益歸因與成對結果](<2026-09-10-qk-hog-masf-attribution.md>)

- [2026-09-10：Bridge E8 相對 B100／原生 MASF／無 MASF 的比較與原方案](<2026-09-10-bridge-baseline-comparison.md>)

- [2026-09-10：Head 對照／P3 bridge 架構、訓練梯度與運算量](<2026-09-10-masf-architecture-compute.md>)

- [2026-09-10：新方向 checkpoint 與 BBAT5 Detect／Pose 比較範圍更正](<2026-09-10-new-direction-bbat-scope.md>)

- [2026-09-10：現有 MASF 權重的 COCO／Pose 資料集直接驗證，完成後停止](<2026-09-10-existing-masf-pose-evaluation.md>)

- [2026-09-10：P2 MASF 結果、無 MASF 融合與 Pose 專項接續](<2026-09-10-p2-result-no-masf-fusion.md>)

- [2026-09-10：最後 P2 MASF 配對與 [-10,0] PWL 契約](2026-09-10-p2-masf-last-trial.md)
- [2026-09-10：Combine 來源相容性稽核與停用計畫（等待候選選擇）](<2026-09-10-combine-source-audit.md>)
- [2026-09-10：方向 1 候選完整驗證與階段整理](<2026-09-10-direction1-verification-close.md>)
- [2026-09-10：方向 1 MASF 真實 loss 校準與梯度橋接訓練](<2026-09-10-masf-task-bridge-training.md>)
- [2026-09-10：MASF E10 驗收、梯度路徑診斷與等待限制](<2026-09-10-masf-e10-gradient-and-wait.md>)
- [2026-09-09：MASF E5 結果與 P3 head 配對適應續訓](<2026-09-09-masf-e5-head-adaptation.md>)
- [2026-09-09：融合前 MASF 增準與 P3 Detect-only 最小位置實驗](<2026-09-09-prefusion-masf-p3-start.md>)
- [2026-09-09：融合前 Rep17 驗收與 B100 MASF 零殘差診斷](<2026-09-09-prefusion-rep17-result-masf-off.md>)
- [2026-09-09：600 秒等待與模型喚醒核對](<2026-09-09-monitor-wakeup-audit.md>)
- [2026-09-09：融合前 layer 17 單點 RepConv 配對與部署驗證](<2026-09-09-rep17-prefusion-start.md>)
- [2026-09-09：HOG 未通過、ball／bat 監督覆蓋與 head BN 診斷](<2026-09-09-hog-result-ball-bat.md>)

- [2026-09-09：依使用者要求，HOG／MASF 加看 COCO ball／bat](<2026-09-09-hog-masf-ball-bat-observation.md>)

- [2026-09-09：E10 收斂判斷與融合前 HOG 原生對照、梯度校準](<2026-09-09-e10-hog-prefusion.md>)

- [2026-09-09：E8 Backbone 解凍結果、完整匯出推論驗證與 E10 延長](<2026-09-09-e8-backbone-inference.md>)

- [2026-09-09：E5 結果與同起點 Backbone 後段適應實驗](<2026-09-09-e5-backbone-adaptation.md>)

- [2026-09-09：E3 結果、E5 成對延長與 Backbone 分段授權](<2026-09-09-e3-review-backbone-scope.md>)

- [2026-09-09：COCO overall／person 優先、取消 ball／bat gate 與成對續訓](<2026-09-09-coco-person-continuation.md>)

- [2026-09-09：Attention 正式 score 梯度斷點、同口徑基準與分段恢復](<2026-09-09-attention-gradient-recovery.md>)

- [2026-09-09：另開融合前 B100 方向 1、來源稽核與基準驗證](<2026-09-09-prefusion-b100-start.md>)

- [2026-09-09：第一輪需求稽核、可重建比較表與部署授權邊界](<2026-09-09-round1-requirement-audit.md>)

- [2026-09-09：BN、MuSGD 更新校準與 J2 BEST 重驗結果](<2026-09-09-bn-musgd-j2-results.md>)

- [2026-09-09：B-HEAD 五 epoch 結果與子分支更新稽核](<2026-09-09-heads-result-update-audit.md>)

- [2026-09-09：AP 排序拆解、雙向 scope 診斷與 B-HEAD 修復](<2026-09-09-ranking-scope-head-recovery.md>)

- [2026-09-09：依原計畫繼續，補齊全量 Pose 錯誤與同圖對照](<2026-09-09-plan-continuation-pose-errors.md>)

- [2026-09-09：QK 兩點診斷完成、固定 scale 等價修正與第一輪決策](<2026-09-09-qk-results-fixed-scale.md>)

- [2026-09-09：MASF bridge 未過，接續 BinaryQK 單點 score 診斷](<2026-09-09-bridge-result-qk-diagnostics.md>)

- [2026-09-09：MASF 零殘差診斷完成，接續 BR-OFF5](<2026-09-09-masf-off-bridge.md>)

- [2026-09-09：RepConv 五 epoch 結果與 MASF 零殘差診斷接續](<2026-09-09-rep17-result-masf-diagnostic.md>)

- [2026-09-09：RepConv layer17 正式接線與連續安靜監測](<2026-09-09-rep17-training-integration.md>)

- [2026-09-09：HOG 結束、安靜監測與 RepConv 單點前置驗證](<2026-09-09-hog-result-repconv-preflight.md>)

- [2026-09-08：HOG 接續、校準修正與 400 秒監測](<2026-09-08-hog-calibration-monitor.md>)

- [2026-09-08：Native5、LR×0.25 與 BN 對照（已完成，精度未通過）](<2026-09-08-native5-and-quarter-lr.md>)
- [2026-09-08：單一主代理、取消配額門檻與 400 秒監測](<2026-09-08-single-agent-400s-monitor.md>)
- [2026-09-08：Parent EMA 原生對照（E1–E5 已完成，E5 安全暫停）](<2026-09-08-parent-ema-native-control.md>)
- [2026-09-08：EMA age 單變因診斷（已完成）](<2026-09-08-ema-age-diagnostic.md>)
- [2026-09-08：方向 1 第一輪執行紀錄（E1 分析暫停、診斷與修補完成）](<2026-09-08-direction1-execution.md>)
- [2026-09-08：方向1最佳候選與 optimizer 整合](<2026-09-08-direction1-best-optimizer-integration.md>)
- [2026-09-08：Final readiness 修正與接續來源盤點](<2026-09-08-final-readiness-correction.md>)
- [2026-09-08：訓練模型恢復計畫](<2026-09-08-trained-model-recovery-plan.md>)
- [2026-09-08：第二輪創新候選研究／joint蒸餾適用性釐清](<2026-09-08-round2-innovation-research.md>)
- [2026-09-08：R2-REGION 區域排序完整規格](<2026-09-08-region-ranking-full-spec.md>)
- [2026-09-08：整合優化總計畫](<2026-09-08-integrated-optimization-roadmap.md>)
- [2026-08-31：YOLO 架構、RepConv 與 BinaryQK 精度恢復研究](<2026-08-31-yolo-repconv-binaryqk-research.md>)
- [2026-09-01：MASF 精度回歸原因診斷](<2026-09-01-masf-regression-diagnosis.md>)
- [2026-09-01：YOLO26 P3 MASF 無實質增益專項診斷](<2026-09-01-yolo26-p3-masf-no-gain-diagnosis.md>)
- [2026-09-01：P3 MASF 優化方向與計畫整理](<2026-09-01-p3-masf-optimization-direction-organization.md>)
- [2026-09-01：P3 MASF 終端架構圖報告](<2026-09-01-p3-masf-terminal-architecture-report.md>)
- [2026-09-01：P3 MASF 主訓練後最小實驗計畫](<2026-09-01-p3-masf-post-base-training-minimal-plan.md>)
- [2026-09-02：BinaryQK 精度恢復方向整理](<2026-09-02-binaryqk-accuracy-recovery-direction.md>)
- [2026-09-03：BinaryQK 少量 scale／codebook 決策整理](<2026-09-03-binaryqk-scale-codebook-decision.md>)
- [2026-09-04：BinaryQK scale／codebook 獨立資料夾整理](<2026-09-04-binaryqk-scale-codebook-folder-organization.md>)
- [2026-09-04：早期論文解讀與衝突安全訓練整理（後續已更正定位）](<2026-09-04-paper31-conflict-safe-training-direction.md>)
- [2026-09-04：論文第 3.1 節對 YOLO26M 的適配更正／P3 HOG 訓練方向](<2026-09-04-paper31-yolo26m-training-adaptation-correction.md>)
- [2026-09-04：COCO person-only 專用 head 方向整理](<2026-09-04-coco-person-specialized-head-direction.md>)

返回[子專案 README](<../../README.md>)。

- [2026-09-12：確認後發布效能報告](<2026-09-12-performance-publish-confirmed.md>)

- [2026-09-13：白話模型說明、開關診斷與 Attention 重訓分流](<2026-09-13-current-model-attention-recovery.md>)

- [2026-09-13：白話報告、BinaryQK 稽核與 GitHub 交付](<2026-09-13-report-update-0913.md>)
