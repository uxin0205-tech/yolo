# 工作紀錄索引

- [2026-11-20 工作區整理與 GitHub 1119 交付](2026-11-20-workspace-github.md)

- [2026-11-19 硬體與精度雙選擇、形狀導讀及根層清理](2026-11-19-selections-and-cleanup.md)

- [2026-11-17 全輪研究整理、統一交接與 1.2074 GiB 清理](2026-11-17-integrated-research-handoff.md)

- [2026-11-17 768×512 完成 20 輪、方向比較與交接核驗](2026-11-17-actshape-portrait-results.md)

- [2026-11-12 Stage9 新增 768×512，同配置 10E＋半 LR 10E](2026-11-12-actshape-portrait-training.md)

- [2026-11-11 Stage9 全部 100 輪完成、最終選型與交付](2026-11-11-actshape-stage9-results.md)

- [2026-11-11 Stage9 最後一組第 9 輪、88 輪進度與精度查核](2026-11-11-actshape-status.md)

- [2026-11-10 Stage9 已完成 84 輪、最後一組精度與 GPU 查核](2026-11-10-actshape-status.md)

- [2026-11-10 Stage9 梯度校準中斷、optimize 核對與有界恢復](2026-11-10-actshape-calibration-recovery.md)

- [2026-11-09 Stage9 CUDA 原生崩潰查核與一次恢復](2026-11-09-actshape-cuda-segv.md)

- [2026-11-09 Stage9 電腦重啟後從中斷處恢復](2026-11-09-actshape-reboot-recovery.md)

- [2026-11-09 Stage9 訓練進度與目前精度](2026-11-09-actshape-status.md)

- [2026-11-08 Stage9 新增硬體尺寸與 GPU 分塊驗證](2026-11-08-actshape-expanded-gpu.md)

- [2026-11-07 RepConv 原生訓練與硬體部署](2026-11-07-repconv-native-training.md)

本目錄保存 `/home/uxin/yolo` 及其子專案的中文工作紀錄，讓程式修改、實驗、驗證、困難與解法可被追溯。

## 紀錄規則

1. 每項修改、實驗或驗證建立或更新一份 `YYYY-MM-DD-主題.md`。
2. 每份紀錄至少包含：任務與範圍、變更內容與原因、驗證方式與結果、困難與解法、未解事項或風險。
3. 沒有遇到困難時，仍在「困難與解法」明記「無」。
4. 資料集相關工作另記實際絕對路徑、任務類型、split，以及是否改動樣本或標註。
5. 新增紀錄時，同步在下方索引加入連結；所有內容使用中文，必要的程式名稱、指令、路徑與原文除外。

## 紀錄範本

```markdown
# YYYY-MM-DD：工作主題

## 任務與範圍

## 變更內容與原因

## 驗證方式與結果

## 困難與解法

## 未解事項或風險
```

## 索引

- [2026-11-08：固定 qSiLU Rep17＋20、清理與 Stage9 特徵圖研究](2026-11-08-actshape-stage9.md)

- [2026-11-08：使用者選定 qSiLU 訓練版 Rep17＋20](2026-11-08-stage8-qsilu-selection.md)

- [2026-11-08：Stage8 完成、無 Rep 比較推薦與中斷恢復](2026-11-08-stage8-results-comparison.md)

- [2026-11-08：Stage8 進度查核與 SiLU 訓練中斷恢復](2026-11-08-stage8-silu-recovery.md)

- [2026-11-07：指定 yolo_actshape 特徵圖研究工作區](2026-11-07-actshape-workspace.md)

- [2026-11-04：確認 Rep 訓練與部署的算子要求](2026-11-04-repconv-deployment-contract.md)

- [2026-11-04：更正 Rep 訓練方式解讀](2026-11-04-repconv-training-clarification.md)

- [2026-11-04：Rep folding 說明與訓練算子調整提案](2026-11-04-repconv-training-proposal.md)

- [2026-11-04：Stage8 完成狀態查核與停止接續](2026-11-04-stage8-status-hold.md)

- [2026-11-03：第八階段 qSiLU RepConv 實作與 GPU queue](2026-11-03-repconv-stage8.md)

- [2026-11-03：全部 RepConv 候選的算子簡化契約](2026-11-03-repconv-operator-contract.md)
- [2026-11-03：第七階段完成、資料夾清理與可攜模型交接](2026-11-03-activation-stage7-handoff.md)
- [2026-11-02：第七階段 Activation 比較與 GPU queue](2026-11-02-activation-stage7.md)

- [2026-10-31：第六階段清理與第七階段準備](2026-10-31-stage6-cleanup-stage7-preparation.md)

- [2026-10-30：第六階段結果、optimize 同設定重驗與交付](2026-10-30-stage6-results.md)

- [2026-10-29：KD 五輪完成與 Pose head 恢復](2026-10-29-stage6-kd-results-head-recovery.md)

- [2026-10-29：KD 恢復進度與 optimize 最終模型比較](2026-10-29-stage6-optimize-comparison.md)

- [2026-10-29：J1 分組診斷與雙教師恢復](2026-10-29-stage6-joint-recovery.md)

- [2026-10-29：完整 Pose 適應完成，進入 J1 聯合訓練](2026-10-29-stage6-joint-progress.md)

- [2026-10-29：融合重建進度，完整 Pose 分支改善](2026-10-29-stage6-pose-progress.md)

- [2026-10-29：以最佳融合結果為目標，optimize 作為參考](2026-10-29-stage6-best-fusion-objective.md)

- [2026-10-29：H1 進度與 head 停止規則修正](2026-10-29-stage6-head-safety-fix.md)

- [2026-10-29：清除舊融合實驗並依 optimize 從頭重建](2026-10-29-stage6-clean-rebuild.md)

- [2026-10-28：核對目前 Softmax 的 PWL 與替代方法](2026-10-28-softmax-status-check.md)

- [2026-10-21：刪除 HOG 大型資源、補清 Stage2 smoke 與前階段稽核](2026-10-21-hog-resource-cleanup.md)

- [2026-10-21：HOG 第一輪精度停止、剝除重載驗證與交付](2026-10-21-hog-stage5-results.md)

- [2026-10-21：保留 MASF 結果並進入第五階段 HOG](2026-10-21-hog-stage5.md)

- [2026-10-20：第四階段 MASF 延長完成與交付](2026-10-20-masf-stage4-results.md)

- [2026-10-20：第四階段延長訓練進度核對](2026-10-20-masf-stage4-progress.md)

- [2026-10-19：第四階段延長與分組學習率比較](2026-10-19-masf-stage4-extension.md)

- [2026-10-19：採用 KD 第五輪並進入 task-specific MASF](2026-10-19-masf-stage4.md)

- [2026-10-19：第 3 階段 KD 五輪完成：小幅提升未達門檻、保留 D1](2026-10-19-attention-stage3-kd.md)

- [2026-10-19：第 3 階段完成、KD 判斷與舊產物清理](2026-10-19-attention-stage3-results.md)
- [2026-10-19：第 3 階段 normalization 與完整 PWL／reciprocal](2026-10-19-attention-stage3.md)
- [2026-10-19：第 2 階段完成、S1 選型與交付核對](2026-10-19-binaryqk-stage2-results.md)
- [2026-10-19：第 2 階段 Q10／二次冪 scale 與 bias](2026-10-19-binaryqk-stage2.md)

- [2026-10-18：B2 運算量、整網占比與 CPU 核對](2026-10-18-binaryqk-operation-cost.md)

- [2026-10-18：Binary 第 1 階段完成、B2 選型、恢復負結果與交付驗證](2026-10-18-binaryqk-stage1-results.md)

- [2026-10-18：依 optimize 重建初篩／恢復、直接刪除錯誤產物並啟動長訓](2026-10-18-binaryqk-recovery-v2.md)

- [2026-10-18：每項研究工作前先參考 optimize](2026-10-18-optimize-reference-before-work.md)

- [2026-10-18：Binary 狀態、0.51～0.52 精度與恢復配方稽核](2026-10-18-binaryqk-status-and-protocol-audit.md)

- [2026-10-18：Binary 第 1 階段重建、清理、驗證與背景工作交接](2026-10-18-binaryqk-stage1-rebuild.md)

- [2026-10-17：補充全階段硬體友善、特徵圖形狀分析與清理要求](2026-10-17-hardware-friendly-shapes-plan.md)

- [2026-10-03：記錄工作起點與 COCO 球棒／球評估目標](2026-10-03-coco-ball-bat-evaluation-goal.md)

- [2026-09-29：清空 GitHub main 內容並保留分支](2026-09-29-main-empty.md)

- [2026-09-29：建立 Cleanup-v1 程式與報告基線](2026-09-29-cleanup-v1.md)

- [2026-09-29：確認 GitHub 帳號與專案連結](2026-09-29-github-connection-verification.md)

- [2026-09-22：保存 YOLO 整合研究計畫與 qSiLU RepConv 接續契約](2026-09-22-yolo-integrated-research-plan.md)

- [2026-09-15：原始模型與最新模型 FLOPs 比較](2026-09-15-flops-comparison.md)

- [2026-09-15：原始模型與最新模型 Recall 比較](2026-09-15-recall-comparison.md)

- [2026-09-15：Rep17＋20 E5 與 YOLO26m 評估口徑稽核](2026-09-15-evaluation-protocol-audit.md)

- [2026-09-15：原始 YOLO11m 與棒球 Pose 權重評估](2026-09-15-original-weights-evaluation.md)

- [2026-09-10：MASF 結果稽核，P2 已完成但未達增準門檻](2026-09-10-masf-results-audit.md)

- [2026-09-09：Optimize 與 Quantize 訓練方法及約 5 倍時間差比較](2026-09-09-optimize-quantize-training-comparison.md)

- [2026-09-08：評估 Astra 輪詢消耗與直接使用 Luna MAX](2026-09-08-astra-polling-luna-evaluation.md)
- [2026-09-08：停用子代理，改採單一主代理](2026-09-08-disable-subagents.md)
- [2026-09-08：核對全域指令長度與 token 消耗因素](2026-09-08-agent-token-overhead.md)
- [2026-09-08：新增 GPU 工作每 600 秒監測規則](2026-09-08-gpu-monitoring-policy.md)
- [2026-09-08：整合 Codex 全域偏好與子代理規則](2026-09-08-codex-global-agent-preferences.md)
- [2026-08-22：建立 BBAT5 固定資料集與中文紀錄全域規則](2026-08-22-bbat5-global-memory.md)
- [2026-08-22：完成 architecture_2 Phase A 與 BBAT5 v1](2026-08-22-architecture-2-phase-a.md)
- [2026-08-22：發布 architecture_2 完整 BBAT5 GitHub Dataset](2026-08-22-architecture-2-github-dataset.md)
- [2026-08-22：統一全專案 BBAT5 資料集](2026-08-22-bbat5-dataset-unification.md)
- [2026-08-23：釐清 BBAT5 Pose／Detect Task View](2026-08-23-bbat5-task-views-clarification.md)
- [2026-08-23：發布 original 資料並清除 Git 權重](2026-08-23-original-data-publication.md)

- [2026-08-26：architecture_2 固定20%初篩與 QAT-lite](2026-08-26-architecture-2-float20-qat-lite.md)
- [2026-08-26：YOLO26m PWL Attention 可攜訓練包](2026-08-26-yolo-attention-portable-final.md)
- [2026-08-27：YOLO26m PWL Attention 可攜包清理](2026-08-27-yolo-attention-portable-final-cleanup.md)
- [2026-08-27：architecture_2 C2／C3完整訓練與量化自動接續](2026-08-27-architecture2-c2-c3-auto-continuation.md)
- [2026-08-27：YOLO26m final 完整 workspace、架構與創新文件](2026-08-27-yolo-attention-final-workspace-architecture.md)
- [2026-08-27：Full35 combined inference 權重大小診斷](2026-08-27-full35-combined-inference-weight-size.md)
- [2026-08-27：architecture_2 Full35 Float20 與安全 GPU 佇列](2026-08-27-architecture2-full35-float20-queue.md)
- [2026-08-27：architecture_2 Float20 成果匯出與 profiling 裝置修正](2026-08-27-architecture2-float20-profile-device-fix.md)
- [2026-08-28：architecture_2 不採用、清理與發布](2026-08-28-architecture2-archive-and-publication.md)
- [2026-08-30：architecture_2 C1～C3逐回合精度曲線](2026-08-30-architecture2-c1-c3-epoch-accuracy-curves.md)
返回[根層 README](../../README.md)。
