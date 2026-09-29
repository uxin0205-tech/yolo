# Pose 端 P3 MASF 比較

## 已完成：直接移接沒有整體精度收益

2026-09-13：原 Pose／Pose P3 MASF／α=0 的全量推論比較、全量特徵殘差診斷、CPU／GPU 成本量測、候選權重重建與 SHA 稽核全部完成。**完整分析、接線圖、精度與成本表請直接閱讀 [RESULTS.md](<RESULTS.md>)。**

ball 關鍵點 AP 上升 0.0470 個百分點，bat 關鍵點下降 0.0487 個百分點，整體 Pose AP 幾乎持平而略降；COCO 全指標不變。暫不以移接候選取代原 Pose。這一步沒有訓練新增 Pose MASF，不能據此否定 Pose 專項訓練可能的收益。

## 固定比較範圍

同一份已暫停 native QK E2 融合模型：YOLO26M、qSiLU、原生 QK＋PWL [-10,0] 20 段、既有 Detect P3 bridge MASF。候選只在 Pose head 的 P3 入口加入 Detect MASF 的獨立參數副本，不改共享 Neck 或 Detect。新分支增加 75,777 個參數。

資料固定為 COCO80 val 5,000 張與 canonical BBAT5 v1 Pose Task View val 683 張；正式 YAML 為 `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml`，registry 為 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。不重切、不抽樣、不改 labels。

## 主要產物

- [完整結果與分析](<RESULTS.md>)、[精度 CSV](<metrics.csv>)、[成本 CSV](<costs.csv>)。
- [固定來源與 SHA](<source-pin.json>)、候選權重（本機保存：`artifacts/comparison-v1/pose_masf/pose-masf-transferred-inference.pt`；本次未上傳）。
- [原始比較 JSON](<artifacts/summary-v1.json>)、[量測 JSON](<artifacts/benchmark-v1.json>)、[特徵診斷](<artifacts/residual-diagnostic-v1.json>)。
- [CPU 前置驗證](<artifacts/preflight-v1.json>)、[最終稽核](<artifacts/final-audit-v1.json>)、[保留／清理盤點](<CLEANUP.md>)。
- [中文工作紀錄](<../../docs/worklogs/2026-09-13-pose-masf-priority.md>)。

## 執行與重建

`run_priority.py` 預設只顯示完成狀態。帶 `--execute` 也會跳過已成功的比較項目，不覆寫既有結果。`build_report.py` 依已保存 JSON 重建報告與 CSV，不使用 GPU。

`pose_candidate.py` 保存模型重建接法；`compare.py` 是正式推論比較；`benchmark.py` 與 `diagnose_residual.py` 是獨立完成的量測／分析；`audit_final.py` 核對 export、數值與來源。候選是需以此客製架構重建的 state-dict-only checkpoint，不是完全原生 YOLO 的通用權重。

後續 Pose 專項等預算訓練只是建議，未啟動。原 Attention E2 續訓與 scale_bias 仍受 pause-request 保護，**完成本比較不會自動解除暫停**。若要重做實驗，新增版本目錄，不覆寫本次成果。

## 保存與 Git 政策

權重、驗證結果、runtime cache 與 provenance 均保留，原檔不刪除、不覆寫。本次沒有 commit／push；後續若發布，應另行確認權重與資料的發布範圍，不把本機 cache 或重複權重默認加入 Git。

## 後續設計（不是新增訓練成果）

[Pose MASF 專項重訓推導與完整架構圖](<../pose_masf_training_v1/README.md>)已另外整理。提案的 β=1、α=0 與累積 8 次尚未套用到本目錄候選；本次移接結果與程式保持原狀。

## 2026-09-13 後續狀態更正

本目錄移接比較與權重保持不變。後續 A 加訓已取消；B 組新程式與 CPU 檢查完成，GPU／等待佇列未啟動。使用者要求先以 `5090 Done 0913` 發布現有圖與分析；上方「沒有 commit／push」描述的是原比較結案時點，不是本次發布狀態。
