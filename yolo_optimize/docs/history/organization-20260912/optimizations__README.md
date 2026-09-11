# 歷史入口快照（非目前狀態）

原路徑：`optimizations/README.md`。原始位元組另存同名 `.txt`；整理日期 2026-09-12。

# 優化方向索引

本目錄把每個可驗證的優化假設放在獨立資料夾，避免「位置、模組公式、訓練策略、量化」一次混改後，
無法判斷精度變化來自哪一個因素。

## 本輪優先入口

本輪接續 `final/full35`，已選原 J3 `best_joint` EMA 為 `PSEL` 並完成重驗。原生 E1–E5、LR×0.25 E1–E2 及 BN-only 評分已完成；沒有新配方通過精度安全線，原 BEST 保持不變，目前 GPU 已停止。詳見結果與推導（本機／歷史參照：`../../../optimizations/integrated-roadmap/native5-results.md`；未隨本次報告發布）。HOG 尚未正式訓練，未來從原 PSEL 獨立比較，不接退化 checkpoint；目前 `training_ready=false` 不代表從未執行實驗。
先盤點與 reproduce、視覺重播及同權重推論設定核對，必要時做原生 loss 修復／小範圍 AdamW 適應；有效 baseline 可以不完美，但載入、evaluator、lineage 必須可靠，基準可接受且八項指標相對 control 與 P0 非劣後，才按證據做單因子方向1消融。通過條件為 AP／joint 提升或可重現主要視覺錯誤改善，不強制所有視覺改善都達 AP +0.001。重建 FP parent→J0 僅為歷史／另案路徑，不是所有方法的前置。完整恢復計畫見訓練模型恢復計畫（本機／歷史參照：`../../../optimizations/trained-model-recovery/plan.md`；未隨本次報告發布）；本輪唯一整合順序見方向1 master plan（本機／歷史參照：`../../../optimizations/integrated-roadmap/direction1-master-plan.md`；未隨本次報告發布），optimizer 規則見optimizer policy（本機／歷史參照：`../../../optimizations/integrated-roadmap/optimizer-policy.md`；未隨本次報告發布）。
正式 baseline 維持 AdamW guard；MuSGD 雖已有 builder 支援，仍是另案 paired challenger，不強制、不以大 batch 觸發，尚無 AP 證據；中途切換的 optimizer state／LR 校準尚待驗證。「BN 較大」只指模型／參數改動較大，不是 batch 或 BatchNorm。HOG、MASF 移位、Q/K、optimizer 不同時首次開。

## 資料夾契約

每個方向至少包含：

- `README.md`：問題、證據、原本／建議架構、範圍與不做事項。
- `plan.md`：實作步驟、對照組、驗證 gate、停止條件與交付物。
- `architecture-report.md`（需要時）：可在 terminal 閱讀的目前／修改後資料流與推導。

實驗開始後可加 `results.md`，但原始 metrics、checkpoint 與大型 artifacts 應留在所屬實驗目錄；
此處只保存可追溯的決策摘要與連結。方向狀態依序使用：
`proposed` → `ready` → `running` → `validated`／`rejected` → `archived`。

## 整合入口

本輪三條主線是架構（MASF／RepConv）、訓練（HOG／條件式投影）與 BinaryQK。本輪唯一當前整合順序只以方向1 master plan（本機／歷史參照：`../../../optimizations/integrated-roadmap/direction1-master-plan.md`；未隨本次報告發布）為準，optimizer 配套見optimizer policy（本機／歷史參照：`../../../optimizations/integrated-roadmap/optimizer-policy.md`；未隨本次報告發布）；整合總計畫（本機／歷史參照：`../../../optimizations/integrated-roadmap/README.md`；未隨本次報告發布）及詳細實驗順序（本機／歷史參照：`../../../optimizations/integrated-roadmap/plan.md`；未隨本次報告發布）保留歷史／另案背景。

## 方向清單

第二輪獨立入口：固定成本下的創新候選（本機／歷史參照：`../../../optimizations/round2-innovation/README.md`；未隨本次報告發布）、最小實驗計畫（本機／歷史參照：`../../../optimizations/round2-innovation/plan.md`；未隨本次報告發布）與方向1 R2-REGION 完整規格（本機／歷史參照：`../../../optimizations/round2-innovation/region-ranking-full-spec.md`；未隨本次報告發布）。第二輪 KD 只有在有 teacher 與有效梯度時才啟動；必須先完成方向1 master 的可靠 baseline，不混入第一部分名詞。僅保留任務保護的固定basis、固定pair預算的區域ranking KD兩個假說，狀態proposed；不插入方向1主流程。

| ID | 類別 | 方向 | 狀態 | 全域位置 | 下一個決策點 |
|---|---|---|---|---|---|
| `OPT-COCO-PERSON-SPECIALIZED-HEAD` | 歷史提案 | COCO person-only 專用 head（本機／歷史參照：`../../../optimizations/coco-person-specialized-head/README.md`；未隨本次報告發布） | `deferred_by_user` | 不列本輪先決 | 使用者重新授權後才評估；不建立 View、不改 head |
| `OPT-P3-HOG-COMPANION-TRAINING` | T/HOG | P3 HOG 訓練期監督（本機／歷史參照：`../../../optimizations/p3-hog-companion-training/README.md`；未隨本次報告發布） | `proposed` | PSEL基準可靠後另案 `W-HOG10` | 與歷史 `F2-PRE-HOG9`／J0 主訓練不混用 |
| `OPT-TRAIN-CONFLICT-SAFE` | G（可選） | Detect優先的衝突安全訓練（本機／歷史參照：`../../../optimizations/training-conflict-safe/README.md`；未隨本次報告發布） | `proposed` | S3：獨立 trigger | 同一 stage 負 cosine≥20% 且 correction median≥0.02 才開 |
| `R0/R1` | R（可選；無獨立資料夾） | [RepConv 局部 recovery 舊研究](<../../research/2026-08-31-repconv-binaryqk.md>) | `proposed` | S4：條件式 R0/R1 | 由總計畫 S4（本機／歷史參照：`../../../optimizations/integrated-roadmap/plan.md`；未隨本次報告發布）決定是否追加；不另建方向 |
| `OPT-P3-MASF-DETECT-ENTRY` | MASF | P3 MASF 改到 Detect entry（本機／歷史參照：`../../../optimizations/p3-masf-detect-entry/README.md`；未隨本次報告發布） | `proposed` | 先 no-MASF recovery bridge，再評估 S5 | 由既有模型取得可接受 bridge 後才比較 MASF；不能直接搬模組 |
| `OPT-BINARYQK-ACCURACY-RECOVERY` | Q0 | BinaryQK 精度恢復（本機／歷史參照：`../../../optimizations/binaryqk-accuracy-recovery/README.md`；未隨本次報告發布） | `proposed` | S6/S7：site screens／matched QAT | baseline config 禁止 `STE=true`；先留診斷／另案 challenger，不說改 YAML 可跑 |
| `OPT-BINARYQK-SCALE-CODEBOOK` | Q1 | BinaryQK 少量 scale／codebook（本機／歷史參照：`../../../optimizations/binaryqk-scale-codebook/README.md`；未隨本次報告發布） | `proposed` | S7 後段：條件式 magnitude recovery | Q0仍有 magnitude 殘差才做 C0／B4／A8 replay |

`person-only` 已由使用者於 2026-09-08 暫緩，保留文件供歷史追溯，不是全域先決。RepConv 目前沒有子資料夾，
只連既有研究與總計畫 S4；不要自行建立新方向。條件式方向沒有觸發時直接跳過，不是為了跑滿表格而強制啟動。

## 全域執行順序

以下先指向本輪唯一當前的方向1 master plan（本機／歷史參照：`../../../optimizations/integrated-roadmap/direction1-master-plan.md`；未隨本次報告發布）；optimizer 細則見optimizer policy（本機／歷史參照：`../../../optimizations/integrated-roadmap/optimizer-policy.md`；未隨本次報告發布）。原有 P-SRC→P-J0 與 J0/HOG 主訓練順序及 S0–S8 保留為歷史／另案，不是所有方法的前置。

~~~text
`final/full35` 接續來源
        → 同口徑比較 J3 `best_joint`／J3 `best_pose`／J2 `best_joint`
        → PSEL 已選：原 J3 `best_joint` EMA（已重驗）
        → 必要原生 loss 修復／小範圍 AdamW 適應
        → baseline 可接受（載入／evaluator／lineage 可靠）
        → 單因子方向1消融（W-HOG10 raw P3／MASF bridge／QK；RepConv等選配）
        → fuse／calibration／export
~~~

只要上游改了 dataset task view、head、shared training或 FP graph，下游既有 HOG、calibration、QAT、PTQ與target profile
都不再是 same-lineage最終證據，必須在新 winner上重做。

## 共通實驗規則

1. 一個資料夾只回答一個主要因果問題；額外想法建立新方向或下一階段，不在首輪混入。
2. 對照組與候選組固定 dataset、split、seed、augmentation、optimizer、epoch 與其他架構開關。
3. 新 BBAT5 detection、pose 或融合實驗只能使用不可變的
   `/home/uxin/yolo/original/pose/derived/bbat5-v1/`；正式設定分別是 `configs/detect.yaml` 與
   `configs/pose.yaml`，不得重切 split 或改動影像／標註。
4. 結果必須同時保存 overall、尺度別、關鍵類別、成本與 paired-seed 資訊；沒有過預先登錄 gate
   就不能宣稱成功。
5. 每個方向都要有明確停止條件；失敗時保留證據，不用再疊更多模組掩蓋原因。
6. 先用 standalone／zero-train gate排除任務與graft錯誤，再把winner帶進shared model；不能把兩層因果混成一輪。

返回[子專案 README](<../../../README.md>)。
