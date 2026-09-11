# YOLO26M 三類優化總計畫

更新：2026-09-08。已完成原生、低 LR 與 BN 對照；未取得可採用的新配方，GPU 已停止。下方 S0–S8 保留歷史／另案規劃，不代表本輪尚未做過 GPU 驗證。

> **本輪當前狀態（2026-09-08）：** `PSEL` 已選定為原 J3 `best_joint` EMA，並完成同口徑重驗。原生對照 E1–E5、LR×0.25 對照 E1–E2 及 BN-only 評分均已完成；精度安全線未過，沒有新 BEST，目前 GPU 已停止，`training_ready=false`。結果見[native5 報告](<native5-results.md>)，後續順序見[方向1 master plan](<direction1-master-plan.md>)，optimizer 規則見[optimizer policy](<optimizer-policy.md>)。HOG 尚未正式訓練；未來仍從原 PSEL 做獨立比較，不接退化 E2／E5，也不要求原生對照必須先超過原 BEST。
>
> 正式 baseline 維持 AdamW guard；MuSGD 雖已有 builder 支援，仍是另案 paired challenger，不強制、不以大 batch 觸發，尚無 AP 證據；中途切換的 optimizer state／LR 校準尚待驗證。「BN 較大」只指模型／參數改動較大，不是 batch 或 BatchNorm。HOG、MASF 移位、Q/K、optimizer 不同時首次開。
>
> 本頁原有 `P-SRC→P-J0`、J0/HOG 主訓練與舊 LR 配方保留為歷史／另案因果流程，不是本輪既有權重 recovery 的必要步驟。新 HOG 名稱為 `W-HOG10`，不與歷史 `F2-PRE-HOG9` 混用。MASF 移位須先由既有模型取得可接受的 no-MASF recovery bridge，不能直接搬模組。Q/K baseline config 明確禁止 `STE=true`，僅留診斷／另案 challenger，不說直接改 YAML 可跑；第二輪 KD 僅在有 teacher 與有效梯度時啟動。

本資料夾是跨方向的順序與比較契約入口；各方向仍保留原始研究。若舊方向的 parent、先後順序或 gate 與本頁／[詳細順序](<plan.md>)衝突，以本次整合說明為準，不回寫歷史實驗結論。

## 1. 三類包含什麼

| 類別 | 主要方法 | 它回答什麼 | 首轮選擇 |
|---|---|---|---|
| A 架構 | P3 Detect-entry MASF、局部 RepConv | 限制 MASF 影響範圍；增加訓練期參數化能力 | 本輪先 no-MASF recovery bridge，再條件式 MASF；RepConv 條件式 |
| T 訓練方法 | HOG companion、Detect 優先梯度投影 | 形狀監督是否改善 P3；任務梯度是否互相抵銷 | PSEL基準可靠後按 master 用 W-HOG10；投影另按梯度證據觸發 |
| Q BinaryQK 恢復 | site isolation、matched QAT、teacher KD、scale 分组 | 哪個 site 敏感；sign/ranking 資訊能否學回；幅度是否仍為瓶頸 | 本輪先 Q 診斷 contract；Q/K STE 僅另案 challenger，scale 最後才開 |

使用者於 2026-09-08 指示 person-only 暫緩：本輪保留 COCO80 Detect＋BBAT5 Pose，不建立 person View、不改 head。既有 person-only 文件保留為 deferred。RepConv 與 scale/codebook 不要求每個都跑；方法數不等於必要 job 數。

## 2. 本次修正的五個順序問題

1. **Float 不等於 FP-QK。** 現有 float-pwl-final.yaml 仍是 Hadamard basis、power_of_two scale；Float 是可微 surrogate backend。不能把 Full35 Float 權重改名 BASE-FP 後當「從未二值化」的 parent。
2. **HOG 不接在 J3 最終點後面當 J1。** 舊 J0/J1/J2/J3 HOG 主訓練規格是歷史／另案流程，包含原 post-MASF 與 `F2-PRE-HOG9` 設計，不能當本輪既有權重 recovery 的前置或混表。本輪 warm-start recovery 若經視覺與設定證據觸發 training-only HOG，名稱為 `W-HOG10`。
3. **不同 evaluator 不混表。** 本輪仍用原 COCO80＋BBAT5 八項指標；官方 COCO API 的 person AP 另欄保存，不能與 Ultralytics internal person AP 直接相減。person-only 已由使用者暫緩。
4. **RepConv 先做單點，雙點條件式。** 舊 R0/R1/R2/R3 全矩陣不作預設。R0/R1 首測 layer17；只有證據支持 layer20 才加 R2，兩者各有效才考慮 R3。
5. **QAT 的歷史 LR 不跨任務搬用。** W-DIR 的全模型 lr=5e-5、40 epochs 是單任務 COCO lineage。新 joint Detect＋Pose 使用匹配的 joint scope 與 role LR，兩臂同配方；不能把舊數字視為已驗證的 joint recipe。

第一點的直接來源：[正式 Float 變體](<../../../yolo_attention_final/final/configs/variants/float-pwl-final.yaml>)、[BinaryScore 實作](<../../../yolo_attention_final/final/yolo_attention/binary_basis.py>)。HOG 舊規格見[2026-09-04 報告](<../../docs/research/2026-09-04-paper31-to-yolo26m-training-adaptation.md>)。

## 3. 歷史／另案：從乾淨來源重建的順序（非本輪 recovery）

以下圖示說明從乾淨來源重建的歷史／另案流程；其 P-SRC→P-J0 與後續參數不適用本輪既有權重 recovery。

~~~text
S0 先固定任務、evaluator、parent 血緣；重用舊 predictions / metrics
 │
 ├─ 舊 Full35／V1-BR：只作診斷參考，不直接冒充乾淨 FP-QK parent
 │
 ▼
S1 保留 COCO80 Detect＋BBAT5 Pose；固定 no-MASF、FP-QK source 契約
 ▼
S2 由核對過的 source 建立 joint model，完成共同 J0
 │  graph：raw P3、原 Conv、FP-QK；aux role 已註冊但關閉
 ▼
S3 訓練方法：同一 J0 各自重跑 J1→J3
 ├─ F0-MATCH ───── 基礎配方
 ├─ F2-PRE-HOG9 ── raw P3 的 HOG；有訊號才補 F1-PRE-LUMA9
 └─ G1-APC ─────── 僅衝突 trigger 達標才開；不疊 HOG
 │  選一個訓練 winner；aux 保持 dormant，部署副本才移除
 ▼
S4 條件式 RepConv recovery：R0 vs R1(layer17)
 │  layer20 的 R2 / 雙點 R3 有證據才追加
 ▼
S5 主訓練後 MASF recovery：CTRL-P3 vs MASF-P3
 │  共用相同無 MASF parent；只作用 Detect P3；Pose / P4 / P5 保護
 ▼
S6 固定最終 FP-QK 權重、heads、MASF、RepConv 狀態
 │  建立同一 parent 的 site10 / site22 / both-binary screens
 ▼
S7 FP-CTRL vs BQK-CAND matched QAT
 ├─ ranking/KL 仍差且有可恢復訊號 → 一個 BQK-KD
 └─ magnitude 殘差仍明顯 → 條件式 B4；A8 僅動態 scale 上界
 ▼
S8 RepConv deployment fuse → 固定係數重校準 → parity → PTQ/export/profile
 │
 ▼
只對通過初篩的比較補 paired seeds；最終整條流程另作重現驗證
~~~

S0–S8 是未來的規劃順序，沒有建立可執行 queue。此次不執行 forward/backward、full validation、calibration 或裝置 profile。

## 4. 為何採這個順序（歷史／另案流程）

以下理由說明的是上段歷史／另案流程，不是本輪既有權重 recovery 的必要步驟。該流程的任務與 heads 保持目前定義。訓練方法會改 backbone／neck 權重，所以在 J1–J3 比較。RepConv recovery 若要做，先在 MASF 之外獨立驗證，兩者不能同一 arm 首次開啟。MASF 按既定需求留在基礎訓練後，僅用小範圍 recovery 檢驗是否值得加入。

最後才做 BinaryQK：若上游權重或圖改變，舊 Q/K 分布、site 敏感度與校準係數都不能直接當新圖的證據。RepConv 的部署融合與固定係數／INT8 最終校準安排在最後，因為真正要部署的是那張圖。

若 S0 無法找到或建立可驗證的 FP-QK source，先完成可做的歷史診斷；整合訓練維持 proposed。由 binary-trained weights 切回 FP 做 warm start 可以另案，但必須明記其來源與所需 recovery，不能當乾淨 FP 基準。這個 parent 建立成本不能藏在「只要兩個 jobs」裡。

## 5. 歷史／另案的主訓練／預期部署圖（非本輪 recovery）

以下圖示保留目前 Full35 與歷史／另案主訓練、預期部署架構，不是本輪既有權重 recovery 入口。

目前 Full35（歷史／另案）：

~~~text
layer16：C3k2 → MASF → p3_shared ─┬→ layer17 → P4 → layer20 → P5
                                 ├→ Detect80([p3_shared,P4,P5])
                                 └→ Pose2   ([p3_shared,P4,P5])
attention sites：目前已採 BinaryQK；Float/Bit-True 是 backend 差異
~~~

歷史／另案主訓練（非本輪 recovery）：

~~~text
layer16：p3_raw ─────────────┬→ layer17 Conv → P4 → layer20 Conv → P5
                             ├→ Detect80([p3_raw,P4,P5])
                             ├→ Pose2  ([p3_raw,P4,P5])
                             └→ temporary 1×1 head → HOG loss（選用時）
augmented RGB → HOG9 target ──────────────────────────┘
J3：HOG 權重已歸零；attention 為本輪固定的 FP-QK reference
~~~

若 MASF／RepConv／BinaryQK 各自通過，預期部署：

~~~text
layer16：p3_raw ─────────────┬→ layer17 Conv* → P4 → layer20 Conv* → P5
                             │
                             ├→ MASF → p3_det ───────────────┐
layer19：p4_raw ─────────────────────────────────────────────┤
layer22：p5_raw ─────────────────────────────────────────────┤
                                                             ↓
                                    Detect80([p3_det,p4_raw,p5_raw])

Pose2([p3_raw,p4_raw,p5_raw])：不經 Detect-only MASF
Conv*：若該處 RepConv 勝出，部署時已融合回原尺寸 3×3 Conv
attention：保留一個通過 gate 的 fixed-PoT site policy
HOG generator／aux head／teacher：均不在部署圖中
~~~

這是條件式目標图；若某候選未過，就採對照，不把未驗證方法全裝進最後模型。

## 6. 必要實驗數與成本

| 區塊 | 首次必要 training arms | 何時增加 |
|---|---:|---|
| person-only H1/H2 | 0 | 使用者暫緩，不列本輪 |
| 共同 J0 | 1 次共同前綴 | source／seed 改變才重建 |
| HOG recipe 初篩 F0/F2 | 2 個 J1→J3 continuations | F2 有訊號才補 F1：+1 |
| conflict G1 | 預設 0 | trigger 達標：+1，僅完全相同控制可重用 F0；否則 G0/G1：+2 |
| RepConv R0/R1 | 預設 0 | 決定研究此增準假說：+2；R2/R3 各條件式 +1 |
| MASF CTRL/MASF | 2 | 通過才補 paired seeds |
| BinaryQK FP-CTRL/CAND | 2 | KD +1；scale 分支另見計畫 |

在「F0/F2＋MASF＋QAT」各初篩都需要執行的情境，是 **6 個比較用 training arms，加共同 J0 與尚未取得的 FP source 建立成本**。補 HOG-specific 負控制後為 7 個比較 arms。這不是 6 個 epochs，也不是所有方向完整驗證的總成本；不同 arm 長度相差很大。

zero-train 完整 validation 另計，不代表 CPU-only。正式新 FP parent 的四種 site policy 如需全量評估，最多為 4 次 evaluation；若已有同 parent 的 FP metrics 可重用，新增 3 次。旧 V1-BR 两個 hybrid screens只代表舊 lineage。

## 7. 共用判定口徑

- 所有 AP 以 0–1 報告；+0.001 是 +0.1 個百分點，屬事先登錄的工程門檻，不是統計顯著性。
- 保留目前八項：COCO overall、COCO person、BBAT box、pose、ball/bat 各自 box與pose。新 run 使用同一 evaluator/backend/schema，其他診斷另欄。
- 沿用現行 joint score 定義與 selector；原最大容許掉點 0.08 只作舊部署底線，不是增準成功門檻。
- HOG／投影要求 joint score 至少 +0.001、八項各不低超過 0.001、person 或 ball pose 至少 +0.002。MASF／RepConv另需 COCO overall 至少 +0.001 且尺度／ball／bat無犧牲。
- canonical person AP 使用固定 COCO category_id=1、maxDets、crowd規則另報。它不可與 internal person AP直接求差。
- 候選一個 seed 過關只標 provisional；paired seeds 1/2 通過才升級證據。若改了 source/head/evaluator，不重用舊控制。
- BBAT5 沒有 test split；不得為本研究自行建立。沒有外部 holdout時如實標示 validation 選模與多重嘗試限制。

各方向更精確的停止／續跑規則、QAT gap 公式與 parent 表見[詳細實驗順序](<plan.md>)。

## 8. 還可以優化什麼

最值得排在新增大模組之前的是：用既有 predictions 分解漏檢／定位／背景誤報，核對 COCO crowd ignore 的 loss 語意，以及消除 fixed-scale 路徑白做的 dynamic reduction。這三件分別提高決策品質、可能修正不合理監督、減少多餘運算。

共享 BN running stats 目前已固定 eval，因此「每個 task 持續改 shared BN 導致漂移」沒有本地支持，不列首案。HOG 本身則仍有 target normalization／tiny-object occupancy／aux 梯度影響需要先釐清，不能先宣稱新增 loss 一定增準。

候選理由、來源、觸發條件與最小對照見[新增方向評估](<../../docs/research/2026-09-08-additional-optimization-priorities.md>)。

## 9. 分工與交付

主代理負責假說、實驗先後、parent與指標口徑、停止條件和是否升格。使用者指定的 gpt-5.6-luna / max 子代理負責確定規格後的文件摘錄、索引同步、格式／路徑檢查。遇到規格矛盾回報，不擅自啟動實驗或改 gate。

- [詳細實驗順序](<plan.md>)
- [來源盤點](<source-inventory.md>)：子代理機械摘錄，保留原文衝突，不能覆蓋本次決策。
- [新增方向評估](<../../docs/research/2026-09-08-additional-optimization-priorities.md>)
- [中文工作紀錄](<../../docs/worklogs/2026-09-08-integrated-optimization-roadmap.md>)
- [方向總索引](<../README.md>)

未移動或刪除舊報告、資料、權重；沒有 commit／push。方法與訓練狀態仍為 proposed，文件完成不等於 ready-to-run。
