# 從 yolo_combine/final 接續：現況、待修復清單與順序

日期：2026-09-08。最新使用者決策：**以 `yolo_combine/final` 接續，但不預設其中所有部分已 train 好。先整理準備，不啟動 GPU。**

本頁優先於前版「成熟權重小幅 recovery」假設。來源已確定，不再要求使用者重新指定整個專案；實際視覺案例仍待補，但不阻擋本次整理。狀態：文件準備完成，`training_ready=false`，没有修好模型或取得新精度的宣稱。

> 後續方向1唯一當前整合順序見[方向1 master plan](<../integrated-roadmap/direction1-master-plan.md>)，optimizer 配套見[optimizer policy](<../integrated-roadmap/optimizer-policy.md>)。`B0` 只保留原 J3 `best_joint` 歷史分數參照；先以同口徑比較 J3 `best_joint`／J3 `best_pose`／J2 `best_joint` 後選 `PSEL`，尚未選定，`best_detect` 因 Pose 未適應不直接作 task=both。正式 baseline 維持 AdamW guard；MuSGD 有 builder 支援但只作另案 paired challenger，中途切換的 state／LR 校準尚待驗證。「BN 較大」只指模型／參數改動較大，不是 batch 或 BatchNorm。GPU 0。

## 1. 接續來源固定在哪裡

以下 `F` 代表 `/home/uxin/yolo/yolo_combine/final/full35`。

| 角色 | 固定來源 | 使用方式 |
| --- | --- | --- |
| 主交付包 | `F` | 唯讀接續來源，不原地修改或覆寫 |
| 程式基準 | `F/code/project/` | `F/run.py` 實際載入這份快照；不能默默換成外面的開發中 `src/` |
| 必要依賴／來源 | `F/source_bundle/` | 使用包內 Full35 子集；不是全部歷史候選的完整研究 bundle |
| 接續比較候選之一 | `F/weights/combined/inference/best_joint.pt` | `B0` 歷史分數參照；須與 J3 best_pose／J2 best_joint 同口徑比較後選 `PSEL`，不直接升格為品質已合格的優化 parent |
| 訓練狀態候選 | `F/weights/combined/full-resume/{best_joint,last}.pt` | best 與 stage-complete last 分開；內容／loader／optimizer 狀態尚未重新載入驗證 |
| 比較／回退 | `F/weights/rollback/j2/`、`F/weights/standalone/` | 對照 shared 取捨；不可把獨立 Pose/Detect 的兩個 trunk 當同一個 shared checkpoint |
| 歷史完整 J2 包 | `final/full35-j2-archive/` | 保留追溯，不混入 J3 主來源，不清理 |

Python 3.12、PyTorch 2.11.0+cu128、Ultralytics 8.4.90 是發布 README 的環境契約；本次沒有安裝、升級或 import 它們。機器可讀接續參考見 [continuation-manifest.json](<continuation-manifest.json>)，**不是可直接交給 trainer 的設定檔**。

## 2. 先把三種「完成」分開

- **交付內容可查**：本次查到 manifest 407 筆，路徑／檔案大小全相符；6 個指定小型文字入口的 SHA256 與 manifest 相符。沒有讀 `.pt` 內容，沒有跑全包 SHA256 `verify.py`，不是完整權重完整性驗證。
- **原實驗跑到終點**：報告記錄 J0/J1 跑滿，J2/J3 正常 early stop。少於 epoch 上限不自動等於中斷或欠訓練。
- **達到現在需要的品質**：仍未確認。舊 acceptance 是共享化後相對 standalone 的每項下降不超過 0.08；不是「所有項目都準、畫面都好」。

因此不把整包叫作「沒訓練過」，也不因為叫 final 就認定全部 train 好。能確認的缺口與待查事項如下。

## 3. 各項狀態與處理清單

| 項目 | 現有證據／狀態 | 接下來怎麼處理 | 現在做了什麼 |
| --- | --- | --- | --- |
| Full35 shared Detect＋Pose | 正式 J0→J3 已跑，seed0、有 Float/Bit-True validation | 作為接續候選；先重現，再決定原生 loss 補訓範圍 | 已固定來源、查報告／日誌；未重驗模型 |
| `best_joint` | J3 global epoch58，joint score 0.711174738975；八項通過舊 gate | 不以 selector 名稱保證品質；和 J2、必要 standalone 同設定比較 | 已列作候選，未更換預設權重 |
| `best_detect` | 報告明確記 global epoch0 的 Pose 尚未適應，八項 gate 未通過 | **不作 task=both 的直接交付／共同 parent**；只保留 Detect 診斷價值 | 已在新入口排除直接採用 |
| `best_pose`／`last` | 有既有 validation；不是全面勝過 best_joint | 必要時同設定重驗，不只挑單一指標最高的檔案 | 保留候選，不自動升格 |
| joint 的 ball box／ball pose | J3 相對 J2 分別 -0.003429／-0.000890 | 列入修復時保護與觀察的指標，不被 joint 平均改善掩蓋 | 已列出退化；不是新實驗數字 |
| shared MASF | 已在現有圖內接受過訓練；不能因此認定位置最優或訓練充分 | 先維持原圖做基準修復；之後才做依賴診斷及必要 no-MASF bridge | 未搬動、未重訓；不能寫成「MASF 沒訓練過」 |
| Q/K binary sign decision | 既有硬體契約凍結 Q/K，baseline `qk_ste=false` | 明確列為沒有在該 joint baseline 中進行 sign-decision 可學更新的部分；若要訓練需獨立 challenger | 已確認限制；不表示整個 attention 都沒有訓練 |
| HOG 輔助／Detect-only MASF／RepConv 新方案 | 屬本專案 proposed 的新處理，沒有本輪新訓練結果 | 基準可接受後，依證據一次只開一個對照 | 整理規格，未實作或啟動 |
| Partial75 | `final/README.md` 明確說此融合流程未執行；來源子集排除它 | 不把其他專案的 Partial75 結果當成這個 final 的已訓練模型 | 保持另案，不預設取代 Full35 |
| seed1／MuSGD／硬體部署 | 報告明確尚未做第二 seed、MuSGD challenger、FPGA/HLS/latency/energy 驗收 | 先完成基準與一項有效優化，再考慮重現／部署 | 保持未執行，不宣稱穩定性或加速 |

數據来源：[發布 README](<../../../yolo_combine/final/full35/README.md>)、[原始最終分析](<../../../yolo_combine/final/full35/analysis/FINAL_ANALYSIS.md>)、[gate CSV](<../../../yolo_combine/final/full35/metrics/gate-deltas.csv>)、[stage policy](<../../../yolo_combine/final/full35/code/project/src/yolo_combine/stage_policy.py>)。上述回歸不是目前視覺問題的已驗證根因。

### 原訓練階段，不要誤判成中斷

| Stage | 報告所載實跑／上限 | 判讀 |
| --- | --- | --- |
| J0 | 8/8 | Pose-head 適應階段已跑；不能證明所有 selector 的 Pose 都合格 |
| J1 | 20/20 | 共享 Neck＋heads 訓練已跑 |
| J2 | 25/80 | 報告記 patience17 正常停止；best global35，last52 |
| J3 | 11/20 | 報告及 `events.jsonl` 尾端都支持 patience5 正常停止；best58，last63 |

本次檢查到的 `events.jsonl` 非 macro 事件只覆盖 J3 epoch53–63；J0–J2 的細節來自發布報告，不冒稱本次已從逐事件記錄全程重建。

## 4. 先處理的設定問題

1. **不能直接把 YAML 當最後完整有效設定。** `configs/joint.yaml` 仍有 stages=J0/J1/J2、enable_j3=false、microbatch64；但 CLI 有 `--enable-j3`／`--j3-detect-microbatch`，發布紀錄記 J3 physical32×4。所以這不是 J3 沒跑的證據，而是接續前需要補齐 resolved config／命令覆寫紀錄。
2. **J3 歷史 warmup 是 3 epochs。** 發布分析如此記載，stage policy 的預設也為3；J0/J1/J2 才各覆寫成1。前版以通用 YAML warmup1 為參考，容易混淆，已在超參數頁更正。未來新 run 可以另訂 warmup，但必須標為新設計。
3. **區分 resume 與新補訓。** 已正常 stage-complete 的 `last` 不能因檔名可 resume 就當成尚有未完成訓練。要增加 epochs、改 LR／scope／loss 時開新 run並記 lineage；不能把重設 optimizer 的 warm-start 稱為 exact resume。
4. **final 內入口可能把輸出寫回 final。** 可攜 `joint.yaml` 的 run root 是 `../runtime`；新工作必须另設輸出與 cache 範圍、核對來源／資料的絕對解析。現在不原地修 snapshot、不啟動 CLI；CLI 即使 preflight 也會 import torch，不符合這次純文字盤點邊界。

## 5. 修正後的順序

```text
yolo_combine/final/full35（唯讀來源）
    │
    ├─ F0：檔案／來源／設定／訓練狀態盤點      ← 本次完成的範圍
    │
    └─ F1：相同設定重現現有模型與完整 validation（未執行）
              │
              ├─ 現有基準已可接受 ─────────────────────┐
              │                                        │
              └─ 基準尚不可接受 → B：原生 loss 補訓／修復│
                                      │                 │
                              重新驗證；失敗就回查       │
                                      └─ 通過 ─────────┤
                                                        ↓
                                            P-READY：可比較基準
                                                        ↓
                                 HOG／MASF／BinaryQK：一次一個方向
                                                        ↓
                                            部署版本複驗與回退
```

F0 中已完成的是下述有限文字／metadata 檢查；完整 checkpoint hash、resolved runtime config 仍是啟動 gate，沒有冒稱全部前置通過。**HOG 不再是修復基準前的第一必跑實驗。** 不預設一定從零重訓，也不預設只補5／10 epochs就足夠。

### B：原生訓練修復的條件式選擇

先保持原 Full35 graph、原 MASF 位置、原 QK固定係數、原 Detect/Pose loss，不加入新的 HOG、蒸餾、RepConv 或 scale。依 F1 的實際問題選一條，不把表中每條都排成必跑：

| 候選 | 何時考慮 | 初步更新範圍 | 預算判讀 |
| --- | --- | --- | --- |
| B-HEAD5 | 有可重現證據指出既有 head 適應不足、shared feature 可保持 | 相應 task head；Pose問題時只開Pose head | 5ep診斷窗口，不代表完成訓練；不預設改用不合格best_detect |
| B-NATIVE10 | 需要小範圍重新協調 joint 的 Neck／heads | Neck＋Detect/Pose heads | 第一個10ep窗口，0/5/10驗證；基準未恢復就不往新模組前進 |
| B-LATE10 | 前項／梯度證據顯示只更新 heads/Neck不足 | 另條件式開backbone layer9+，保留Q/K與固定係數限制 | 另案10ep窗口，不和HOG同時首次解凍 |
| 更早階段重建 | parent載入不相容、必要訓練狀態不可恢復，或小範圍修復有明確失敗證據 | 從final內可追溯standalone／早期來源重新規畫 | 明列新增成本與範圍，再決定；不是默認重跑全部J0→J3 |

保留 P0（修復前）作參照，B 的收益先稱原生補訓／修復收益，不歸功於新方法。只有 B 的產物被驗收為 P-READY，才讓新優化的 control/candidate 同時由 P-READY 開始。

歷史 gate `-0.08` 原樣保留供可比性；本輪另列修復判定，不能用「舊gate仍pass」代替修復成功。暫定保護八項相對P0不退超過0.001、事先指定的弱項／視覺症狀有改善；這是待案例確認的工程門檻，不是成熟模型的證明或統計顯著性。如果拿不到可驗證的視覺／訓練問題，維持未確認，不以無限增加epochs替代診斷。

## 6. 超參數現在怎麼看

完整表與新舊差異見 [hyperparameters.md](<hyperparameters.md>)。原先的 W-HOG10／MASF／QK 超參數保留為 **P-READY 之後的條件式提案**，不是目前已核准的訓練 queue。

基準修復的首版參考值：imgsz640、AdamW、Detect logical128（32×4）／Pose16、task weights1/0.25、weight decay2.7e-4、clip10；B-NATIVE10 的 Neck1e-5、heads2.5e-5、warmup3epochs。B-HEAD5 的目標head5e-5／warmup1；B-LATE10 新開late-backbone1e-6、其餘沿B-NATIVE10。每項都是可根據已重現問題修訂的候選，不說最佳或已充分訓練。

若恢復的是完全相同中斷 run，按其實際 full-resume 狀態而非上述 fresh optimizer提案；目前 J3 的既有停止事件不符合「已知中斷」條件。

## 7. 本次已整理與尚未執行

已完成：來源／權重角色清單、有限manifest稽核、stage證據分級、已知未適應候選排除、設定差異與warmup更正、基準修復優先順序、停用的接續manifest、入口及中文工作紀錄同步。

尚未執行：載入checkpoint、重建模型、合成tensor測試、模型validation、訓練、任何GPU工作、全包hash、硬體校準／測時、原始final修改、資料修改、下載、commit/push或刪除。實作與訓練目前皆未交付成可執行命令，避免把整理完成誤解成可以直接啟動。

資料始終保留 COCO80 及 canonical bbat5-v1（registry `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`），不新增 split、不抽樣、不改影像／labels；person-only暫緩。所有archive、權重、logs與來源都保留；本次清理盤點沒有提出可刪除目標，不需要清理授權。
