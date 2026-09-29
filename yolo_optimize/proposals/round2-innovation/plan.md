# 第二輪最小實驗計畫

日期：2026-09-08。狀態：`proposed`。研究已提出，任何模型實驗均未啟動。兩個候選獨立篩選，不預設組合。

## 0. 啟動前契約

1. 第一輪先保留未fuse的winner、對應FP teacher與完整manifest；第二輪不能直接修改唯一deploy checkpoint。
2. `R2-P`為第一輪選中的未fuse、fixed-PoT BinaryQK模型；`R2-T`為第一輪同架構的FP-QK control teacher。兩者精確hash／source history尚未建立，欠缺時不標ready。
3. 凍結COCO80／BBAT5 Pose任務、evaluator、八項指標、site policy、MASF／RepConv狀態與量化邊界；person-only不介入。
4. 先驗證R2-P的Q/K可訓練scope、STE、optimizer membership、BN及固定係數重校準規則。延用第一輪已通過的配方，不借用舊standalone LR。若第一輪仍未通過，先解決前置，不直接開KD。
5. COCO80入口 `/home/uxin/yolo/coco2017.yaml`；Pose入口 `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml`；registry `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。不重切、不增減影像、不更改標註；正式BBAT5沒有test split。
6. train-only cache／calibration只能沿用已核准manifest及assignment。若沒有合法既有View，記錄缺口；不自行抽另一套BBAT5校準資料。token-pair選取是loss內運算，不改資料成員。

~~~text
第一輪鎖定 R2-P／R2-T
   ↓
讀既有AP與誤差報告：ranking仍有殘差？
   ├─ 否 → 不開第二輪BinaryQK新方案
   └─ 是 → 固定投影可重fold？
            ├─ 是 → 優先R2-BASIS
            └─ 否 → 記錄硬體限制，考慮R2-REGION

任何新winner → 重新fuse／校準／完整部署驗證
~~~

## 1. R2-BASIS：先0訓練篩選

### B0：代數／介面前置，未執行

- 從實際Q/K projection解析head數、d與layout；檢查最終affine之後是否還有非線性／位置變換，確認D可吸收的位置。
- 純數值檢查FP dot parity、D=I原圖parity、sign(0)、整數飽和／rounding、Hadamard重複基底及固定scale格式。這是未來CPU測試，不是本次已有結果。
- 輸出op catalog必須顯示無新增dense rotation、動態selector或第三basis；不通過即停止。
- 本地hardware contract目前凍結Q/K與係數：在另立版本與明確核准重建前，本方向不得升為ready。不能為了讓D可fold而直接刪除assert或繞過契約。

### B1：三個0-training policies

| ID | 配置 | 目的 |
|---|---|---|
| B-CURRENT | 現行D=I；同程序重校準 | 基準 |
| B-RANDOM | 事前固定的一個非退化隨機D | 排除「任何旋轉都會有收益」 |
| B-TASK | 有限候選池中按worst-task ranking選D | 任務保護候選 |

候選池最多8個包含I與B-RANDOM；固定seed和同一兩任務cache，排除代數等價模式。首輪只處理第一輪診斷最敏感的一个binary site，其他site不變，不掃site×head×模式全組合。

全部D用相同校準量／規則估cI,cH，formal val不參與D或scale選擇。B-TASK未勝過current/random就停止；若選回I或與random相同，只報沒有額外任務選擇證據。

若已有同parent/evaluator的B-CURRENT metrics可重用，最多新增2次完整evaluation；否則3次。這些是未來模型evaluation，**不是GPU-free**。

### B2：只有有訊號才補創新對照

從同一候選池以平均ranking誤差選出B-MEAN；若與B-TASK為同一D，不能宣稱minimax目標有獨立貢獻。不同才補一次evaluation，核對較弱任務的保護是否改善。

若basis有ranking及AP正訊號但仍需訓練恢復，僅追加B-CURRENT-QAT／B-TASK-QAT兩臂，從同R2-P、同第一輪scope／LR trace／更新次數出發。只改固定basis，禁止同時加region KD。沒有訊號不加epochs尋找故事。

## 2. R2-REGION：先K0/K-UNIFORM，通過才K-REGION；有訊號補K-FG

詳細 hook／梯度／超參數以[R2-REGION 完整規格](<region-ranking-full-spec.md>)為準。

| ID | KD | 必須相同的項目 |
|---|---|---|
| K0 | 不開額外KD | R2-P、native losses、training trace、資料及scope |
| K-UNIFORM | 均勻query/pair ranking KD | R2-T、pair預算B、loss型式與係數校準規則 |
| K-REGION（條件式） | per-instance／有效點支持區域與背景分配 | 同上，不增加teacher或KD總pair數 |
| K-FG（条件式） | 普通foreground/background分配，沒有keypoint／instance配額 | K-REGION有訊號才補，區分現成mask與提案貢獻 |

均由同一R2-P開始，不從K-UNIFORM winner接續K-REGION。若第一輪KD控制的teacher／sampling／trace完全相同才可重用，僅loss名稱相同不足以重用。

實作前預先鎖定B、margin／τ、背景配額、empty-region回退、各task的GT→token映射，並保存配置；這些目前尚未定值，故非ready-to-run。只用train-only gradient ratio校準一次；K-UNIFORM／K-FG／K-REGION採相同校準規則，不能一邊有更大aux梯度卻稱公平。

同步測mu=0等價、raw feature／o2m/o2o梯度路徑、teacher eval與stop-gradient、augmentation對齊、valid-point mask、無GT影像native negatives不變、resume與AMP。任何一項未通過不進正式比較。

## 3. 共用通過／停止條件

- 沿第一輪八項及同evaluator，AP以0–1計；正式val只供事前定義的模型比較，不反覆微調取樣配額。
- 量化gap／ranking未改善就不保留。初篩要求COCO overall或joint score至少+0.001、八項各不低超過0.001；要聲稱任務保護，另需原弱項（由第一輪事前鎖定）至少+0.002。這些是工程gate，不是統計顯著性。
- B-TASK要勝B-RANDOM及B-MEAN；K-REGION要勝K-UNIFORM及K-FG。只胜無改動baseline不足以證明所提創新機制。
- 初篩只有seed0；候選通過後，對最關鍵control/candidate補paired seeds1/2。若差異不穩定，降為未證實，不以平均微小正值硬稱成功。
- 部署圖的binary coverage、basis數、位寬、完整packing／epilogue成本需一起報；不以理論MACs代替延遲。fixed-fold parity不過，先不採B方案。
- 原先已部署圖如需變更，保存舊版並重做S8；沒有target設備時成本保持待測。

## 4. 收斂與本次狀態

預設先挑一個方向，不建立兩套完整factorial。第一輪仍是主線；第二輪無實測收益，缺少cache／teacher／硬體授權契約時只保存提案。

使用者要求週額度剩73%即收尾。本端沒有帳戶額度工具，不能把token數換算週百分比；本次停止擴展文獻範圍，完成有限候選與文件後交付，未宣稱已讀到73%。若使用者回報到線，立即停止新研究／實驗安排。

返回[第二輪入口](<README.md>)與[推導報告](<../../docs/research/2026-09-08-round2-innovation-analysis.md>)。
