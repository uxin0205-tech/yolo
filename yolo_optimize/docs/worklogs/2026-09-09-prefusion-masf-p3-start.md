# 融合前 MASF 增準：P3 Detect-only 最小位置實驗

## 目標與診斷結果

使用者再次明確指出 MASF 是為提高精度而設計。本階段不以刪除 MASF 為終點，而是驗證它在合理位置與訓練範圍下能否增準。

`b100-masf-off-probe-v1` 已正常完成：只改記憶體中的 alpha，來源 SHA 未變，完整 COCO val5000、0 次更新。

| 指標 | B100 原始 | 暫時 alpha=0 | 差值 |
| --- | ---: | ---: | ---: |
| overall | 0.503589001 | 0.503705170 | +0.000116168 |
| person | 0.624111237 | 0.624053821 | −0.000057415 |
| sports ball | 0.516234472 | 0.518438402 | +0.002203930 |
| baseball bat | 0.469482418 | 0.472911863 | +0.003429445 |

原 alpha=0.103149414。固定 CPU160 探針的 P3／P4／P5 特徵相對 L2 變化約 17.44%／16.86%／6.26%，證實共享位置會傳播變化；這是單一合成輸入的路徑驗證，不是完整資料分布統計。AP 未呈現大的主要指標依賴，但也不能據此斷言 MASF 毫無價值。瞬間關閉不等於重訓後效果。

## 為何不額外先跑 B100 關閉 bridge

我們已完成 A0 分段恢復，已有 **無 MASF 的 late E8 EMA 探索 parent**，且完成完整 COCO 與獨立推論匯出驗證。正式位置實驗直接在這個 parent 加零殘差分支，因此沒有先移除已使用模組造成的分布跳變，不需要為了換位再對 B100 額外訓練 5 epochs。這是依現有證據縮減原計畫的 bridge，不是把 B100 原成果覆寫。

## 三組與架構

三組共用 parent SHA256 `27c09c6685a003e9ff106898fd898199e184ee195cce643dd668079abadb9342`，相同完整資料順序、P3 head 學習率與 scheduler；不加 HOG 或 RepConv。

```text
control：
layer16：p3_raw ─┬─> layer17 → P4 → layer20 → P5
                └─> Detect([p3_raw, P4, P5])

shared（原位置的同起點對照）：
layer16：p3_raw → MASF → p3_shared ─┬─> layer17 → P4 → layer20 → P5
                                   └─> Detect([p3_shared, P4, P5])

fork（預計增準位置）：
layer16：p3_raw ─┬─> layer17 → P4 → layer20 → P5
                └─> MASF → p3_det ─┐
layer19：p4_raw ───────────────────┤
layer22：p5_raw ───────────────────┤
                   Detect([p3_det, p4_raw, p5_raw])
```

MASF 仍是原 DW3／DW5 + 1×1 project，公式 `x + alpha * context(x)`。兩個 MASF 組只重用 B100 訓練過的 context 作初始化，包括 BN 統計；alpha 重設為 0，原模型其他權重保持 late E8。這不是直接把 B100 使用中的 MASF 搬位。兩組 context 初值必須逐 tensor 相同。

fork 實作為繼承原生 Detect 的入口轉換，不修改輸入 list，不使用永久 forward hook 或跨 forward 快取。保留 one-to-one detach，因此 MASF 接收原生 one-to-many 的特徵梯度，不能宣稱 one-to-one 也直接反傳到 MASF。P3 的 one-to-many／one-to-one 預測分支均允許更新。

## 訓練設定與硬體界線

- 各 5 epochs，完整 COCO80 train118287／val5000，imgsz640；實體 batch32 × 累積4 = logical128，AMP FP16。
- fresh AdamW，betas=(0.948,0.999)、eps=1e-8、weights decay=0.00027、clip10；warmup1，10-epoch cosine／native loss horizon，三組使用相同前 5 epochs。
- 只更新 P3 head，LR=2e-6；MASF context LR=1e-5、alpha LR=1e-4。其他 Backbone、Neck、attention、P4／P5 head 參數與全部 BN running statistics 固定。
- alpha 初始 0；訓練後限制在 [-0.25,0.25]。零 gate 的第一步 context 梯度應為 0、alpha 梯度必須非零。此後 gate 開啟才讓 context 接收任務梯度。
- EMA 從相同 parent、age7400 開始，只以原生 decay 更新 active parameters，固定 state 在 live／EMA 中都逐 tensor 驗證。
- 每 epoch 分別驗證 EMA／live、保存完整 snapshot。overall／person 跌超過 parent 的 0.005 暫停分析；ball／bat 只觀察。相對 parent 與同預算 control 的主要保護項不退超過 0.001，且至少一項提升 0.001 才暫時接受；單次 seed 不構成統計顯著證明。
- alpha 是訓練後固定常數，不使用每張圖動態 scale／selector。沒有聲稱新增硬體實測加速；MASF 仍有原 context 運算成本。

## 驗證流程與紀錄

先做 AST 與實際整圖前置：零 gate CPU160 全輸出精確一致、原參數全保持、fork 在 alpha=0.1 時 raw P3／P4／P5 與 P4／P5 的 one-to-many／one-to-one logits 精確不变；shared 應能量測到跨尺度變化。兩個 MASF 組初始完整 COCO 的四項 AP 需與 parent 差值不超過 1e-8。

再做各組 128 張真實資料、1 次更新 smoke，確認相同首批 trace／loss、梯度存在、固定 state 不變，以及 Bit-True 推論匯出 CPU160 重載精確一致。全部通過才串行正式三組；每個 smoke 與正式 run 都是獨立新輸出，不拿 smoke 當正式 resume。

實際前置已完成：兩個 MASF 組的完整 COCO 四項 AP 相對 parent 差值皆為 0，零 gate CPU160 全輸出精確一致，fork 的非零 gate 路徑隔離驗證通過。三個 smoke 的首批 trace 同為 `251651aca5ce60c5cd20e34794613cf9983e2298410558d419088a297d4db5bb`，native loss 同為 212.19842529296875。

shared／fork 第一批 gate 梯度範數分別為 0.0286865234／0.4169921875，更新後 alpha 約 −1.0097e-5；零 gate 下 context 第一批梯度為 0 符合鏈式法則，不能誤判為梯度斷路。兩者均可往正或負殘差方向學習，不能單憑一批 gate 梯度大小推論最終 AP。固定 state、128 張／1 次更新、EMA age7401 與三組 Bit-True 匯出 CPU160 重載均通過。

正式 control 於 UTC 2026-09-09 13:44:09 啟動，session 70915；queue 後續自動接 shared／fork 各 5 epochs。訓練 AP 尚待完成，不將前置通過當成增準成功。原始權重與已完成結果未修改。

## 困難與未解事項

### 使用者追加訓練自主權

使用者明確允許調整各部分訓練方式與增加 epoch 數。此授權仍屬融合前第一輪與 MASF 增準目標，不改資料版本、不恢復融合後或 person-only 範圍。

目前先保留正在執行的三組相同 5-epoch 比較；取得趨勢後，可對有改善證據的候選與必要對照延長共同預算，或調整學習率、optimizer、Backbone／Neck／head／MASF 訓練範圍。不能將固定／凍結範圍視為永久限制，也不因「允許加訓」就重跑已正常完成的實驗。若續訓，先驗證 checkpoint 中 optimizer、EMA、criterion 年齡與資料順序契約，再啟動；目前尚未實作此新 MASF 流程的延長 resume，不宣稱可直接無縫延長。

本次僅記錄授權與條件式調整原則，沒有修改進行中的 shared job 或既有結果；未做額外 GPU／log 檢查。困難：無。

### Shared 完成事件補記

UTC 2026-09-09 15:06:36，shared 正常完成五回合，每回合完整 118287 張／925 次更新，約 520–523 秒。E1 overall／person=0.508242593／0.627397040；E5=0.507614304／0.627747652。E5 相對同回合 control 的 overall −0.000359456、person +0.000059398，沒有通過增準門檻。EMA alpha 從 E1 的 0.00841165 增加到 E5 的 0.02021287，故不能把這組結果歸因為 gate 一直沒有更新；但 gate 有更新不代表 AP 必須增加。

目前不直接延長 shared 配方，等待 fork 同预算結果後判斷位置與訓練範圍。已收取完成摘要，queue 同時自動啟動 fork；沒有重跑或修改 shared checkpoint，也沒有讀正在訓練的 fork log。困難：無。

### Control 完成事件補記

UTC 2026-09-09 14:23:05，control 正常完成五回合，全部為 118287 張／925 次更新，每回合約 465–467 秒（含 EMA／live 完整驗證）。E1 overall／person=0.508156300／0.627462316，E5=0.507973760／0.627688255，均維持 parent 附近，未達增準門檻。已檢查完成 summary；沒有重跑。Queue 已自動開始 shared，fork 尚待接續；不能提前下 MASF 增準結論。此階段執行困難：無。

困難：原 head 有 one-to-one detach，不能假設僅加分支就有所有 loss 的直接梯度；本版保持原生語義並訓練 P3 的兩套 head。其他困難：無。移位是否改善仍是待驗證假設；若三組未增準，需先檢查配方與 gate／context 學習情況，不以刪除 MASF 宣稱達成增準，也不盲目疊加新模組。
