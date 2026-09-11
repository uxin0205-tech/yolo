# MASF 論文適用條件與 BBAT5 小框核對

## 變更、原因與方法

使用者詢問為何論文有改善、本地沒有，以及 ball／bat 小物件效果。使用 research 技能查核 FPN 原論文與 COCO 官方 evaluator，主代理單獨完成。核對本地 P2 訓練範圍、bridge 校準與七組 BBAT5 指標，新增研究報告 `docs/research/2026-09-10-masf-paper-small-objects.md`。

## CPU 驗證結果

完整 BBAT5 validation 683 張、932 框計數斷言通過。ball 393 框，其中原圖面積 <32² 有 291（74.05%）；bat 539 框，其中 72（13.36%）。等比例最長邊 640 後短邊 <8 px：ball 12、bat 0。只讀影像尺寸／原 labels，不做新的資料切分或修改，不執行 GPU。

已有 BBAT5 box AP：bridge 相對無 MASF E8 control，ball +0.0105／bat +0.0075 個百分點；P2 相對 E5 control，ball -0.0957／bat +0.0606。不能稱小物件穩定增準，亦不能由現有局部微調宣布 MASF 永久無效。

## 困難、解法與限制

搜尋到的 EFPN 預印本已撤回，因此不使用其效果作方法證據；不藉此否定其他论文。使用者未指定本次要比較的具體論文，因此不假定模組與訓練條件相同。正式 APsmall 尚未計算，尺寸分布不是 AP。先前 COCO train-prefix HOG 的小球覆盖缺口不當成 BBAT5 全集統計。

其餘困難：無。研究報告區分實作已知、原因假設、待驗證項目；原結果、權重及停止狀態保留。沒有 GPU、新訓練、資料變更、刪除、commit 或 push。
