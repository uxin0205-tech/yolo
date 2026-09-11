# 2026-09-09：E10 收斂判斷與融合前 HOG 原生對照

## E10 結果與目前最佳

兩組正常完成完整 COCO 訓練／驗證。EMA internal AP50–95（0–1）如下：

| 權重 | overall | person |
| --- | ---: | ---: |
| narrow E8 | 0.507432701 | 0.626632484 |
| narrow E9 | 0.506760235 | 0.626776852 |
| narrow E10 | 0.507053398 | 0.626971527 |
| late E8 | 0.508153635 | 0.627520517 |
| late E9 | 0.506875100 | 0.627129130 |
| late E10 | 0.507314536 | 0.627198679 |

最後回合不是最佳；late E8 仍是這輪 overall／person 同時最好的 late 權重。其相對 A0 增益與完整匯出、推論證據見 E8 工作紀錄，但對同預算 narrow 最佳 E8 的增幅仍未達預設 +0.001 門檻，沒有正式 winner。不無限延長同一窄範圍或同一解凍 LR，也不以 B100→late 的總差異冒充純 Backbone 效果。

## 下一個必要方向：W-CTRL5／W-HOG10

依原方向 1 與使用者明確回合設定：原生 Detect loss 對照固定 5 epochs；HOG 最多 10 epochs、patience 4，warmup 1。從已驗證可載入且 AP 保留的 late E8 EMA 建立兩組同起點的新階段，**這是探索 parent，不代表 late E8 已正式勝出或替換 A0／B100**。若 HOG 提早停止，只比較共同回合；若 HOG 超過5回合，不將額外預算的改善宣稱純 HOG 因果。

兩組都重新建立相同 AdamW 群，Neck LR1e-5、head LR2.5e-5、HOG LR3e-4，betas=(0.948,0.999)、weight decay0.00027、clip10、10-epoch cosine／criterion horizon、warmup1。EMA 權重從 late E8 EMA 初始化並使用已知 age7400，不把新 optimizer 階段稱為無中斷 resume。physical32×4、全量 COCO80 train118287／val5000、AMP FP16 不變。

可訓練 Neck 是 layer13／16／17／19／20／22 的非 attention 參數，另加 layer23 Detect 與獨立 training-only HOG head。Backbone 與 attention 參數固定，非 head BN running statistics 固定；兩組保留相同 score surrogate 以便對可訓練的上游 Neck 傳梯度，不在這個比較再改 Q/K 訓練規則。head 與 Neck 範圍對照相同，與之前 narrow／late 不直接比較為純 HOG 效果。

## HOG 實作契約

直接重用 `src/yolo_optimize/hog.py`，不改來源模組：raw P3（layer16、256 channels）→1×1 Conv→9 bins，僅 2313 個輔助參數。目標從同一批增強後 RGB 產生，cell8、unsigned orientation、相鄰 bins 線性分配、cell內 L1 機率正規化；不是 block L2-Hys 描述子。

GT 使用全部80類：cell中心落在任意有效框內的 hard union mask，低於能量門檻的 cell 無效，剩餘有效 cell 均勻平均，每張後再 batch-sum。**不是較早論文構想中的 soft mask×energy weighting**；本次重用的是已有工程測試的版本，不暗稱所有舊公式已實作。

HOG E1 關閉，E2 漸開，E3–E8維持，E9–E10關閉；control 全程關閉但保有同初始化 dormant aux／optimizer 群。P3 擷取使用單次 forward 的區域 hook，finally 移除，不在 model 保存 feature／target，AMP retry 重新生成當次輸入目標。validation 及 export 明確刪除 `hog_aux`，不改3-channel RGB與原推論图。

## 已完成驗證

既有 HOG 12 項 CPU 測試全部通過：方向、週期bins、座標、空框、常數圖、有效cell normalization、batch-sum切批等價與 feature／GT 梯度契約。新訓練入口初次 AST 發現一處 `report.update` 錯字，已在 GPU 執行前修正並通過重驗。

實際 train-only 校準使用固定前256張增強影像，沒有 optimizer 更新、沒有用 val 調 μ。真實 callsite 的 μ=0 native loss，以及 layer16／Detect代表性參數 gradient 逐值一致。有效cell482776，253／256張有有效cell。

校準 μ=0.8750700409454109，使 raw P3 輔助／原生梯度全域比例為5%，各microbatch約0.02953–0.06897，中位數0.05005284；cosine約0.00003059。μ不是「佔總梯度87.5%」，也不能由接近正交的梯度宣稱精度必然改善。證據：`artifacts/prefusion-hog-calibration-v1/summary.json`。

新增 `train_hog.py`、`run_hog_pair.py` 與 HOG export 剝除入口。正式排程前還需兩臂真實 macro 更新與 strip驗證：control aux 不更新、HOG aux 和P3 producer梯度非零、固定state不漂移、資料trace相同，並以實際模型 CPU160 比較剝除前後輸出、state只少兩個aux keys。smoke 的強制開HOG僅用來驗證作用路徑，更新丟棄；正式E1仍關閉。

## 困難與解法

舊 HOG 文件混有融合後 joint gate、soft mask構想與 dormant J0 起點；本次明確採融合前COCO、目前已測試的 target定義與同初始化新階段，並保留差異說明。曾有語法錯字，已修正；未將錯字或過期文件規則帶入正式 GPU 訓練。圖片檢視的 bwrap 限制仍存在，未冒稱目視效果。

## 未解事項與風險

真實更新／strip smoke與正式 HOG AP 尚待完成。late E8 只是探索 parent；最後仍須對原始 A0／FP 與 matched control 共同核對。HOG若無收益停止該候選，不無限掃 μ／bins／位置；若有收益且要歸因方向先驗，才考慮同預算 LUMA 負控制。尚未執行第二輪、person-only 或融合後任務。

## 真實更新驗證與正式啟動補記

2026-09-09 15:55（Asia/Taipei）兩臂各128張真實macro均正常完成。queue實際檢查通過：同資料trace、原生與HOG臂P3 producer梯度非零、control的aux無梯度／無optimizer state、HOG aux有梯度與step1、固定參數與非head BN state不變、EMA7401。兩臂實際模型CPU160推論strip逐值一致，state只少 `hog_aux.projection.weight` 與 `hog_aux.projection.bias`。證據是 `artifacts/prefusion-hog-proof-v1.json`；沒有OOM或新增runtime困難。

15:55:35 啟動 `prefusion-hog-control-v1`，正常完成後queue接續 `prefusion-hog-hog-v1`；monitor session93000，每次shell wait最多600秒，正常不讀log或額外查GPU。正式AP尚待完成事件，沒有將校準通過當成精度改善。
