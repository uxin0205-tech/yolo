# 現有 MASF 權重：COCO 與 BBAT5 Pose 資料集直接驗證

名稱、原架構／新架構圖、bridge 訓練原理與 640 輸入的運算量，見 [ARCHITECTURE.md](<ARCHITECTURE.md>)。

ball／bat 不能直接等同小物件：CPU 核對 BBAT5 val 原圖框面積 <32² 的比例為 ball 74.05%、bat 13.36%；這是尺寸分布，不是 APsmall。論文適用條件、實際 box 變化與未完成的尺寸分組驗證見[小物件研究](<../../docs/research/2026-09-10-masf-paper-small-objects.md>)。

> 使用者最新範圍更正：本次只比較 optimize 新方向重訓模型。第 1 節七組 checkpoint 皆符合；第 2 節「舊正式 Pose 的 MASF 開／關」僅保留歷史旁證，**不得用來判斷新方向 P2／P3 是否該保留 MASF**。本次另確認 BBAT5 Detect／Pose val 的 683 張影像、932 個 class／box 標註完全一致，故第 1 節 box AP 適用兩個入口的相同框。新 P2／P3 沒有 Pose head，不能提供其 keypoint AP。見[範圍更正紀錄](<../../docs/worklogs/2026-09-10-new-direction-bbat-scope.md>)。

## 結論與停止狀態

**可以直接使用現有權重，這次比較沒有新增訓練。** 七個已訓練 Detect checkpoint 已在完整 canonical BBAT5 Pose validation 683 張上驗證 box AP；原正式 Pose checkpoint 則比較 MASF 開／關的 box 與 keypoint AP。完成後停止，不接續 MASF 訓練、J1/J2、activation 或方向 2，等待使用者決定。

原正式 Pose 模型中的 P3 shared MASF 對 ball／bat **框定位有正向效果**，但 keypoint 並未全面改善。既有 COCO 訓練的 P2／P3 候選在 BBAT5 的變動很小、方向混合，尚無一致優勢。因此不能一概說 MASF 沒用，也不能說加入後所有精度都提高。

## 1. 已訓練 Detect 權重直接驗證

下表全部是 AP50–95，0–1 單位。COCO 欄位引用同一 checkpoint／EMA 的既有完整 5000 張驗證；BBAT5 欄位是本次新測的 683 張。沒有硬接新 Pose head，Detect 權重不能提供 keypoint AP。

| 已訓練權重 | COCO overall | COCO person | BBAT ball box | BBAT bat box |
| --- | ---: | ---: | ---: | ---: |
| P3 對照 E5 | 0.507974 | 0.627688 | 0.298412 | 0.560151 |
| P3 shared MASF E5 | 0.507614 | 0.627748 | 0.298586 | 0.558641 |
| P3 Detect-only MASF E5 | 0.507963 | 0.627627 | 0.297688 | 0.559877 |
| Head 對照 E8 | 0.508267 | 0.627699 | 0.298893 | 0.559707 |
| P3 bridge MASF E8 | 0.508212 | 0.627664 | 0.298998 | 0.559782 |
| P2 對照 E5 | 0.508420 | 0.627698 | 0.298705 | 0.561273 |
| P2 shared MASF E5 | 0.508271 | 0.627797 | 0.297748 | 0.561879 |

下表各候選與對應無 MASF control 相減，不拿 P2 E5 對 P3 E8 當位置收益。P3 bridge 對 Head control 是整體方案比較；若單獨研究 bridge，正確對照是同樣已有 P3 MASF 的 `masf-head-fork-v1`，見架構說明：

| 候選相對配對 control | Δ COCO overall | Δ COCO person | Δ ball box | Δ bat box |
| --- | ---: | ---: | ---: | ---: |
| P3 shared E5 | -0.000359 | +0.000059 | +0.000174 | -0.001510 |
| P3 Detect-only E5 | -0.000011 | -0.000061 | -0.000724 | -0.000274 |
| P3 bridge E8 | -0.000055 | -0.000035 | +0.000105 | +0.000075 |
| P2 shared E5 | -0.000149 | +0.000099 | -0.000957 | +0.000606 |

P2 不增加 Detect head；它會影響共享下游特徵。P3 shared、P3 Detect-only 與 bridge 是不同接線／訓練路徑，不是同一操作的重複名稱。上述微小變動未做多 seed 或統計顯著性驗證，不宣稱穩定增益。P3 bridge 的 box 微升也不足以抵銷它在 COCO overall／person 的微降。

## 2. 原正式 Pose checkpoint 的 MASF 推論消融

來源為 `p0-full35-p3-b32a4-e100max-seed0/weights/best.pt`，完整路徑與 SHA256 在原始 JSON。這裡 run 名稱中的 p3 是歷史訓練階段，**實際插入位置另由模型檢查確認為 `model.16.p3_masf`**，不可只憑名稱判定。

模型已用 Pose 訓練。原 alpha=0.1927490234375；關閉組只在記憶體把 alpha 設為 0，其餘權重、資料、輸入設定不變，原檔未修改。

| AP50–95 | MASF 開啟 | MASF 關閉 | 開啟－關閉 |
| --- | ---: | ---: | ---: |
| 整體 box | 0.630964 | 0.626017 | +0.004947 |
| ball box | 0.510747 | 0.505527 | **+0.005220** |
| bat box | 0.751180 | 0.746506 | **+0.004674** |
| 整體 keypoint | 0.912161 | 0.912907 | -0.000747 |
| ball keypoint | 0.876263 | 0.876010 | +0.000253 |
| bat keypoint | 0.948058 | 0.949804 | **-0.001746** |

因此，在這個現成 Pose 模型裡，MASF 對 ball／bat box 分別有約 **+0.522／+0.467 個百分點**的差異，但 bat keypoint 約 **-0.175 個百分點**。這是已訓練模型的推論消融，表示目前預測對該模組的依賴；不是「從頭不加 MASF 訓練」的對照，不能推導所有架構或訓練方法都會有同樣差異。

目前沒有本次驗證可用的「已用 Pose 訓練、插入 P2 的 Pose checkpoint」證據。已測 P2 是 COCO Detect checkpoint 在 BBAT5 影像上的 **box 泛化**，不是 P2 keypoint 增益。不能用原 Pose P3 階段的名稱混充 P2／P3 位置對照。

## 3. 評估方法與來源

- 資料：不可變 `bbat5-v1`，原正式 Pose YAML，683 張完整 validation；不切分、不抽樣、不修改 labels。
- Detect 使用原 Pose label parser 讀取既有框；僅將預測 COCO `sports ball=32 → ball=0`、`baseball bat=34 → bat=1`，排除其他類別，不把 COCO person=0 當 ball。
- 類別映射的合成資料檢查及實際 683 張資料數檢查均通過。Keypoints 不會被當成額外 box；Detect 表沒有假造 keypoint 指標。
- imgsz 640、batch 16、workers 4、BitTrue、PWL [-10, 0]、20 段、相同內部 AP 口徑。没有測硬體延遲，亦不聲稱純整數無除法部署。
- COCO 使用同 epoch 已完成 EMA 結果，不重跑既有正常驗證；各列附 checkpoint SHA256 與原 summary 路徑。
- Pose MASF 開啟組使用本次工作先前已完整重驗的 baseline；關閉組此次新驗證。兩者同來源 SHA256、同 BitTrue 與同 canonical split。

完整原始數值：[summary.json](<../artifacts/existing-masf-bbat-v1/summary.json>)。執行入口：[evaluate_existing_masf.py](<../evaluate_existing_masf.py>)。結果執行 exit 0，UTC 2026-09-10 03:15:37–03:16:03；七個 Detect 验證加一個 Pose 關閉驗證，皆完整 683 張。

## 4. 另行保留的 J0 與未執行工作

先前已啟動的無 MASF J0 完成 8 epoch／2984 步；COCO overall／person 固定 0.508330／0.627858，Pose AP 由初始 0.584586 恢復到 0.807073，但仍低於原獨立 Pose 0.912161，沒有通過 best_joint gate。這是換 trunk 後的 head 適應，不拿來當上述原 Pose MASF 開／關的公平對照。

新 Pose P3 MASF 正式 8 epoch **沒有開始**。短更新 smoke 在「optimizer 已清空梯度後讀取 `.grad`」的附加斷言退出；因此該失敗不能證明 MASF 梯度斷掉。原 runner 明確在更新後 `zero_grad(set_to_none=True)`；若未來恢復訓練，需將觀測移到 optimizer step 前並重新驗證，不能刪斷言偷跑。

P2 Pose 訓練入口只是未完成驗證的準備稿，沒有 GPU 執行，也沒有產生 P2 Pose 訓練成果。使用者改為優先現有權重後，不再修復／重啟新訓練 queue。

本次全部權重、failed smoke 與來源紀錄保留；不清理、不刪除、不 commit 或 push。下一步只由使用者決定是否採用 MASF，以及是否需要進一步 Pose 成對訓練。
