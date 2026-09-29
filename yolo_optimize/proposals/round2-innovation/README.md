# 第二輪：固定成本下的 BinaryQK 與任務保護創新

日期：2026-09-08。狀態：`proposed`；沒有啟動實驗。person-only 暫緩，維持 COCO80 Detect＋BBAT5 Pose。這是第二輪的統整入口，不覆寫[第一輪](<../integrated-roadmap/README.md>)。

> 本輪唯一當前方向1順序以[方向1 master plan](<../integrated-roadmap/direction1-master-plan.md>)為準，optimizer 配套見[optimizer policy](<../integrated-roadmap/optimizer-policy.md>)。`final/full35` 是接續來源；先比較 J3 `best_joint`／J3 `best_pose`／J2 `best_joint` 後選 `PSEL`，尚未選定，原 J3 `best_joint` 只作 `B0` 歷史分數參照，`best_detect` 不直接作 task=both。`training_ready=false`，不跑 GPU；基準可靠前不進第二輪。方向1後才考慮 R2-REGION，且僅在 teacher／有效梯度具備時啟動，不混第一部分名詞。正式 baseline 維持 AdamW guard；MuSGD 有 builder 支援但只作另案 paired challenger，中途切換的 state／LR 校準尚待驗證。新 HOG 名稱為 `W-HOG10`，不與歷史 `F2-PRE-HOG9` 混用；MASF／HOG／QK／optimizer 不同時首次開。

結論：有比繼續堆模組更值得研究的問題，但尚不能說精度一定更好。首選保留兩個假說；通用旋轉、普通KD、修改loss名稱，都不能單獨當作創新。

| 第二輪假說 | 原本 | 提案差異 | 創新待證之處 |
|---|---|---|---|
| R2-BASIS：任務保護的固定二值基底 | 固定 identity＋Hadamard、固定PoT係數 | 離線選固定符號基底，兼顧Detect／Pose ranking；部署不做每圖基底／scale選擇 | 受硬體限制的dual-basis選擇目標，是否勝過單一隨機基底及平均誤差選擇 |
| R2-REGION：固定預算的區域排序蒸餾 | 全域／均勻 attention KD | 同樣token-pair預算，按物件與有效關鍵點分配，兼留背景 | 是否勝過均勻KD和普通前景KD，而非只是加權或增加teacher計算 |

方向1：`R2-REGION` 固定預算區域排序蒸餾的[完整規格](<region-ranking-full-spec.md>)。

研究主題首選 R2-BASIS，因為它直接處理二值化的資訊損失且有明確代數推導；但本地hardware contract目前確實凍結Q/K與係數，因此它是需要另立硬體版本契約的研究候選。沿目前限制，先考慮R2-REGION；它也必須先通過第一輪可訓練scope查核。不是兩個都必跑，也不首輪疊加。

~~~text
第一輪固定未fuse winner、FP teacher、evaluator與資料
                    │
          ┌─────────┴──────────┐
          ↓                    ↓
 R2-BASIS獨立比較       R2-REGION獨立比較
 現行／隨機／任務選擇   無KD／均勻KD／區域KD
          │                    │
          └─────────┬──────────┘
                    ↓
     有訊號才補創新對照與paired seeds
                    ↓
       選winner → 重新fuse／校準／部署驗證
~~~

這裡的「第二輪」指取得第一輪可重現基準之後，不是把改動直接套在已匯出的INT8 engine。第一輪未處理的 Q/K trainability／STE／parent 問題按候選 scope 個別檢查；第二輪 KD 只有在有 teacher 與有效梯度時才啟動，不重新包裝成第二輪創新。

- [機理推導、創新邊界與風險](<../../docs/research/2026-09-08-round2-innovation-analysis.md>)
- [最小實驗順序與停止條件](<plan.md>)
- [一手來源摘錄](<../../docs/research/2026-09-08-round2-primary-evidence.md>)
- [中文工作紀錄](<../../docs/worklogs/2026-09-08-round2-innovation-research.md>)

本次僅研究與文件整理，GPU、模型forward、訓練、校準及AP測量均為0；尚未完成系統性新穎性檢索，不宣稱全球首次。
