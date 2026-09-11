# EMA age 配對診斷：結果與後續

## 結論

延續 EMA age 把同一訓練軌跡的評分拉回原 BEST 附近；fresh EMA 與 live 仍退步。這確認 EMA age 對「同一軌跡的評分結果」有實測影響，不表示 live 學習已改善，也不代表新的架構帶來增益。原始 BEST 不自動替換。

## 同口徑結果

| BitTrue internal AP | 原 BEST | Live | Fresh EMA | Continued EMA |
|---|---:|---:|---:|---:|
| COCO Box | 0.49802234 | 0.49630797 | 0.49645324 | 0.49798245 |
| COCO Person Box | 0.62038149 | 0.62047695 | 0.62049103 | 0.62045619 |
| BBAT Box | 0.63003552 | 0.62241971 | 0.62229276 | 0.62965370 |
| BBAT Pose | 0.90371717 | 0.89839624 | 0.89849829 | 0.90401884 |
| Ball Box | 0.50743704 | 0.49738589 | 0.49712595 | 0.50692652 |
| Bat Box | 0.75263401 | 0.74745352 | 0.74745958 | 0.75238089 |
| Ball Pose | 0.85990872 | 0.85344145 | 0.85359953 | 0.86046282 |
| Bat Pose | 0.94752563 | 0.94335104 | 0.94339706 | 0.94757486 |
| Joint | 0.71117474 | 0.70719942 | 0.70724672 | 0.71122600 |

| Joint backend | 原 BEST | Live | Fresh EMA | Continued EMA |
|---|---:|---:|---:|---:|
| float | 0.71118211 | 0.70717311 | 0.70723696 | 0.71121794 |
| bittrue | 0.71117474 | 0.70719942 | 0.70724672 | 0.71122600 |

Continued 相對 fresh joint：+0.00397928；相對原 BEST：+0.00005127。後者太小，不宣稱有意義或顯著改善。Continued 仍有部分 Box AP 低於 parent，不能只報 joint 上升。

## 為何會這樣

EMA 更新為 Eₜ = dₜEₜ₋₁ + (1−dₜ)Wₜ，其中 Wₜ 是同一份 live 權重，d(u)=0.9999×(1−exp(−u/2000))。Fresh 的 u 從 0 開始，continued 從 paired snapshot 的頂層 ema_updates 開始；兩份模型起點完全相同。

本次成功 463 個 macros；ages 由 0／26597 變成 463／27060。Continued 的初始 parent 係數乘積為 0.95409191，亦即對可平滑 state 約保留 95.41% 起點權重，所有新 live 觀察合計約 4.59%。固定 state 則使用 exact-copy。

所以 continued 接近原 BEST 有合理機制：它對新、且此刻表現較差的 live 更新反應很慢。EMA 改善評分不等於 live 優化方向正確；後續仍需觀察 live，並用原 parent 作非劣比較。

## 公平性與驗證

這是一次 native 訓練、同步兩個 EMA observer，不是兩次獨立重複實驗。已由 4 項 CPU 測試確認：不回饋 loss／optimizer、相同 live 軌跡的各 EMA 結果與分開計算一致、AMP overflow 不多算更新、固定 state 不變。GPU 結束後的更新計數、固定 state 與驗證前後 state digest 也通過檢查。

原 source、原 BEST、COCO80 與 canonical BBAT5 未變；HOG、MASF 移位、BinaryQK、RepConv 均未加入本次診斷。warmup=1 epoch，AdamW／LR／BN／criterion 皆沿用既定對照。完整 snapshots 在驗證之前保存，兩份共用同一 epoch-end RNG 邊界；三份 inference 權重明確區分 live／ema。

GPU child 執行約 13.08 分鐘。600 秒監測及提前完成偵測由 supervisor 執行；本次只有一個正式訓練 epoch，不能列成兩個獨立 epoch 的統計重複。

## 後續決策

後續既定 native5／HOG10 對照共同採 parent EMA age，兩者仍從同一原始 BEST 重建 fresh optimizer，不從本次診斷候選偷偷續跑。每 epoch 另記 live BitTrue AP 作健康診斷；模型評分與原來的安全暫停仍以 EMA 八項 AP 對 parent 為準。這是共同續訓規則，不是 HOG 特有收益。

先做原生 5-epoch 對照並依安全 gate 決定是否繼續；HOG 的 train-only μ 尚未校準，不能宣稱 HOG 已具備精度收益。部署仍保留原 BEST。

## 產物

結果目錄：`/home/uxin/yolo/yolo_optimize/artifacts/direction1-20260908/ema-age-paired`。

- `summary.json`：三版本 Float／BitTrue 完整結果。
- `checkpoints/fresh-epoch-0001.pt`、`continued-epoch-0001.pt`：相同 live 邊界的兩份 snapshot。
- `inference/live-epoch-0001.pt`、`fresh-epoch-0001.pt`、`continued-epoch-0001.pt`：僅供診斷，未升格 BEST。

工作紀錄見 [EMA age 診斷](<../../docs/worklogs/2026-09-08-ema-age-diagnostic.md>)。
