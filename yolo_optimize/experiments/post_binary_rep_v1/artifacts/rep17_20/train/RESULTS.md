# rep17_20：5 epoch 與獨立驗證結果

BinaryQK＋PWL 參考是已完成 scale_bias E5（其上游為 MASF B5），沒有重跑未改架構的加訓對照。所有差值不能分離結構與適應訓練的收益。

| AP50–95 (%) | MASF B5 | BinaryQK E5 起點 | Rep E5 | 相對 BinaryQK (pp) |
| --- | ---: | ---: | ---: | ---: |
| coco/box/map50_95 | 50.2215 | 50.5248 | 50.6181 | +0.0933 |
| coco/person/box/map50_95 | 62.4625 | 62.6208 | 62.6750 | +0.0543 |
| bbat/box/map50_95 | 60.9473 | 60.8534 | 60.8983 | +0.0448 |
| bbat/pose/map50_95 | 88.9071 | 88.8678 | 88.8582 | -0.0095 |
| bbat/ball/box/map50_95 | 49.0353 | 49.3118 | 49.2151 | -0.0967 |
| bbat/ball/pose/map50_95 | 85.3175 | 85.7615 | 85.4588 | -0.3028 |
| bbat/bat/box/map50_95 | 72.8593 | 72.3951 | 72.5815 | +0.1864 |
| bbat/bat/pose/map50_95 | 92.4968 | 91.9740 | 92.2577 | +0.2838 |

工程参考門檻（同時保護 BinaryQK E5 與原 MASF B5）：未通過，不建議取代。未自動升版；不增加 epoch。

COCO AP 最佳回合是 E5，僅作補充；主要比較仍固定 E5。

Float 與 BitTrue、逐回合完整指標、scale/bias 整數碼及完整續訓 checkpoint 均保存在本目錄。

硬體限制：此處未量測實板 latency／energy。Rep17/20 以已 fold 的單 Conv 作推論；scale/bias 固定常數，無逐圖片估計尺度；PWL[-10,0]20段，最後正規化仍是軟體除法参考。
