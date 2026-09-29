# 2026-09-10：MASF 實驗結果與 P2 完成狀態稽核

## 任務與範圍

依使用者要求檢查 MASF 實驗結果。主範圍為 `yolo_optimize` 融合前 Full35-B100 的 P3 shared／fork、提高 head LR、訓練梯度橋接及最後 P2 配對；另區分融合後移除診斷與 `yolo_quantize` 的 MASF 三元量化實驗。

單一主代理唯讀查核程式、設定、summary、progress 與 queue。新增本工作紀錄及可重算附件，更新工作紀錄索引與兩處過期的 P2 狀態入口；沒有修改模型程式、資料、權重或 queue 原始證據，沒有啟動／停止 GPU 訓練，也沒有開始新的融合工作。

## 核心結論

融合前目前測試的 MASF 位置與訓練配方均沒有達到既定增準門檻。這是「目前沒有足夠證據支持採用」，不代表 MASF 在任何模型或訓練條件下一律無效。

重要的新發現：最後 P2 配對已完成，queue 為 `awaiting_analysis`，兩組均為 `complete`、各 5 個完整 epoch；原 README 還寫正式訓練已啟動，沒有更新完成結果。重算原始 summary 與 gate 判定完全一致，5 個 epoch 全部不合格。

## 最新 P2 結果

兩組同起點為已驗證的無 MASF control E8 EMA，來源 checkpoint SHA-256 `3aeeaec2b379d465ceb5aee6051e5c11772e856497676ec9c87f2cb511a0c06c`。相同 AdamW、head LR1e-5、fresh optimizer、warmup1、10-epoch horizon 的前5回合、EMA age14800、physical32×4＝logical128、imgsz640、完整 COCO train118287／val5000。皆訓練原 Detect head；候選另加入 P2 MASF，其他參數與全 BN 統計固定。

P2 在 layer2、stride4、256通道；Detect 仍使用 P3/P4/P5，沒有新增 P2 prediction head。preflight 記錄初始完整 COCO 四項 AP 差值為0、零gate整圖等價、非零gate可反傳；兩處 PWL 範圍皆為[-10,0]、20段。這些是既有驗證證據，本次未重新執行。

以下 AP 為 EMA COCO internal mAP50–95，顯示成百分比；差值為候選減去同回合對照的「百分點」，不是相對百分比。

| E5 指標 | 無 MASF control | P2 MASF | 差值（百分點） |
| --- | ---: | ---: | ---: |
| COCO overall | 50.841959 | 50.827072 | −0.014887 |
| Person | 62.769817 | 62.779742 | +0.009926 |
| Sports ball | 51.555115 | 51.429993 | −0.125122 |
| Baseball bat | 48.066534 | 48.084257 | +0.017723 |

完整主要指標趨勢如下，沒有只挑 E5 下結論：

| Epoch | Overall 差值（百分點） | Person 差值（百分點） | 原定 gate |
| --- | ---: | ---: | --- |
| 1 | +0.002421 | −0.002451 | 未過 |
| 2 | −0.009149 | +0.005476 | 未過 |
| 3 | −0.006292 | +0.008000 | 未過 |
| 4 | −0.016891 | +0.029512 | 未過 |
| 5 | −0.014887 | +0.009926 | 未過 |

原 gate 要求：同回合 overall、person 都不低於對照，至少一項增加0.001（即0.1個百分點），且两項都不低於 parent。E1 person 下降，E2–E5 overall 下降；各回合最大主要指標增量也沒有到0.1個百分點。不是因為四捨五入或極微小 parent 差值才沒過。

E5 的 P2 主要 AP 均略高於自己的 parent，但這不能取代與同預算 control 的比較。新增訓練本身也可能帶來變動，不能把相對 parent 的所有改善歸給 MASF。

P2 EMA alpha 從E1的0.0004666，逐步到E5的0.0164923；没有一直停在0，也沒有碰到±0.25上限。非零alpha及既有smoke梯度只排除完全沒有開啟的情況，並不證明充分收斂。

平均每 epoch（含 EMA／live 驗證）為 control489.40秒、P2 680.30秒，即8.16對11.34分鐘，增加约39.0%。這是歷史訓練回合耗時，不能當成部署推論加速或延遲測量。P2增加75,777參數，既有前置估算每張640影像多約1.900544G卷積MAC，未包含BN／activation／記憶體流量。

來源：[P2 control summary](<../../yolo_optimize/studies/pre-fusion-full35-b100/artifacts/masf-p2-control-v1/summary.json>)、[P2 candidate summary](<../../yolo_optimize/studies/pre-fusion-full35-b100/artifacts/masf-p2-p2-v1/summary.json>)、[P2 queue 與原 gate](<../../yolo_optimize/studies/pre-fusion-full35-b100/artifacts/masf-p2-queue-v1-state.json>)、[原始設計](<../../yolo_optimize/docs/worklogs/2026-09-10-p2-masf-last-trial.md>)。

## 融合前 P3 與梯度系列

以下各列清楚指定比較對照與回合，不把不同訓練預算混為公平勝負；完整30筆同回合差值見附件CSV。

| 方法 | 完成範圍／比較 | 指定回合 overall／person 差值（百分點） | 判斷 |
| --- | --- | --- | --- |
| 原 shared P3 位置 | shared 與無 MASF control，E1–E5 | E5：−0.035946／+0.005940 | 未取得足夠增益 |
| P3 Detect-only fork | fork 與同 control，E1–E5 | E5：−0.001069／−0.006107 | 移位減輕部分 shared 退化，但不代表增準 |
| P3 head LR提高5倍 | fork 與無 MASF control，配對續訓E6–E10 | E10：−0.008377／+0.009731 | 加訓後仍無穩定增益 |
| one-to-one梯度橋接 | bridge 與原生fork，從同一fork E5續訓E6–E10 | E8：+0.005117／+0.012210 | 部分小幅改善，未達門檻 |
| 橋接後與無 MASF比較 | bridge 與無 MASF control，同E8 | −0.005514／−0.003533 | 仍沒有支持保留MASF的足夠精度收益 |

最後一列兩個E8候選已完成獨立匯出及完整COCO5000重驗，四項AP差值皆0；8張固定validation圖片推論只屬案例，不是獨立test。[集中驗收報告](<../../yolo_optimize/reports/direction1-20260910/README.md>)與[原始驗收summary](<../../yolo_optimize/studies/pre-fusion-full35-b100/artifacts/direction1-candidate-verification-v1/summary.json>)保留完整數值。

已完成的學習與梯度診斷表明：P3 shared／fork的context有更新，E5 gate分別約0.0202／0.0161，不是完全沒有學到參數。原生one-to-one detach確實阻止它直接監督MASF，原one-to-many則仍可傳梯度；增加經校準的one-to-one梯度後雖有局部微幅改善，仍未達方法收益門檻。因此不能再單靠「可能沒有梯度」解釋全部負面結果。這輪真實loss校準只有256張、單一配對seed，不能聲稱已找出所有原因。

## 融合後實驗必須分開解釋

舊融合後模型含BBAT5 Pose任務。直接關掉shared MASF時，Ball Pose mAP50–95約下降0.446451個百分點。接續的無MASF短恢復於E4因Ball Box相對原BEST下降0.606903個百分點超過安全線而停止；沒有得到可接受的no-MASF bridge。

因此舊融合後實驗的決策是保留原shared MASF；不能把本次融合前的「新方案不採用」解讀成應立即刪除舊融合模型中的MASF。兩者任務、parent與訓練歷史不同。[舊bridge結果](<../../yolo_optimize/docs/worklogs/2026-09-09-bridge-result-qk-diagnostics.md>)與[原始summary](<../../yolo_optimize/artifacts/direction1-20260908/masf-off-parent-bridge/summary.json>)相符。

## Quantize 的 MASF 實驗回答另一個問題

`masf-exact-ternary`、`masf-paper-twn`、`masf-twn`三組QAT各完成5epochs。以下E5數值是相對量化accepted reference、包含activation替換的最差總下降，並非MASF單一模組的獨立下降。

| QAT配置 | 最差mAP50下降（百分點） | 最差mAP50–95下降（百分點） |
| --- | ---: | ---: |
| Exact ternary | 0.948787 | 1.068758 |
| Paper TWN | 0.874402 | 1.064401 |
| Filterwise TWN | 0.826444 | 0.981274 |

三者E5皆在量化搜尋門檻內（mAP50下降不超過1.5pp、mAP50–95不超過4pp）。這支持「既有模型的MASF權重可在這些搜尋配置下接受量化」，不支持「加MASF比不加更準」。它們使用量化parent與BBAT5 search資料範圍，不能與融合前COCO增準或正式BBAT5 Pose結果直接比較。[QAT來源報告](<../../yolo_quantize/docs/reports/2026-09-07-continuous-qat-recovery-results.md>)。

## 實務判斷

按本次結果，融合前目前這幾套MASF方案沒有足夠收益抵銷增加的計算與訓練成本。P2已是既定最後位置實驗，5回合全不合格；原queue亦記錄失敗後方向為`combine_without_masf`。本次查核支持結束這條MASF增準分支、以已驗證無MASF來源作後續規劃，不支持無限延长相同配方。

這是結果判讀，不是新模型升格或已開始後續訓練的宣告。本次沒有更動原模型，既有融合後權重與所有實驗仍保留。

## 驗證方式與結果

執行：

```bash
python3 docs/worklogs/assets/2026-09-10-masf-results-audit/recalculate.py
```

核對8組融合前run共40個完整epoch，每回合118287張／925步；40份epoch checkpoint存在、320個EMA／live AP有限且在0–1。重算6種比較共30筆同回合四項AP差值，核對共同parent與首macro trace；後者不等於本次逐張驗證所有資料順序。

P2兩組progress各4625筆，E1–E5各925筆，最終optimizer_steps=4625。重算五回合eligible皆false，與queue差值逐項一致。另驗證三組MASF QAT completion為5epochs，核對completion及15份逐回合metrics檔案SHA-256與既有來源pin一致。

附件：[audit JSON](assets/2026-09-10-masf-results-audit/audit.json)、[同回合差值 CSV](assets/2026-09-10-masf-results-audit/paired-deltas.csv)、[重算程式](assets/2026-09-10-masf-results-audit/recalculate.py)。本次未 import Torch、未重跑 GPU 推論，不把既有驗收冒稱本次新測試。

更新study README與集中報告頂端的P2狀態，保留歷史實驗敘述；根README已有資料集規範與工作紀錄入口，不重複修改。新增文件與導航連結完成存在性檢查。

## 困難與解法

預設sandbox再次出現`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`，改用通過審核的唯讀命令與工作紀錄寫入；沒有自動審核拒絕。文件有過期「尚在執行」敘述，以完成summary、完整progress及queue交叉核實後修正最新入口。其餘困難：無。

## 未解事項與風險

沒有多seed顯著性、獨立test、使用者實際影片或部署硬體延遲資料；微小AP差異不解讀成統計顯著。P2有訓練期EMA／live完整驗證紀錄，但未在本次執行訓練後P2候選的獨立export全量重驗。由於P2未過方法gate，本次不額外啟動该驗收或新訓練。未證明更長訓練或其他scope永遠無效，也沒有證據支持繼續投入同一配方。
