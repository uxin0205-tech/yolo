# architecture_2 正式報告

狀態：等待 architecture_1 正式交付驗收與正式實驗。

## 可重現性與 lineage

記錄硬體、軟體版本、git revision、spec version/hash、architecture/training/dataset/parent/checkpoint hashes、seed、physical batch、resolved args、optimizer groups、best epoch、停止原因、假設與未決衝突。不得用 smoke 指標填寫正式排名。

## 架構差異與權重轉移

對 C0/C1/C2/C3 與任何被 trigger 的 C3-P5/R1，記錄實測 graph delta、matched/missing/unexpected/shape-mismatch tensors、Params、FLOPs、peak VRAM、latency、receptive field／small-object 解讀，以及 Detect/Pose head contract。

## Detect 選型

報告 Bit-True mAP50-95 與相對 C0 的 PASS/CONDITIONAL/REJECT，並說明 C_best 為何存在或為 null。回答 C1 是否支持未來 e=0.25、C2/C3 的 accuracy/cost 取捨、C3-P5 是否恢復精度，以及 R1 是否通過 `max_abs_diff <= 1e-4`。

## Pose transfer（只有使用者決定執行時）

以相同 grouped split、Pose26 head seed、P1–P4 transitions、optimizer groups 與預算比較 C0/C_best。研究指標為 Pose mAP50-95，官方 Pose+Box combined fitness 另列。記錄 group/image 數量、零洩漏證據、四座標 patch manifest 與 `fliplr=0` 原因。若未執行，明確寫「使用者未啟用 Pose」，不可視為缺失結果。

## 未來雙來源融合

記錄來源 A/B 的 checkpoint、architecture YAML、head contract 與所有 hashes。`fusion_mode` 與 `switch_policy` 必須由使用者選定後才可填寫；不得把 weight interpolation、ensemble、routed switch 或 staged transfer 混為同一實驗。

## 量化

對 C0/C_best 報告 Q0-Q1 PTQ gap、Q0-Q2 QAT gap 與 Q2-Q1 recovery；列出 quantized/unquantized nodes，所有資料列標註 `simulation_only=true`，不得宣稱真實 INT8 deployment。

## 證據支持的下一步

只提出目前結果支持的下一個實驗。0.005/0.008 accuracy gates 與 8% cost gate 是本案工程門檻，不是通用標準。
