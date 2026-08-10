# YOLO11 BinaryAttention 完整實驗報告

本報告由 26 個 canonical research artifacts、`binary_attention_summary.json/csv` 與 `paper_qat_selection.json` 自動重建；指標均來自各 run 的 COCO validation artifact。

## 1. 研究問題與範圍

本實驗研究在 YOLO11m 偵測模型中，將 attention 的 Q/K 計算改為 binary/dual-basis 形式後，能否透過短期 QAT fine-tuning、知識蒸餾、relative-position bias 與局部 8/4-bit fake quantization 維持接近全精度 attention 的 COCO 準確率。

這是 **10-epoch attention-only adaptation**：只訓練 attention parameters，並非論文的 **300-epoch full-model** ImageNet recipe。因此結果可回答本專案的相對消融問題，但不能直接重現或取代論文主表。

- 模型與初始化：YOLO11m；所有 T/N run 從同一 full-precision source checkpoint 獨立初始化。
- 資料：完整 COCO2017 train2017 118,287 張、val2017 5,000 張；輸入 640 px。
- 訓練：10 epochs、seed 0、deterministic、AdamW、lr0 `5e-5`、cosine decay 至 `5e-6`、weight decay `0.02`。
- Batch：micro batch 16、gradient accumulation 8、effective batch 128；AMP；8 workers；disk cache。
- 參數範圍：主要 T variants 約 200,960 個 attention parameters 可訓練、約 19.91M 個 parameters 凍結；dense bias/N4 variants 約 217k 個可訓練參數。
- 權重一致性：validation metrics 與 strict checkpoint 都採同一個 epoch-last EMA 權重。
- 論文脈絡：https://arxiv.org/abs/2603.09582；官方程式：https://github.com/EdwardChasel/BinaryAttention。

## 2. 實驗流程

1. E 系列在不訓練下量測直接替換 attention 的準確率衝擊。
2. T0–T5 比較 FP、sign、scaled-sign 與三種 dual-basis attention；T3/T4/T5 本身不使用 KD。
3. 從 T1–T5 選出 T4，測試 T6 的 positional（O）、feature（F）、attention（A）KD 與兩兩組合。
4. 使用最佳 positional+feature 組合建立 T6，再比較 T7-D/T7-R 的 2D relative-position bias。
5. 在最佳 dense 2D bias 上測試 P、V、P/V 8-bit fake quantization。
6. N4 取 T7-D 的 dense-bias 結構選擇，但不繼承 T7-D 的 KD 或 checkpoint；N4 各 run 仍從同一 FP source 獨立初始化，測試 FP/8-bit/4-bit magnitude side channel，再將最佳 I4 與 P/V 8-bit 結合。

量化契約：scaled Q/K 使用 `mean_abs_channel_token_per_sample_head`；P8 使用 `static_unsigned_1_over_255`；V8 使用 `max_abs_token_per_sample_head_channel`。
COCO 尺寸分組採官方 bbox area：small `< 32²`、medium `32²–96²`、large `>= 96²` pixels；APs/APm/APl 由 canonical strict weights 在 val2017 5,000 張影像上以 COCOeval 重新計算。

## 3. 完整結果

### 3.1 E 系列：零訓練替換

參考值：`E0` COCO mAP50–95 = 0.51085。

| Variant | 實驗內容 | 原 mAP | COCO mAP | AP50 | APs | APm | APl | Precision | Recall | Δ COCO mAP | Binary QK / Softmax / PV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | FP attention，零訓練基準 | 0.50481 | 0.51085 | 0.68145 | 0.33363 | 0.56644 | 0.66956 | 0.73356 | 0.61810 | +0.00000 | 0 / 1 / 1 |
| E1-S | sign-only binary Q/K，零訓練 | 0.45070 | 0.45930 | 0.61505 | 0.28321 | 0.50373 | 0.62014 | 0.69075 | 0.54771 | -0.05155 | 1 / 1 / 1 |
| E1 | scaled-sign binary Q/K，零訓練 | 0.48121 | 0.48056 | 0.64232 | 0.29818 | 0.53223 | 0.64129 | 0.69613 | 0.58708 | -0.03029 | 1 / 1 / 1 |
| E2-DUAL | matched residual dual basis，零訓練 | 0.48710 | 0.49565 | 0.66194 | 0.31405 | 0.54803 | 0.65191 | 0.69549 | 0.59801 | -0.01521 | 2 / 1 / 1 |

零訓練時，sign-only 造成最大下降；scaled-sign 與 dual-basis 能保留更多準確率。這也說明後續 QAT fine-tuning 是必要步驟，而不是可省略的附加操作。

### 3.2 T0–T5：attention 結構與 QAT

參考值：`T0` COCO mAP50–95 = 0.51267。

| Variant | 實驗內容 | 原 mAP | COCO mAP | AP50 | APs | APm | APl | Precision | Recall | Δ COCO mAP | Binary QK / Softmax / PV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | FP attention-only fine-tuning 控制組 | 0.50672 | 0.51267 | 0.68243 | 0.33102 | 0.56532 | 0.67330 | 0.72326 | 0.62743 | +0.00000 | 0 / 1 / 1 |
| T1 | sign-only QAT | 0.50188 | 0.50789 | 0.67716 | 0.33074 | 0.55952 | 0.66853 | 0.72533 | 0.61481 | -0.00478 | 1 / 1 / 1 |
| T2 | scaled-sign QAT | 0.50346 | 0.50951 | 0.67849 | 0.33201 | 0.56311 | 0.66997 | 0.71815 | 0.62300 | -0.00316 | 1 / 1 / 1 |
| T3 | parallel dual binary attention | 0.50211 | 0.50798 | 0.67711 | 0.32850 | 0.55988 | 0.66805 | 0.72311 | 0.61283 | -0.00470 | 2 / 2 / 2 |
| T4 | full-basis residual dual attention | 0.50586 | 0.51163 | 0.68109 | 0.33103 | 0.56511 | 0.67134 | 0.72610 | 0.62149 | -0.00104 | 4 / 1 / 1 |
| T5 | matched-basis residual dual attention | 0.50447 | 0.51007 | 0.67914 | 0.32888 | 0.56377 | 0.66793 | 0.72167 | 0.62327 | -0.00260 | 2 / 1 / 1 |

T4（full-basis residual dual attention）是 T1–T5 中最佳 binary branch，AP50–95 0.50586，僅比 T0 低 0.00086。T2 的 scaled-sign 優於 T1 sign-only；T3 的 parallel dual 計算成本較高，但沒有換得較佳準確率；T5 matched-basis 優於 T3，但仍低於 T4。

### 3.3 T6：KD component 消融

參考值：`T4` COCO mAP50–95 = 0.51163。

| Variant | 實驗內容 | 原 mAP | COCO mAP | AP50 | APs | APm | APl | Precision | Recall | Δ COCO mAP | Binary QK / Softmax / PV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T4 | full-basis residual dual attention | 0.50586 | 0.51163 | 0.68109 | 0.33103 | 0.56511 | 0.67134 | 0.72610 | 0.62149 | +0.00000 | 4 / 1 / 1 |
| T6-O | T4 + positional KD | 0.50570 | 0.51131 | 0.68034 | 0.33076 | 0.56481 | 0.67016 | 0.72721 | 0.61900 | -0.00032 | 4 / 1 / 1 |
| T6-F | T4 + feature KD | 0.50547 | 0.51139 | 0.68085 | 0.33020 | 0.56481 | 0.66946 | 0.72464 | 0.62032 | -0.00024 | 4 / 1 / 1 |
| T6-A | T4 + attention KD | 0.50539 | 0.51151 | 0.68126 | 0.33252 | 0.56336 | 0.66843 | 0.73616 | 0.61417 | -0.00012 | 4 / 1 / 1 |
| T6-O/F | T4 + positional/feature KD | 0.50624 | 0.51176 | 0.68107 | 0.33126 | 0.56445 | 0.67106 | 0.71769 | 0.62611 | +0.00013 | 4 / 1 / 1 |
| T6-O/A | T4 + positional/attention KD | 0.50549 | 0.51146 | 0.68051 | 0.33251 | 0.56308 | 0.66911 | 0.72334 | 0.62305 | -0.00017 | 4 / 1 / 1 |
| T6-F/A | T4 + feature/attention KD | 0.50563 | 0.51184 | 0.68127 | 0.33378 | 0.56275 | 0.67061 | 0.71644 | 0.62619 | +0.00021 | 4 / 1 / 1 |
| T6 | 選定 positional/feature KD 的正式重現 | 0.50624 | 0.51176 | 0.68107 | 0.33126 | 0.56445 | 0.67106 | 0.71769 | 0.62611 | +0.00013 | 4 / 1 / 1 |

positional+feature（T6-O/F）是最佳 KD 組合，正式 T6 的 deterministic 重現得到相同 AP50–95 0.50624。單獨 positional、feature、attention 或含 attention 的兩兩組合均未超過 T4；因此本資料下的提升主要來自 positional 與 feature supervision 的互補。

### 3.4 T7：relative-position bias 與 P/V 量化

參考值：`T6` COCO mAP50–95 = 0.51176。

| Variant | 實驗內容 | 原 mAP | COCO mAP | AP50 | APs | APm | APl | Precision | Recall | Δ COCO mAP | Binary QK / Softmax / PV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T6 | 選定 positional/feature KD 的正式重現 | 0.50624 | 0.51176 | 0.68107 | 0.33126 | 0.56445 | 0.67106 | 0.71769 | 0.62611 | +0.00000 | 4 / 1 / 1 |
| T7-D | T6 配置 + dense 2D relative-position bias | 0.50616 | 0.51165 | 0.68107 | 0.33050 | 0.56429 | 0.67057 | 0.72343 | 0.62004 | -0.00010 | 4 / 1 / 1 |
| T7-R | T6 配置 + decomposed 2D relative-position bias | 0.50577 | 0.51166 | 0.68107 | 0.33218 | 0.56512 | 0.67014 | 0.71913 | 0.62532 | -0.00010 | 4 / 1 / 1 |
| T7-P | dense bias + P 8-bit fake quantization | 0.50562 | 0.51143 | 0.68081 | 0.33122 | 0.56340 | 0.67007 | 0.72469 | 0.62186 | -0.00032 | 4 / 1 / 1 |
| T7-V | dense bias + V 8-bit fake quantization | 0.50506 | 0.51082 | 0.67981 | 0.32882 | 0.56415 | 0.66931 | 0.72446 | 0.62080 | -0.00093 | 4 / 1 / 1 |
| T7-PV | dense bias + P/V 8-bit fake quantization | 0.50513 | 0.51114 | 0.68096 | 0.33196 | 0.56401 | 0.66936 | 0.72478 | 0.62197 | -0.00062 | 4 / 1 / 1 |

dense 2D bias（T7-D）優於 decomposed 2D bias（T7-R）0.00039，故後續採 dense 2D。P8、V8 與 P8/V8 都造成小幅下降，其中只量化 P 的 T7-P 最接近未量化 T7-D。這些差距僅有 1e-4 至 1e-3，單一 seed 下不應宣稱具統計顯著性。

### 3.5 N4：magnitude side channel

參考值：`T7-D` COCO mAP50–95 = 0.51165。

| Variant | 實驗內容 | 原 mAP | COCO mAP | AP50 | APs | APm | APl | Precision | Recall | Δ COCO mAP | Binary QK / Softmax / PV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T7-D | T6 配置 + dense 2D relative-position bias | 0.50616 | 0.51165 | 0.68107 | 0.33050 | 0.56429 | 0.67057 | 0.72343 | 0.62004 | +0.00000 | 4 / 1 / 1 |
| N4-FP | 非 KD magnitude side channel，FP magnitude | 0.50384 | 0.50949 | 0.67804 | 0.32854 | 0.56243 | 0.66922 | 0.72633 | 0.61812 | -0.00217 | 2 / 1 / 1 |
| N4-I8 | 非 KD magnitude side channel，8-bit magnitude | 0.50374 | 0.50918 | 0.67898 | 0.32825 | 0.56307 | 0.66789 | 0.71635 | 0.62523 | -0.00248 | 2 / 1 / 1 |
| N4-I4 | 非 KD magnitude side channel，4-bit magnitude | 0.50406 | 0.50986 | 0.67905 | 0.32985 | 0.56332 | 0.66830 | 0.72175 | 0.62094 | -0.00179 | 2 / 1 / 1 |
| N4-PV | N4-I4 + P/V 8-bit fake quantization | 0.50402 | 0.50976 | 0.67919 | 0.32896 | 0.56311 | 0.66836 | 0.72779 | 0.61781 | -0.00189 | 2 / 1 / 1 |

N4 內部以 I4 最佳（0.50406），比 N4-FP 高 0.00022，也略高於 I8；加入 P/V 8-bit 後為 0.50402。但所有 N4 variants 都低於 parent T7-D，表示 magnitude side channel 在本 10-epoch attention-only 設定下沒有帶來淨收益。

### 3.6 COCO mAP 與物件尺寸 AP 完整總表

`原 mAP50–95` 是訓練完成時保存的 Ultralytics evaluator 指標；`COCO mAP/APs/APm/APl` 是使用相同 canonical strict weight 重新推論後，由官方 COCO annotation 與 COCOeval 計算。兩者 evaluator 規則不同，因此 COCO−原 mAP 不應解讀為模型重新訓練後的提升。

| Variant | 原 mAP50–95 | COCO mAP50–95 | AP50 | AP75 | APs | APm | APl | COCO−原 mAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.50481 | 0.51085 | 0.68145 | 0.55205 | 0.33363 | 0.56644 | 0.66956 | +0.00604 |
| E1-S | 0.45070 | 0.45930 | 0.61505 | 0.49924 | 0.28321 | 0.50373 | 0.62014 | +0.00859 |
| E1 | 0.48121 | 0.48056 | 0.64232 | 0.52214 | 0.29818 | 0.53223 | 0.64129 | -0.00065 |
| E2-DUAL | 0.48710 | 0.49565 | 0.66194 | 0.53799 | 0.31405 | 0.54803 | 0.65191 | +0.00855 |
| T0 | 0.50672 | 0.51267 | 0.68243 | 0.55711 | 0.33102 | 0.56532 | 0.67330 | +0.00595 |
| T1 | 0.50188 | 0.50789 | 0.67716 | 0.55155 | 0.33074 | 0.55952 | 0.66853 | +0.00601 |
| T2 | 0.50346 | 0.50951 | 0.67849 | 0.55410 | 0.33201 | 0.56311 | 0.66997 | +0.00605 |
| T3 | 0.50211 | 0.50798 | 0.67711 | 0.55135 | 0.32850 | 0.55988 | 0.66805 | +0.00587 |
| T4 | 0.50586 | 0.51163 | 0.68109 | 0.55591 | 0.33103 | 0.56511 | 0.67134 | +0.00577 |
| T5 | 0.50447 | 0.51007 | 0.67914 | 0.55516 | 0.32888 | 0.56377 | 0.66793 | +0.00560 |
| T6-O | 0.50570 | 0.51131 | 0.68034 | 0.55478 | 0.33076 | 0.56481 | 0.67016 | +0.00561 |
| T6-F | 0.50547 | 0.51139 | 0.68085 | 0.55609 | 0.33020 | 0.56481 | 0.66946 | +0.00592 |
| T6-A | 0.50539 | 0.51151 | 0.68126 | 0.55600 | 0.33252 | 0.56336 | 0.66843 | +0.00612 |
| T6-O/F | 0.50624 | 0.51176 | 0.68107 | 0.55617 | 0.33126 | 0.56445 | 0.67106 | +0.00552 |
| T6-O/A | 0.50549 | 0.51146 | 0.68051 | 0.55666 | 0.33251 | 0.56308 | 0.66911 | +0.00597 |
| T6-F/A | 0.50563 | 0.51184 | 0.68127 | 0.55774 | 0.33378 | 0.56275 | 0.67061 | +0.00621 |
| T6 | 0.50624 | 0.51176 | 0.68107 | 0.55617 | 0.33126 | 0.56445 | 0.67106 | +0.00552 |
| T7-D | 0.50616 | 0.51165 | 0.68107 | 0.55534 | 0.33050 | 0.56429 | 0.67057 | +0.00549 |
| T7-R | 0.50577 | 0.51166 | 0.68107 | 0.55602 | 0.33218 | 0.56512 | 0.67014 | +0.00589 |
| T7-P | 0.50562 | 0.51143 | 0.68081 | 0.55566 | 0.33122 | 0.56340 | 0.67007 | +0.00581 |
| T7-V | 0.50506 | 0.51082 | 0.67981 | 0.55449 | 0.32882 | 0.56415 | 0.66931 | +0.00576 |
| T7-PV | 0.50513 | 0.51114 | 0.68096 | 0.55420 | 0.33196 | 0.56401 | 0.66936 | +0.00601 |
| N4-FP | 0.50384 | 0.50949 | 0.67804 | 0.55397 | 0.32854 | 0.56243 | 0.66922 | +0.00565 |
| N4-I8 | 0.50374 | 0.50918 | 0.67898 | 0.55312 | 0.32825 | 0.56307 | 0.66789 | +0.00544 |
| N4-I4 | 0.50406 | 0.50986 | 0.67905 | 0.55473 | 0.32985 | 0.56332 | 0.66830 | +0.00580 |
| N4-PV | 0.50402 | 0.50976 | 0.67919 | 0.55384 | 0.32896 | 0.56311 | 0.66836 | +0.00574 |

- COCO mAP50–95 最高：`T0`，`0.51267`。
- APs 最高：`T6-F/A`，`0.33378`。
- APm 最高：`E0`，`0.56644`。
- APl 最高：`T0`，`0.67330`。

## 4. 選擇結果與主要結論

- 全部正式 run 的 FP 控制組 T0：AP50–95 `0.50672`。
- 最佳 KD binary 結果 T6：AP50–95 `0.50624`，相對 T0 `-0.00048`。
- 最終候選族群最佳模型：`T7-D`，AP50–95 `0.50616`，相對 T0 `-0.00056`。
- N4 族群最佳 magnitude：`N4-I4`，AP50–95 `0.50406`。
- 最終 strict checkpoint：`../runs/T7-D/full/089411e2-9035-4602-9722-32edba717bd4/checkpoints/strict-last.pt`。

核心結論是：scaled/dual-basis QAT 可在只微調 attention 10 epochs 的條件下，將 binary attention 恢復到非常接近 FP 控制組；T4 是最佳基本結構，positional+feature KD 再縮小差距。Relative-position bias、P/V 量化與 magnitude side channel 沒有進一步超越 T6，但 T7-D 仍是預先定義 final-family 中最佳候選。

## 5. Final audit 失敗原因與修正政策

目前正式 final audit：`ok=true`；verified variants `26/26`；errors `0`。

26/26 canonical variants 均有 completed、valid-for-research artifact、COCO metrics、strict checkpoint、config hash、architecture manifest 與 attention diagnostics。原 audit 失敗並非缺少模型或指標。

失敗原因有兩項：

1. epoch EMA 會對即使凍結且值理論上不變的浮點 tensor 執行乘加，造成 ulp 等級捨入。本批 artifacts 的最大 frozen parameter delta 為 `6.67572021e-06`，最大非 attention BN buffer delta 為 `2.86102295e-06`；所有 run 都沒有遺失 frozen tensor，且 9 個 source-initialized attention tensors 確實改變。Audit 因此採 fail-closed absolute tolerance `1e-5`，超過才判定 freeze violation。
2. Ultralytics 清理 T0 `last.pt` 時把 epoch metadata 設為 -1，舊 repair 程式因此寫成 epoch 0。T0 的 `training_curves.csv` 與原始 `results.csv` 均有 10 個完整 epoch rows，現以此作為 epoch-last EMA 證據。

這項修正不改模型權重、不改 validation metrics，也不把大型差異藏在容差內；它只修正原本不適用於 EMA 浮點運算的 bit-exact 稽核條件，並保留最大差異與容差供重現。

## 6. 限制與論文使用方式

- 所有結果只有 seed 0；小於約 0.001 AP 的差距應視為趨勢，不能當作統計顯著結論。
- 本研究是 COCO detection 的 10-epoch attention-only adaptation，不是 300-epoch full-model replication。
- Q/K binary operation、P8/V8/INT8/INT4 都在 PyTorch 中做 QAT/fake quantization；尚未測量真實硬體 latency、energy、memory bandwidth 或模型檔案壓縮率。
- T7-D 是 final-family selection；若研究問題是純準確率，應同時報告 T0 與 T6，避免把 T7-D 誤稱為全矩陣最高分。
- 建議論文主表報 AP50–95、AP50、Precision、Recall；若要主張穩定優勢，需補多 seed 或更長訓練。

## 7. 研究產物

- `binary_attention_summary.csv/json`：26 個 canonical runs 的完整可機讀彙整。
- `run_reports_index.md`：每個 run 與 artifact 路徑。
- `paper_qat_selection.json`：T6、T7、N4 與 final-family 的選擇依據。
- `figures/`：loss、variant accuracy、T6/T7/N4 比較圖。
- `logs/formal-plan/final-audit.json`：最終 fail-closed audit 結果。

## 8. Canonical weight archive

已集中保存 `26` 個消融實驗正式權重。每個權重旁均保留 model YAML、resolved config、validation metrics、checkpoint manifest 與 status。

| Variant | AP50–95 | 保存權重 | SHA-256 | 載入驗證 |
|---|---:|---|---|---|
| E0 | 0.5048069544762941 | `../final_weights/E0/weight.pt` | `e5f59dc0f988ea90ce50358ac40a8db367974fb3720ccd093c76370b7f960345` | legacy_schema2_strict_state_dict_reload |
| E1-S | 0.45070248673578833 | `../final_weights/E1-S/weight.pt` | `57aab0a136b291a45fbcdb904752587cbbb8d40c7d264133c688c37579505cd2` | legacy_schema2_strict_state_dict_reload |
| E1 | 0.48121393727721495 | `../final_weights/E1/weight.pt` | `f42b6bf173ddeef5ccc51e0ce4d1527c2d60011c62b373ef4b296f17fbe38d5c` | legacy_schema2_strict_state_dict_reload |
| E2-DUAL | 0.4871000483801535 | `../final_weights/E2-DUAL/weight.pt` | `d3965671dbade20ec490c93b51c0d5cb337ec51bde51549eef3b4a3860f39433` | legacy_schema2_strict_state_dict_reload |
| T0 | 0.50672 | `../final_weights/T0/weight.pt` | `f8cff6e352ab6a91b81b5d046fc89857d54e144561b7eff5e2c004a446a41a20` | schema3_strict_reload |
| T1 | 0.50188 | `../final_weights/T1/weight.pt` | `40f459d536182d6e2952f7e430c2384d15c2fbeef182604d5ff6339d97306f06` | schema3_strict_reload |
| T2 | 0.50346 | `../final_weights/T2/weight.pt` | `3b1964380074ee5fc8f706293f82fa827da07bb6879c4af2da558cd43b894d48` | schema3_strict_reload |
| T3 | 0.50211 | `../final_weights/T3/weight.pt` | `9bc2eea0bc67da5b03b526479348e584fae56274bc94b059bc6f1ca7ee1ab8b7` | schema3_strict_reload |
| T4 | 0.50586 | `../final_weights/T4/weight.pt` | `e09ff0d1aed7479f6cbe6b922537377fb5d699ec0267c38c1b1517b77f93c970` | schema3_strict_reload |
| T5 | 0.50447 | `../final_weights/T5/weight.pt` | `5990b765f65fc3028970c86e627f4824f928d22ad93a9ca3092ed6da93b3f0d5` | schema3_strict_reload |
| T6-O | 0.5057 | `../final_weights/T6-O/weight.pt` | `2af313905e152310fc25f5d6ba80475ff6a8a9982c301e4be5afc743c19dc291` | schema3_strict_reload |
| T6-F | 0.50547 | `../final_weights/T6-F/weight.pt` | `a7a62ca2c02173630032bf751138557ddb655ae594a0705cbda18614983f00fb` | schema3_strict_reload |
| T6-A | 0.50539 | `../final_weights/T6-A/weight.pt` | `284a8636bfe39f6dbeb6929c75a0af9b8a1407c738def4c1f3f48a3203cb94fa` | schema3_strict_reload |
| T6-O/F | 0.50624 | `../final_weights/T6-O-F/weight.pt` | `b3d3d684171fff48be7a1a2355559e268844655bfde815a2d1af91c13f7ba11c` | schema3_strict_reload |
| T6-O/A | 0.50549 | `../final_weights/T6-O-A/weight.pt` | `3a24eae6a84e53de2ffe84786457de65770f6bb18c3efbcbd46765fbe5037233` | schema3_strict_reload |
| T6-F/A | 0.50563 | `../final_weights/T6-F-A/weight.pt` | `e32f48b7a83cfceaf1b41896a413b70d10e5320e23d8cbebca74da6badbb8423` | schema3_strict_reload |
| T6 | 0.50624 | `../final_weights/T6/weight.pt` | `550cb5124d4f7380c17d4e7ea3e03f2fb86704e0d9888512ec3c78953a0d77dd` | schema3_strict_reload |
| T7-D | 0.50616 | `../final_weights/T7-D/weight.pt` | `f81d98c5d5852a30ddbc7d1ff26291787e3dbad067d3d4f030f1232d14449812` | schema3_strict_reload |
| T7-R | 0.50577 | `../final_weights/T7-R/weight.pt` | `dfdf833f901d74c3bc18fac8406cd6c9094e0900846d13913947f683b4268009` | schema3_strict_reload |
| T7-P | 0.50562 | `../final_weights/T7-P/weight.pt` | `95b6accfb58c31ed5c907cd488f149aff987148841988d74779a55bd3f6a7ea1` | schema3_strict_reload |
| T7-V | 0.50506 | `../final_weights/T7-V/weight.pt` | `7e0e5429043810025d34f019ca0dd936624f78b52b71e915f71649bef59149b6` | schema3_strict_reload |
| T7-PV | 0.50513 | `../final_weights/T7-PV/weight.pt` | `d5a116509fb8b4db4d8f8a3001186b7e1972536d86d25d47e25274bcfaba2b26` | schema3_strict_reload |
| N4-FP | 0.50384 | `../final_weights/N4-FP/weight.pt` | `614cbeb78a511bc22998511b2ffaa9ed2be31193cfd0acc021c0ff657161168c` | schema3_strict_reload |
| N4-I8 | 0.50374 | `../final_weights/N4-I8/weight.pt` | `cd92ed9fd26a6e364c238ca9c035e297e74c4f9740b67b1ee542dace6112c8d1` | schema3_strict_reload |
| N4-I4 | 0.50406 | `../final_weights/N4-I4/weight.pt` | `624d8453b65f75f5067a72aeb0a6de8e4658ddffe96df58daa2bf5b274af9c38` | schema3_strict_reload |
| N4-PV | 0.50402 | `../final_weights/N4-PV/weight.pt` | `dac4c0445137281634e510fc934064973e602a7071d1a2ba390ad510210c3a3a` | schema3_strict_reload |

保存目錄的 `weight.pt` 是與報告 metrics 對應的 canonical strict weight，不可刪除。已刪除 `58` 個冗餘 Ultralytics `best.pt`/`last.pt`，共 `2.554 GiB`。這些檔案的 optimizer/resume 狀態只能藉由重新訓練重建；26 組 canonical strict weights、模型設定、驗證指標及雜湊均完整保留。
