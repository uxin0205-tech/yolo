# Clean BBT5 正式實驗報告

## 結論摘要

- Validation 選出的 strict-fair 模型：B0-Clean。
- 完成 12 個 strict-fair 架構 × 2 seeds，加 2 個 P2 control stages × 2 seeds，共 28 個正式 jobs。
- 所有模型從官方 COCO80 YOLO11m clean initializer 開始；沒有使用看過 BBT5 的 detect initializer。
- Historical test 已被先前研究查看，只能作歷史報告，不能宣稱 unseen BBT5 泛化。

## 資料與統一評估

- Source Dataset：bbt5-detect-baseline/dataset。
- Locked Dataset：train 1,987、validation 300、historical test 291 個 unique frames；group/hash overlap 皆為空。
- Trainer YAML 物理上沒有 test key；selection.json 凍結後才執行 historical test。
- 統一設定：imgsz 640、batch 16、conf 0.001、IoU 0.7、rect、FP32 inference、faster-coco-eval 1.7.2。
- 數值均列兩個 seeds 的 mean ± sample std；只有兩個 seeds，仍不足以精確估計 variance。

## 每個實驗在做什麼

| 實驗 | 類別 | 唯一差異 |
|---|---|---|
| B0-Clean | strict_fair | 原始 P3/P4/P5 三尺度 YOLO11m baseline；不加入 P2 或 MFAM。 |
| P2-Direct-Clean | strict_fair | 四尺度 P2/P3/P4/P5 baseline；新增標準高解析 P2 detection head。 |
| P2-PaperFormula-Clean | strict_fair | 在 P2 加入論文式 MFAM：DW3、DW5、factorized 7、factorized 9，加 identity 與兩層 1x1 residual fusion。 |
| P2-Lite35-Clean | strict_fair | 在 P2 只保留 DW3、DW5 與兩層 1x1 residual fusion。 |
| P2-Lite35-F7-Clean | strict_fair | 在 P2 使用 DW3、DW5、1x7→7x1 與兩層 1x1 residual fusion。 |
| P2-Partial50-Clean | strict_fair | P2 前 50% channels 使用 Lite35 MFAM，其餘 channels exact bypass。 |
| P2-Partial25-Clean | strict_fair | P2 前 25% channels 使用 Lite35 MFAM，其餘 channels exact bypass。 |
| P3-M7-Clean | strict_fair | 只在 P3 加入 legacy MFAM：DW3、DW5、factorized 7 與單層 1x1 fusion；不含 9x9。 |
| P3-Lite35-Clean | strict_fair | 只在 P3 加入 DW3、DW5 與論文式雙 1x1 fusion。 |
| P3-Lite35-F7-Clean | strict_fair | 只在 P3 加入 DW3、DW5、1x7→7x1 與論文式雙 1x1 fusion；不含 9x9。 |
| P3-Partial50-Clean | strict_fair | P3 前 50% channels 使用 Lite35 MFAM，其餘 channels exact bypass。 |
| P3-Partial25-Clean | strict_fair | P3 前 25% channels 使用 Lite35 MFAM，其餘 channels exact bypass。 |
| P2-Control-Clean-Head | optimization_control | P2 最佳化 control A：只訓練新增 P2 slot/head，最多 20 epochs；不進 strict-fair ranking。 |
| P2-Control-Clean-Full | optimization_control | P2 最佳化 control B：承接同 seed 的 Head best，全模型最多 80 epochs、lr0=0.001；不進 strict-fair ranking。 |

## Strict-fair validation 排名

| 排名 | 實驗 | mAP50–95 | AP_S | AP_M | AP_L | Ball AP | Bat AP | GFLOPs | FP16 p50 ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | B0-Clean | 0.46070 ± 0.01995 | 0.24724 ± 0.03167 | 0.70516 ± 0.01692 | 0.89839 ± 0.04302 | 0.26499 ± 0.07677 | 0.65641 ± 0.03687 | 67.646 | 3.641 |
| 2 | P2-PaperFormula-Clean | 0.45251 ± 0.01368 | 0.25092 ± 0.01847 | 0.69509 ± 0.02978 | 0.89928 ± 0.01991 | 0.24056 ± 0.04743 | 0.66445 ± 0.02008 | 91.072 | 4.716 |
| 3 | P2-Lite35-Clean | 0.45127 ± 0.01204 | 0.21872 ± 0.02605 | 0.74697 ± 0.00943 | 0.89633 ± 0.02129 | 0.21054 ± 0.01734 | 0.69200 ± 0.00674 | 90.862 | 4.382 |
| 4 | P3-Partial25-Clean | 0.44953 ± 0.01042 | 0.21238 ± 0.02747 | 0.74245 ± 0.00852 | 0.90210 ± 0.01845 | 0.20760 ± 0.01851 | 0.69147 ± 0.00234 | 67.779 | 3.666 |
| 5 | P2-Partial25-Clean | 0.44910 ± 0.00286 | 0.21827 ± 0.02689 | 0.70576 ± 0.03884 | 0.87320 ± 0.06380 | 0.23862 ± 0.06809 | 0.65958 ± 0.07381 | 89.122 | 4.427 |
| 6 | P2-Partial50-Clean | 0.44763 ± 0.00310 | 0.23113 ± 0.00875 | 0.73605 ± 0.02197 | 0.92498 ± 0.00085 | 0.21344 ± 0.00348 | 0.68182 ± 0.00968 | 89.493 | 4.607 |
| 7 | P2-Direct-Clean | 0.44659 ± 0.00760 | 0.23366 ± 0.02360 | 0.73796 ± 0.01884 | 0.86188 ± 0.02594 | 0.19323 ± 0.01770 | 0.69995 ± 0.00251 | 88.962 | 4.427 |
| 8 | P3-M7-Clean | 0.44113 ± 0.01316 | 0.23703 ± 0.01947 | 0.73260 ± 0.01502 | 0.85508 ± 0.04967 | 0.21152 ± 0.00419 | 0.67075 ± 0.03052 | 68.642 | 3.722 |
| 9 | P2-Lite35-F7-Clean | 0.44023 ± 0.00257 | 0.23680 ± 0.00115 | 0.73365 ± 0.02366 | 0.88474 ± 0.00596 | 0.19341 ± 0.01866 | 0.68705 ± 0.01352 | 90.954 | 4.395 |
| 10 | P3-Lite35-Clean | 0.43785 ± 0.00889 | 0.21039 ± 0.00639 | 0.69275 ± 0.08270 | 0.92409 ± 0.01017 | 0.19576 ± 0.00947 | 0.67993 ± 0.00831 | 69.435 | 3.656 |
| 11 | P3-Partial50-Clean | 0.43065 ± 0.02142 | 0.21349 ± 0.00888 | 0.71301 ± 0.03652 | 0.80945 ± 0.07221 | 0.19169 ± 0.00489 | 0.66962 ± 0.03794 | 68.121 | 3.715 |
| 12 | P3-Lite35-F7-Clean | 0.43041 ± 0.01951 | 0.21331 ± 0.02936 | 0.69055 ± 0.06347 | 0.89656 ± 0.03123 | 0.20681 ± 0.00441 | 0.65401 ± 0.04342 | 69.481 | 3.856 |

## Strict-fair historical test

| 實驗 | mAP50–95 | AP_S | AP_M | AP_L | Ball AP | Bat AP | Tiny Ball R |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0-Clean | 0.54255 ± 0.06874 | 0.27741 ± 0.08908 | 0.58040 ± 0.09002 | 0.63458 ± 0.05670 | 0.47867 ± 0.07515 | 0.60643 ± 0.06234 | 0.32000 ± 0.11314 |
| P2-Direct-Clean | 0.57584 ± 0.00559 | 0.34821 ± 0.00406 | 0.57575 ± 0.13304 | 0.64558 ± 0.00409 | 0.50675 ± 0.03027 | 0.64493 ± 0.01909 | 0.64000 ± 0.00000 |
| P2-PaperFormula-Clean | 0.59378 ± 0.02073 | 0.34615 ± 0.03928 | 0.61480 ± 0.06471 | 0.66526 ± 0.02448 | 0.53867 ± 0.04643 | 0.64890 ± 0.00498 | 0.58000 ± 0.08485 |
| P2-Lite35-Clean | 0.57019 ± 0.02350 | 0.36517 ± 0.00187 | 0.58767 ± 0.01612 | 0.65685 ± 0.03856 | 0.52100 ± 0.03750 | 0.61937 ± 0.00949 | 0.62000 ± 0.02828 |
| P2-Lite35-F7-Clean | 0.58982 ± 0.00541 | 0.35551 ± 0.03121 | 0.61844 ± 0.05446 | 0.65184 ± 0.01246 | 0.51742 ± 0.00038 | 0.66222 ± 0.01044 | 0.64000 ± 0.00000 |
| P2-Partial50-Clean | 0.57621 ± 0.01268 | 0.34018 ± 0.00546 | 0.56842 ± 0.00032 | 0.67090 ± 0.00694 | 0.52722 ± 0.00114 | 0.62519 ± 0.02422 | 0.64000 ± 0.00000 |
| P2-Partial25-Clean | 0.52308 ± 0.09361 | 0.29488 ± 0.06346 | 0.51108 ± 0.09629 | 0.61807 ± 0.08948 | 0.45727 ± 0.12246 | 0.58890 ± 0.06475 | 0.56000 ± 0.28284 |
| P3-M7-Clean | 0.57586 ± 0.00196 | 0.33897 ± 0.00838 | 0.59529 ± 0.04472 | 0.64750 ± 0.01017 | 0.51841 ± 0.00801 | 0.63330 ± 0.00408 | 0.62000 ± 0.02828 |
| P3-Lite35-Clean | 0.58962 ± 0.00991 | 0.37453 ± 0.03524 | 0.61821 ± 0.00406 | 0.67196 ± 0.00368 | 0.53046 ± 0.02162 | 0.64877 ± 0.00180 | 0.70000 ± 0.08485 |
| P3-Lite35-F7-Clean | 0.56029 ± 0.03731 | 0.34334 ± 0.05326 | 0.50051 ± 0.05806 | 0.67787 ± 0.02447 | 0.48943 ± 0.03464 | 0.63116 ± 0.03998 | 0.62000 ± 0.02828 |
| P3-Partial50-Clean | 0.54302 ± 0.05069 | 0.38391 ± 0.01329 | 0.57162 ± 0.04337 | 0.58869 ± 0.06739 | 0.44711 ± 0.09517 | 0.63894 ± 0.00622 | 0.68000 ± 0.05657 |
| P3-Partial25-Clean | 0.58576 ± 0.00292 | 0.36955 ± 0.02532 | 0.58133 ± 0.03313 | 0.67485 ± 0.00513 | 0.52978 ± 0.03468 | 0.64175 ± 0.02885 | 0.62000 ± 0.02828 |

## P2 optimization control（不與 strict-fair 混排）

| 實驗 | Split | mAP50–95 | Ball AP | Bat AP |
|---|---|---:|---:|---:|
| P2-Control-Clean-Head | val | 0.00599 ± 0.00224 | 0.00113 ± 0.00160 | 0.01084 ± 0.00608 |
| P2-Control-Clean-Head | test | 0.00579 ± 0.00803 | 0.00579 ± 0.00819 | 0.00579 ± 0.00787 |
| P2-Control-Clean-Full | val | 0.49288 ± 0.03876 | 0.26938 ± 0.09166 | 0.71638 ± 0.01415 |
| P2-Control-Clean-Full | test | 0.65049 ± 0.00138 | 0.62066 ± 0.01940 | 0.68033 ± 0.02216 |

## 分析

1. Validation 選出 B0-Clean（0.46070 ± 0.01995）；只領先第二名 P2-PaperFormula-Clean 0.00819 AP。差距小於兩者的 seed 標準差，應解讀為 B0 目前最穩健，而不是已證明絕對優勝。
2. P2-Direct 相對 B0 的 validation mAP50–95 下降 0.01410，GFLOPs 增加 31.5%、FP16 p50 latency 增加 21.6%。P2-PaperFormula 將品質差距縮至 0.00819，但 GFLOPs/latency 仍增加 34.6%/29.5%。
3. P2-Direct 的 Bat AP 比 B0 高 0.04355，但 Ball AP 低 0.07175；P2-PaperFormula 的 AP_S 只比 B0 高 0.00368。因此新增高解析 head 沒有自動轉化成整體或 Ball 優勢。
4. P3-Partial25 排名第 4，與 B0 的品質差為 0.01116，GFLOPs/latency 只增加 0.2%/0.7%；它是本輪新增模組中較合理的硬體友善折衷，但尚未超過 B0。
5. P2 Control-Full validation 為 0.49288，高於 B0 0.03218；然而它承接 Head-only checkpoint 且使用不同兩階段 schedule。這支持『P2 的主要瓶頸包含最佳化策略』，不能拿它宣稱 P2 架構公平勝過 B0。Head-only 本身接近失敗，亦表示只更新新增 head 不足。
6. Historical test 最高是 P2-PaperFormula-Clean（0.59378），並非 validation 選出的 B0。這個排名反轉不得用來事後改選模型；它反而顯示 validation/test 難度或分布不同，也再次說明 historical test 不能當 unseen selection set。
7. P3-Lite35-F7 與 P3-Partial50 的 seed 42 都在第 31 epoch 停止、最佳點在 epoch 1，但 seed 43 可訓練到 79–83 epochs；這是明顯的最佳化不穩定，不應只看兩 seed 平均掩蓋。
8. AP_S/M/L 使用 COCO area 定義；Ball tiny/small/large recall 另使用 locked dataset 的 short-side <8、8–16、>16 px 定義，兩者不可混稱。

## 公平性與限制

- Initializer 未接觸 BBT5，已解決舊 yolo11m_bat_detect_init.pt 的資料暴露問題。
- 前五個 formal jobs 在 patience=100 下啟動；之後改為 patience=30。既有曲線回放未改變前五組的 best checkpoint，但這仍是中途 protocol amendment，報告必須保留。
- 只有 seeds 42/43；0.x AP 的小差異不得過度解讀。
- Historical test 不是 Final Holdout；真正 unseen 結論仍需新影片／新比賽資料。
- Selection 只使用 validation，control 不進 strict-fair ranking。

## 證據位置

- artifacts/clean-bbt5-ablation/audit/formal_runs.json：28 jobs、hash、lineage、fresh-process reload。
- artifacts/clean-bbt5-ablation/final_audit.json：metrics/profile/selection/report 的最終一致性 gate。
- artifacts/clean-bbt5-ablation/evaluation/：逐 seed validation/test metrics、predictions、錯誤案例。
- artifacts/clean-bbt5-ablation/profiles/：Params、GFLOPs、activation/traffic、GPU peak memory、FP16 latency。
- artifacts/clean-bbt5-ablation/selection.json：不可變 validation selection freeze。
- clean_bbt5_study/results/per_seed_metrics.csv：逐 seed 原值。
- clean_bbt5_study/results/comparison_mean_std.csv：mean ± std 原始欄位。
