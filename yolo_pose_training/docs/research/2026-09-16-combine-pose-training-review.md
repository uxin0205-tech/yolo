# 參考 yolo_combine 改善官方 YOLO26m Pose 訓練

## 結論

仍有改善機會，但目前沒有證據顯示只增加原方案的 epoch 就會進步。值得優先測試的是較低 MuSGD LR、較平緩的 LR 排程、逐步解凍，以及關閉 Mosaic 的整套 640 訓練策略。各項皆是待驗證假設，不是保證提高精度的方法。

本次只研究、建立候選設定，沒有啟動 GPU，也沒有修改已完成的四階段配置。所有候選仍使用原 bbat5-v1 train／val assignment，不重新切分、不抽樣、不修改影像或 labels。

## 實際讀取的證據

1. [正式 Pose stage policy](/home/uxin/yolo/yolo_combine/src/yolo_combine/pose_stages.py) 與[底層階段定義](/home/uxin/yolo/yolo_combine/src/yolo_combine/_pose_stages_impl.py)。只讀底層檔會誤讀 lr0=0.01；公開 policy 會覆寫成 0.00038。
2. [P3 實際 args.yaml](/home/uxin/yolo/yolo_combine/variants/full35/artifacts/pose/p0-full35-p3-b32a4-e100max-seed0/args.yaml)、同目錄 results.csv，以及 P1／P2 實際 args.yaml／results.csv。數字以已執行 artifact 為準。
3. [獨立 Pose 正式評估](/home/uxin/yolo/yolo_combine/variants/full35/artifacts/standalone-baseline/p3-seed0/validation/bittrue/metrics.json)、[formal gate](/home/uxin/yolo/yolo_combine/variants/full35/baselines/formal-gate-p3-seed0.json)、[評估設定](/home/uxin/yolo/yolo_combine/src/yolo_combine/validation.py)。
4. [模型來源](/home/uxin/yolo/yolo_combine/src/yolo_combine/source.py) 與[Pose trainer](/home/uxin/yolo/yolo_combine/src/yolo_combine/baselines.py)：Full35 自 Detect 模型移植 trunk，建立新 Pose26 head，包含 MASF 與修改版 attention；不是本專案官方 pose.pt 的同起點實驗。
5. [本次統一 960 結果](/home/uxin/yolo/yolo_pose_training/artifacts/comparison.json)、各階段 results.csv，以及本地 Ultralytics 8.4.90 trainer 的 accumulate／weight decay／warmup 實作。
6. [官方 YOLO26 訓練配方](https://docs.ultralytics.com/guides/yolo26-training-recipe/) 和[訓練參數](https://docs.ultralytics.com/modes/train/)。官方配方中 Detect S/M/L/X 也有 MuSGD lr0=0.00038、momentum=0.948、weight_decay=0.00027；該表是 COCO Detect 配方，不是 BBAT5 Pose 的最優保證。實際版本以本地套件為準。

## yolo_combine 的獨立 Pose 訓練方法

| 階段 | 訓練範圍 | 設定／實跑 epochs | 最佳 epoch | Physical batch | nbs |
| --- | --- | --- | --- | --- | --- |
| P1 | 凍結 layers 0–22，只訓練 head | 17／17 | 17 | 128 | 128 |
| P2 | 凍結 layers 0–10，訓練 neck＋head | 22／22 | 20 | 64 | 128 |
| P3 | 解凍可訓練參數，仍保護硬體固定參數 | 100／88，patience 20 | 68 | 32 | 128 |

三階段全部使用 640、MuSGD、lr0=0.00038、lrf=0.5、momentum=0.948、weight_decay=0.00027、cos_lr=True、mosaic=0、close_mosaic=0、degrees=0、translate=0.1、scale=0.5、fliplr=0、warmup_epochs=3。實際 warmup_bias_lr=0.1，不能漏記成 0。

這與融合 J0/J1/J2/J3 不同。融合的分組 LR、BN 固定及 stale 8 輪 LR×0.5，是雙任務共享 trunk 的措施；不能說獨立 Pose P3 已驗證這些技巧。本研究主要參考獨立 P1/P2/P3。

## 與本次方案的差異

| 項目 | 本次 | combine 獨立 Pose | 解讀 |
| --- | --- | --- | --- |
| 解凍 | freeze=10 後直接全部解凍 | head-only → neck/head → full | 可測試降低新任務早期對預訓練特徵的干擾 |
| MuSGD LR | 0.001 | 0.00038 | 本次高約 2.63 倍，較低 LR 值得做同起點比較 |
| LR 最終倍率 | Stage 4 為 0.05 | 0.5 | combine 排程較平緩，但較低 lr0；不能只看倍率 |
| nbs | 64 | 128 | 改變累積 batch 與 optimizer 更新次數，未必必然更好 |
| weight decay | 0.0005 | 0.00027 | 同時更改會混淆歸因，適合後續整套 recipe 比較 |
| Mosaic | 前兩階段 0.5 | 全程 0 | 可測試是否減少棒球場景拼接失真；目前只有相關性 |
| 解析度 | 640 後 960 | 全程 640 | 960 未必總是更好，要同一權重評估 640／960 |
| 模型 | 官方 YOLO26m Pose | Full35、MASF、修改 attention | 架構及初始化不同，不能把精度差全部歸因於 recipe |

本次 batch16／nbs64 在暖身後約累積 4 次；nbs128 則約 8 次。physical32／nbs128 約累積 4 次。暖身期 accumulation 會漸變。累積不會把 BatchNorm 的 physical batch 改大，也不與真實 batch128 逐值等價。現有 batch16／nbs64 的 weight decay 縮放為 16×4÷64=1，沒有 combine 曾遇到的 weight decay 翻倍問題。

## 成果比較與限制

以下均為同一 canonical formal val，但 combine 是 640／Bit-True，這次是 960／官方架構；僅供辨識方向，並非控制實驗。combine 指的是獨立 Pose P3，不是融合最終 J3。

| 指標 mAP50–95 | 本次 Stage 4 | combine 獨立 P3 |
| --- | --- | --- |
| Box overall | 62.43% | 63.10% |
| Pose overall | 90.51% | 91.22% |
| ball box | 51.36% | 51.07% |
| bat box | 73.50% | 75.12% |
| ball pose | 86.42% | 87.63% |
| bat pose | 94.60% | 94.81% |

差距不大，且本次 ball box 略高；不能說 combine 全面較好。這次 Stage 3 的 Pose mAP 90.85% 又高於 Stage 4，須保留 Pose 最佳候選而不是只看綜合平均。

本次 Stage 3 第 8 輪最佳、Stage 4 第 15 輪最佳，後期驗證下降；combine P3 第 68 輪最佳。這支持「不同 recipe 可能需要不同訓練長度」，不支持「本次加到 200 輪一定提高」。

## 建議按成本與可解釋性排序

1. **先做同起點、只改 LR 的 MuSGD 候選。**從目前 Stage 3 best.pt 開始，維持 960、batch16、nbs64、epochs100、patience30、lrf0.05、momentum0.9、weight_decay0.0005 與所有 augmentation，只將 lr0 從 0.001 改為 0.00038。比較現有 Stage 4 即可，不必重跑前三階段。候選 YAML 已保存，尚未排隊。不能從 Stage 4 best 繼續訓練再稱為同起點 LR 比較。
2. **有改善再分別測試平緩排程與 nbs。**同起點將 lrf 改0.5；再單獨比較 nbs128。不要第一次同時更改 LR、nbs、WD、augmentation 而宣稱找到了原因。若只求效果，也可測一個明確標記的整套 recipe 候選，但它不能解釋單項貢獻。
3. **如果要重頭改善適應流程，再跑官方架構的 combine 式三階段。**仍從官方 yolo26m-pose.pt 初始化，640、mosaic0、依 head-only（freeze23）→neck/head（freeze11）→full 的順序，最多17／22／100輪，patience10／12／20；建議 physical batch32、nbs128，避免照抄128造成OOM。MuSGD 0.00038、lrf0.5、momentum0.948、WD0.00027可作整套候選。保留目前 warmup_bias_lr=0 而非盲目照抄0.1，並明記這是改編而非完全複製。此方案需要完整新實驗，不修改現有權重。
4. **最後才考慮短 960 精修與更精確選模。**先證明640候選更好，再做較短高解析度精修；同時比較 ball／bat 個別 Box、Pose、recall，並補關鍵點像素／正規化距離誤差。高 Pose mAP 不等於已證明像素級精度。若多次選參，必須揭露 formal val 被反覆用於選模；不自行新建 split。

分組 backbone/head LR、固定 BN 等可列為後續研究，但對官方 Pose trainer 需要額外實作及驗證，不應直接複製融合 trainer。

## 本次交付與未解事項

已建立 `configs/candidates/stage4_musgd_lr00038.yaml`，通過官方設定解析並確認與現有 Stage 4 只有實驗名稱及 lr0 不同。没有 GPU 訓練或重驗。本次只使用主代理；未更改 yolo_combine。下一步是否提升只能由實驗確認，單 seed 小差距不能宣稱統計顯著。

困難與解法：combine 底層預設與公開 policy 有覆寫差異，透過實際 args.yaml 和 results.csv 交叉核對解決。未解事項：同解析度、同 backend、同初始化的跨配方實驗尚未執行。
