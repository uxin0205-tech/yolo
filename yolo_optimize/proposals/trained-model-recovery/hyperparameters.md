# 已訓練權重 recovery：超參數規格

日期：2026-09-08。以下分開列「歷史文字證據」和「新建議」，不把建議值說成已驗證最佳值。現在未啟動任何模型／GPU 工作；正式設定仍需先固定使用者實際 checkpoint。

> 最新前置：使用者指定從 `final/full35` 接續但不保證 train 好。先依[final接續清單](<final-readiness.md>)同口徑比較 J3 `best_joint`／J3 `best_pose`／J2 `best_joint` 後選 `PSEL`，再做基準確認／必要原生 loss 修復；不能直接把下面 HOG 小恢復當下一個必跑。方向1唯一當前順序見[方向1 master plan](<../integrated-roadmap/direction1-master-plan.md>)，optimizer 配套見[optimizer policy](<../integrated-roadmap/optimizer-policy.md>)。`training_ready=false`；本頁不是可執行 trainer 設定。正式 baseline 維持 AdamW guard；MuSGD 有 builder 支援但只作另案 paired challenger，中途切換的 state／LR 校準尚待驗證。「BN 較大」只指模型／參數改動較大，不是 batch 或 BatchNorm。

## 1. 歷史 J3 的可核對設定

來源根目錄 `/home/uxin/yolo/yolo_combine/final/full35/`：

- `RELEASE_STATUS.json`：selector=`j3_best_joint`，best global epoch=58、J3 local zero-based epoch=5，stage-complete last global epoch=63。
- `README.md`：J3 實際 Detect physical microbatch=`32×4`，即一個 logical batch=128。`configs/experiment-joint.yaml` 雖寫 microbatch=64，但 stage/J3 欄位也有舊內容，不能直接當最後有效配置。
- YAML 共同設定：Detect logical128、Pose16；每 macro 2 個 Detect logical batches＋1 個 Pose batch；reference batch64，task weights Detect1/Pose0.25；AdamW betas=(0.948,0.999)、weight decay=0.00027、warmup1、起始因子0.1、cosine末因子0.5、gradient clip10、shared BN statistics frozen。
- **本次更正：** 上列warmup1只是通用YAML／J0–J2設定；final的`stage_policy.py`與`analysis/FINAL_ANALYSIS.md`支持J3實際warmup=3。新run若採1epoch須標成新提案，不得稱為沿用J3。J3/microbatch的CLI覆寫也需納入resolvedconfig。
- `outputs/training/logs/macro.csv` 本次以標準庫串流讀取；每個下列 `.decay` LR 欄位都有 5,093 筆數值記錄、step 23,819–28,911。最大值如下；它們是**該日誌區間觀察到的峰值**，不是從 checkpoint 讀出的當前 LR，也不是證明每一步所有參數都有 gradient。

| Role | J3 日誌觀察到的最高 LR |
| --- | ---: |
| backbone | 3.8e-6 |
| neck | 1.9e-5 |
| Detect head | 5e-5 |
| Pose head | 5e-5 |
| MASF | 3.8e-5 |
| attention | 5e-7 |

以下以歷史峰值約一半作Neck／heads的保守候選，但不再假定現有表徵已充分訓練。5–10epochs只是診斷／比較窗口，不能由LR比例推導精度改善，也不能因窗口結束就宣布基準已修好。

## 1.1 先做基準修復：只按證據選一條

| 候選 | 窗口 | 峰值LR／範圍 | warmup | 前提 |
| --- | --- | --- | --- | --- |
| B-HEAD5 | 5ep | 證據指向的task head 5e-5；其餘freeze | 1ep | 例如Pose適應不足且shared feature可保留；不默認從best_detect啟動 |
| B-NATIVE10 | 10ep | Neck1e-5；Detect/Pose heads各2.5e-5 | 3ep | 先保持原圖與loss，條件式小範圍joint重新協調 |
| B-LATE10 | 另案10ep | late backbone layer9+ 1e-6；Neck1e-5；heads2.5e-5 | 3ep | 只調Neck/heads不足且有證據，才額外放行late backbone |

共同沿用imgsz640、AdamW betas(0.948,0.999)、eps1e-8（提案）、weight decay2.7e-4、Detect logical128=32×4／Pose16、每macro 2 Detect＋1 Pose、reference64、taskweights1/0.25、clip10、shared BN statistics freeze、相同augmentation／EMA策略。B分支不加HOG、KD、RepConv、不搬MASF、不改Q/K或固定係數；未列可訓練的參數保持freeze。0/5/10或0/5完整驗證，修復未驗收就不進新優化；是否延長或改scope另記決策，不自動把所有分支跑一遍。

上述是新run的warm-start提案；真正相同中斷run的exact resume需保留其原完整狀態，不套fresh optimizer。現有J3有正常early-stop證據，不應默認是待續跑的中斷任務。

## 2. 基準驗收後的新優化：共同設定

| 欄位 | 首版提案 | 理由／限制 |
| --- | --- | --- |
| 初始權重 | 同一已驗收 P-READY 的相同 EMA/live | P0只有通過品質驗收才可直接成為P-READY |
| optimizer state | 兩臂一致 fresh AdamW | warm-start，不聲稱 exact resume |
| image size | 640 | 與目前推論／訓練入口對齊；先不掃解析度 |
| Detect logical batch | 128，physical32×4 | 每 macro 2 個 logical Detect batches，不能稱單個 physical batch256 |
| Pose batch | 16，每 macro 1 個 | 保留 joint 任務比例 |
| reference batch | 64 | 保留原 batch/loss normalization，勿再重複縮放 task weights |
| task weights | Detect=1.0、Pose=0.25 | 首輪不改任務權重，也不設成所有 loss 相等 |
| AdamW | betas=(0.948,0.999)、eps=1e-8 | eps 是本輪明定值，需與實際 optimizer 設定核對 |
| weight decay | 2.7e-4 | 沿既有 eligible weight 分組；bias/BN 不新增 decay |
| gradient clipping | global norm 10，AMP unscale 後 | 同 control；不逐 task 任意更改 clipping 規則 |
| scheduler | cosine，1 epoch warmup，start factor0.1，final factor0.5 | 表中的 LR 是 warmup 後峰值 |
| AMP／EMA | 延續 P0 實際策略，兩臂相同 | 複驗數值與 resume 行為；不默默換 EMA來源 |
| BN | shared running statistics frozen；head BN 沿既有策略 | 兩臂相同；新 RepConv BN 另測 identity/fusion |
| augmentation | 鎖定 parent 的實際 resolved recipe | 已有 mosaic=0、Detect flip=0.5／Pose flip=0 的記錄須與實際來源核對；mixup 等未完整確認欄位不杜撰值 |
| seed | 0；只有有效候選再做 paired1/2 | 不先做大 grid |
| checkpoint 選擇 | 固定末輪為配對主結果，best 另表 | 不各挑不同更新量宣稱 matched comparison |
| 驗證週期 | 10ep：0/5/10；5ep：0/5 | 完整既有 val；無新增抽樣／split |
| early stopping | 不做常態 patience 搜尋；只保留安全中止 | NaN/Inf 或中期严重退化記為失敗，不自動加碼 |

資料保持原 COCO80／canonical bbat5-v1，person-only 不啟用。若跑之前發現實際 microbatch／AMP／augmentation 與文字來源不一致，先更新 manifest；所有配對臂共同修正，不只改其中一臂。

## 3. 每個方向的可訓練範圍與 LR

| 分支／arm | 長度 | 可訓練參數與峰值 LR | 明確凍結／限制 |
| --- | ---: | --- | --- |
| W-CTRL10 | 10ep | Neck 1e-5；Detect/Pose heads 各2.5e-5 | backbone、attention、現有 MASF、QK固定係數 |
| W-HOG10 | 10ep | 同 control；新 HOG head 3e-4 | 同上；不搬 MASF、不改 QK |
| BR-KEEP5／BR-OFF5 | 各5ep | Neck 1e-5；Detect/Pose heads 各2.5e-5 | backbone、attention、MASF參數；僅候選使 shared殘差有效gate=0 |
| M-CTRL5 | 5ep | Detect P3 box/class predictors 2.5e-5 | P3 producer、其餘 Neck／heads、Pose 全凍結 |
| M-DET5 | 5ep | 同 control；新位置 MASF context 5e-5、residual gate 1e-4 | gate初始有效值0；實際白名單需覆蓋 o2m/o2o |
| R-CTRL5／R-175 | 各5ep | Neck（候選含新Rep branches）1e-5；兩 heads 各2.5e-5 | backbone、attention、MASF凍結；只換layer17 |
| Q-CHALLENGER（尚不可直接執行） | 暫定10ep | Neck 1e-5；heads2.5e-5；若合法解凍 Q/K 則5e-7 | 先完成独立challenger契約及梯度測試；其餘attention、backbone、MASF與固定係數先凍結 |

M 分支只有在 bridge 通過後才建立新 Detect-only MASF；若沒有 bridge，不把現有 shared-MASF checkpoint 當作可無損換位的來源。Q 表格是預備值，不是允許在正式 baseline 開 `qk_ste=true`；現有程式會拒絕該設定。蒸餾超參數不在 teacher／live梯度契約尚未成立時硬填進訓練檔。

## 4. HOG loss 的權重不是直接設 μ=0.05

採 `L = L_native_joint + μ(t) L_HOG`。不同 loss 的數字尺度不相同，所以 0.05 指的是**加權後輔助梯度相對主梯度的目標比例**，不是 μ 本身。

在 raw P3 live activation `z` 上，先於指定的既有 training 資料順序估計：

```text
g_main = RMS(∂L_native_joint / ∂z)
g_hog  = RMS(∂L_HOG / ∂z)
μ0 = 0.05 × median(g_main) / (median(g_hog) + ε)
```

需按實際 macro 的 Detect/Pose 權重與 reference normalization 計算，不把不同 batch 的 tensor 直接相加。僅在既有 training trace 上量測，不從 validation 調 μ。若有效 HOG cell 或任一梯度中位數太小／非有限，停用候選並調查，不利用極大的 μ 硬補。此量測會需要 backward，**本次未執行**。

首版 schedule：epoch1 μ=0 並量測；epoch2 線性升至 μ0；epoch3–8 固定 μ0；epoch9–10 μ=0，交還原生任務收斂。監測實際梯度比，0.02–0.10 是觀察安全帶，不是自動掃一堆 loss weight；持續超出時暫停檢查，不依 validation 表現動態追數字。若只有合成資料的測試就通過，不代表正式 training trace 的梯度比例也已驗證。

目標9個方向bins、cell8、box mask與無能量cell忽略规则見[主計畫](<plan.md>)。本輪的設計與舊從J0開始的HOG實驗不同，須用新ID與新manifest。

## 5. 視覺回放與正式 AP 的設定不能混用

| 設定 | 視覺回放 | 正式 validation |
| --- | --- | --- |
| imgsz | 先忠實重現使用者設定；候選預設640 | 固定原 evaluator 的尺寸與預處理 |
| conf | 原設定優先；程式預設0.25 | 不用0.25任意截斷AP；0.001僅為待核對提案 |
| iou | 原設定優先；程式預設0.70 | 與 NMS／end2end 分支共同鎖定 |
| max_det | 程式預設300，需核對實際 | 記錄export cap與API maxDets；不能混稱同一值 |
| task／class filter | both；預設不加person-onlyfilter | COCO80與BBAT5原任務／類別映射 |
| backend／EMA | 同部署狀態 | 同一主表必須相同；其他backend獨立列欄 |
| 動態閾值／平滑 | 首輪關閉新機制，重現既有策略 | 不用視覺平滑後的輸出冒充原模型AP |

若最後證據顯示問題主要在閾值或時序呈現，另提出對應最小改動；不把 HOG／MASF／QK 訓練當成必經流程。
