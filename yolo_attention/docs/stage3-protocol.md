# 第 3 階段正式 protocol

完成狀態：2026-10-19 11:57，選定 D1，官方 AP 0.514189349；交付 [result/stage3](../result/stage3/README.md)。本階段無需恢復訓練；KD 可行性與教師限制另見 [恢復判斷](stage3-recovery-kd.md)，本次未啟動 KD。以下保留實驗開始前固定的 protocol。

2026-10-19 使用者已指定開始。實作及新結果由 `yolo_attention` 維護，parent 為 `../yolo_binaryqk/result/stage2/epoch-05.pt`，SHA `9cacbd802f089a6e0aca87c2e0d1743d74458e7c030029be3076dd3090b48dea`，官方 AP50–95 0.5141516577。保持 B2 sign＋Hadamard、S1 的 16 個 Q10 scales、signed16／1024 bias、V／P·V／projection、SiLU 及資料／evaluator。

實作前已唯讀核對 optimize、Attention 與 final；完整來源、SHA、差異與事前門檻見 [reference-review.json](../artifacts/stage3/reference-review.json)。[指定論文](https://www.mdpi.com/2079-9292/14/12/2337) 的 reciprocal LUT／乘法 normalization 作概念依據；原文保留共用 divider，Newton 用於 LayerNorm 平方根。本輪的 reciprocal PWL、reciprocal Newton 及整數格式均為專案延伸。

## 步驟與候選

1. CPU 單元與數值 reference：合法／非法分母、二次冪／查表邊界、最大 token、tail、exp(0)=1、完整中間精度及捨入。先通過才開 GPU。
2. 真實資料診斷：固定完整 COCO val 順序的前 16 張，batch 2，只作診斷，不改正式資料。兩處各四個 heads 記錄 score 範圍／tail mass、exp、reciprocal、輸出量化、row-sum、P 與 P·V 誤差。
3. N1～N9：Float PWL[-8,0]/16、Float PWL[-10,0]/20、BitTrue PWL[-10,0]/20、score LUT（step 0.125／floor -8）、PoT weights（floor -8）、Normalized HardSigmoid、Multimax Top-1／3／5。沿用已核對的數學定義，先固定共同 exact 分母；Multimax 保持其原生 stick-breaking 定義並單獨標示剩餘 sigmoid，不將它改成另一種方法。
4. D1～D3：固定同一 BitTrue PWL exp[-10,0]/20，僅替換分母 reciprocal。D1 為 256-entry midpoint LUT，D2 為 16 段端點插值，D3 為 64-entry midpoint LUT 加一次 Newton。
5. 真實資料中最差 site/head 的 PV relative L2 不超過 0.5 者，做完整 COCO val5000；被排除者保存理由，不寫假 AP=0。完整驗證先保存候選 checkpoint 再嚴格重載，評估實際部署圖，禁止隱式 fusion。
6. 依精度／成本選型，不自動加訓。必要恢復訓練至多少數候選，須先查核對應 optimize 配方、固定預算、部署前向與 backward proxy，再啟動；不新增 control。

沒有 ReLU、exact Softmax 正式對照組或 [-8,0] BitTrue 候選。exact 僅用於成對數值診斷，既有 S1 官方 AP 作任務參考。完整任務評估沿用 E2E FP32、conf 0.001／IoU 0.7／max_det 300、原有完整 val5000、官方 COCO API 及 internal 八項相關 AP，不重切或抽換資料。

## 整數契約

Stage 2 score 是 signed18／1024；row-max subtraction 的最壞差值須 signed19。reference 用 int64 容器保留中間精度，轉 Q8.8 時 nearest ties 向正無窮，然後 clamp[-10,0]。PWL 21 個 UQ1.15 endpoints 合計 42 bytes，interpolation 完整乘積後 floor >>7，最大 endpoint 為 32768，能表示 1。

權重整數相加，N=400 時分母上限 13107200，需 uint24；最大 N=1024 時上限 33554432，需 uint26。leading-one 以整數比較／移位得到 exponent，mantissa 截至 Q1.16。倒數回傳 Q1.16 mantissa code 與 exponent，不先把很小的絕對 reciprocal 塞回 Q1.16 而造成下溢。

D2 的 slope 為有號；D3 使用 `r1=r0*(2-m*r0)`，每個完整乘積後依明定邊界捨入。probability 使用 UQ1.24，固定乘 `2^-24` 接既有浮點 P·V。沒有以浮點除法再次修正 row-sum。reciprocal 對分母 0、負數及溢位直接失敗；整數部署路徑測試禁止動態 div／log／exp／frexp／ldexp。

事前 reciprocal 相對誤差門檻：D1 0.0021、D2 0.0011、D3 0.00012。輸出量化每元素誤差上限 2^-25，1024 tokens 的累加量化誤差上限 1024×2^-25；總 row-sum 仍須加上 reciprocal 近似誤差，不能只報量化誤差。

## 選型與交付

完整硬體友善路徑須與 parent 及最佳比較候選的官方 AP 差距各不超過 0.001。D1～D3 內先選 AP 最高，完全同分才選較小 reciprocal ROM。這個容差參考 final 的工程 gate，事前固定，非統計顯著性門檻。

若完整 PWL 未通過，報告保留全部比較，先分析其他 normalization 的完整分母／剩餘運算，或對少數合格候選適應；不能直接把浮點 reference 宣稱成完整交付。通過時 handoff 嵌入來源、S1 完整契約、實際 normalization／reciprocal、整數常數、位寬、rounding、activation 與 next-stage=false。

目前無訓練預算消耗；最多 12 個 normalization 候選做一次完整驗證，實際由真實資料 gate 減少。GPU 每 600 秒監測、每 5 秒檢查完成／退出；程式無進度超過 1800 秒只停止自己的 child。完成第 3 階段後停止，不自動做 MASF。

## 單一入口

在 `yolo_attention` 執行；設定 `PYTHONPATH=src:../yolo_binaryqk`，Python 使用 `../.venv/bin/python`。

```bash
python -m yolo_attention.cli stage3 reference
python -m yolo_attention.cli stage3 init --output artifacts/stage3/<新run>
python -m yolo_attention.cli stage3 run --output artifacts/stage3/<run>             # dry run
python -m yolo_attention.cli stage3 run --output artifacts/stage3/<run> --execute
python -m yolo_attention.cli stage3 status --output artifacts/stage3/<run>
python -m yolo_attention.cli stage3 plot --output artifacts/stage3/<run>
```

仍由既有 QueueExecutor／ResearchQueueBackend 執行，不走歷史 59-job 矩陣。stage3 的 VALIDATE job 可輸出完整 AP 或明確的診斷排除；metrics 的 formal 欄位才決定是否完整 AP，避免因 job 名稱誤判。中斷／失敗後不可假填成功；保留實際 queue 與錯誤，再診斷。`STOP` 在診斷 batch／評估 heartbeat 邊界生效。

## 舊能力與清理

final 的 PWL 數學與現有 Attention 主體一致，必要的自包含 checkpoint、materialize-before-evaluate、嚴格 activation／normalization 檢查已接入本輪；訓練與監測沿用已驗證 engine，沒有複製另一套 Attention class。final 內 59 份重複 source 已移除，改為相對連結；核心演算法只有 `yolo_attention` 一份實體。optimize 仍用到的外部路徑與不同歷史訓練 API、唯一權重保留，未整套封存，見 [清理與相容驗證](../artifacts/stage3/cleanup-core-duplicates.json)。

既有 IntegerLUTSoftmax 為 score LUT＋U8 probability，仍有整數 division／可選 row correction；不納入本輪正式矩陣，不能當成此次完整 reciprocal。BDCN V3 fixed／R1 的歷史 internal AP 為 0.506566／0.506562，近似二次冪分母舊診斷 AP 僅 0.004872，參考 [既有比較](../reports/comparison.csv)；這些與本輪官方 AP／新 parent 不可直接混比，也不重啟舊訓練矩陣。

QKV、P·V、projection、Conv、SiLU 仍是浮點。運算量／ROM 為解析成本，訓練或 GPU reference 的耗時不等同硬體綜合／板端效益。


## 2026-10-19：已授權 KD 延伸

零加訓選型與原 D1 結果保持原樣；使用者另授權本 Stage 追加單一最多五輪的 KD 恢復。事前教師／學生 SHA、方法差異、train-only 校準、八項 AP 門檻與回退策略見 [KD 固定配方](stage3-recovery-kd.md)。新 source.json 獨立記錄本輪程式，不回寫原正式選型的歷史 source snapshot。
