# 第 2 階段：固定 Q10／二次冪 scale 與分解式 bias

使用者於 2026-10-19 指定開始。實作前已核對 optimize 的程式、實際設定、來源血緣及結果，見 [來源核對與 SHA](../artifacts/stage2/reference-review.json)。Q10 參照 `StaticDyadicScore`，bias 參照 `DyadicRelativeBias`；訓練參照 `post_masf_hardware_v1` 及其 EMA、loss、cosine 上游。歷史 scale_bias E5 internal AP 為 0.505248，原工程 gate 未通過，不作本輪精度保證。

## Parent 與交接

兩組均直接從 [第 1 階段 result](../result/stage1/README.md) 的 `epoch-00.pt` 開始，SHA `db8b11ddec3c3263e9a79d815645f320dc74cb646f2ea638b8ded4c2efe268f9`；官方 COCO AP50–95 為 0.509889。B2 sign＋Hadamard、兩處 Attention、每處四個 heads、native Softmax、SiLU、V／P·V／projection 均繼承。S2 不繼承 S1 的訓練結果，正式 run 不繼承 smoke 權重。

依最新使用者指示，階段選定成果實體放入 `result/stageN/`，下一階段在自己的目標資料夾工作；共用算子仍由 `yolo_attention` 維護。舊 run 的唯一路徑相容連結不是第二套權重。此輪 S1／S2 輸出位於 `artifacts/stage2/`；切換第 3 階段前再將選定候選整理為 `result/stage2/`，由 `yolo_attention` 接續。

## 候選及整數契約

| 項目 | S1 | S2 |
| --- | --- | --- |
| scale | `m/1024`，`m=1…1024` | `2^k`，`k=-10…0` |
| 訓練 latent | 同一 B2 的 16 個浮點係數 | 同一 B2 的 16 個浮點係數 |
| 推論運算 | 整數乘法 | `count << (k+10)` |
| bias | 水平／垂直零表開始，signed16／1024 | 相同 |

S2 是本專案延伸，沿用 bounded amplitude STE；不是宣稱 optimize 已驗證的候選。兩組 scale／bias 在訓練可更新，推論快取固定量化常數。bias 為 `by[yi-yj]+bx[xi-xj]`，不因增加 basis 重複表。每個 site 共 `4×63×2` entries，兩處合計 1008 entries／2016 bytes。所有 round 採 nearest ties-to-even；bias 飽和範圍為 `[-32768,32767]` codes。

binary dot 範圍 `[-32,32]`，兩 basis score 加 bias 的共同 Q10 累加需 signed18，reference 使用 int32，最後乘固定 `2^-10` 交給既有 Softmax。訓練前向使用同一整數結果，反向使用浮點代理。明確提供 H／W，各不超過 32、N 不超過 1024，不猜測必為正方形；目前正式輸入 640×640，兩處各 20×20。

每張 640×640 有 2.56M 次 scale 計算，S1 是常數乘法、S2 是 shift。QKV、P·V、projection、native Softmax 與其他 Conv 仍為浮點；binary reference 尚未打包，未做 RTL／HLS／板端量測，不宣稱整網定點或實測加速。PWL、reciprocal 與 qSiLU 由後續指定階段接續。

## 固定預算與選型

每組 5 epochs，同 seed 1，COCO train2017 全部 118287 張；epoch 0 及每輪評估 val2017 全部 5000 張，沿用官方 COCO API 與 internal evaluator，回報整體、person、ball、bat 共八項官方 AP。BBAT5 v1 不變。

可訓練兩處 Attention 與 Detect head；所有 BN running statistics 固定，DFL 固定。AdamW betas `(0.948,0.999)`、weight decay `0.00027`；scale、bias、BN affine 不 decay。分組 LR：Attention `1e-5`、head `5e-6`、scale `2e-4`、bias `1e-5`。前 1 epoch warmup 由 0.1 到 1，再 cosine 到 0.5。

physical batch 16、logical batch 256；native E2E batch-sum loss 乘 `64/actual_macro_images`，尾批保留，每 epoch 463 optimizer steps。FP32 參數配 AMP FP16、gradient clip 10、初始 scaler 1024，overflow 最多重算 16 次，不跳過資料。EMA `.9999/tau2000` 僅更新可訓練參數。記錄兩處 Q/K、scale、bias 共十項梯度與實際更新；檢查固定權重／BN 沒有漂移，resume 保留 live／EMA／optimizer／scaler。

best 包含 epoch 0，以完整官方 AP50–95 選擇；S1／S2 完全相同 AP 才優先 S2 shift 成本。不加訓 baseline/control。初始 AP 低於 B2 的 90%，或後續較自身 epoch 0 下降超過 0.05，停止診斷；正常兩組均跑滿五輪，不因先看到結果改預算。選型後停止，不自動開始第 3 階段。

## 執行與監測

從 repository 根層執行 `PYTHONPATH=yolo_binaryqk:yolo_attention/src .venv/bin/python -m binaryqk.scale --help`。正式隊列以 `queue --output <stage2新目錄>` 啟動；`--smoke` 僅驗證每組 512 張／兩個 macro steps，不是正式 AP。

背景 supervisor 每 600 秒寫入 GPU、進度與 heartbeat；每 5 秒檢查子程序是否結束。無進度超過 1800 秒只終止自己的 child；GPU 被其他運算程序占用時等待，不終止他人工作。`STOP` 檔案在 batch 邊界停止；`--continue-queue` 必須通過來源 SHA、recipe、完成結果及 resume 驗證。正式啟動後不修改實作或來源核對檔。

長訓穩定後退出 working，背景依規則保存監測與結果。兩組完成後自動產生比較 JSON／CSV、曲線、常數、報告及 handoff；人員檢視後才指定下一階段。曲線可由 `python -m binaryqk.scale_report --plot <run目錄>` 僅讀小型 JSON 重畫。
