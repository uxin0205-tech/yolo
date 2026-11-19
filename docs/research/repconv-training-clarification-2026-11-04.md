# RepConv 實際訓練方式與原始實作核對

2026-11-04 最新確認：**簡化算子是部署／推論要求，不是 RepConv 訓練階段的強制要求。** 訓練可依原本 RepConv 流程使用常規算子；部署時才完成 folding 與 qSiLU、完整 PWL／reciprocal 等優化，並以最終部署版本驗收。此為需求澄清，尚未授權啟動新實作或訓練，仍等待使用者確認。

## 已確認的要求

- 訓練：按原本 RepConv 訓練流程處理，允許常規算子；不強制套用我們的簡化算子。
- 部署：完成 Rep folding 及已選硬體友善優化；qSiLU、完整 PWL exp／reciprocal 等要求在最終推論模型驗收。
- 訓練版本與部署版本各自保存設定與指標，最終以優化後完整 AP 決定是否採用。
- 「訓練可以不簡化」不等於所有訓練算子都已指定必須還原，也不等於已決定 parent、範圍、輪數或啟動時間。先前單組 5 輪提案不因本次澄清自動生效。

之前把部署要求延伸成訓練必須 qSiLU／D1，是代理對使用者要求的誤讀。以下內容記錄已完成實驗的實際做法，不代表未來必須照同樣限制訓練。

## 已完成的 Stage8 實際怎麼做

起點是同一份 Stage7 qSiLU E8，Rep17、Rep20、Rep17＋20 各自獨立訓練 5 輪。

```text
訓練：原 Conv3×3＋BN ─┐
                      ├→ 相加 → qSiLU
      新 Conv1×1＋BN ─┘

驗證／部署副本：Folded Conv3×3+bias → 相同 qSiLU
```

- 實作使用本輪的 `QSiLURepConv` 包裝類別，沒有直接沿用 Ultralytics 的 `RepConv` 類別；兩條線性分支與相加後 activation 的結構相同。
- 原 3×3 Conv／BN 繼承 parent；新增 1×1 Conv，將該分支 BN 的 gamma／beta 初始化為零，使起點輸出保持 parent。gamma 是可學習值，沒有永久鎖成零。
- 訓練 live 模型與 EMA 一直保留兩個分支。三組 smoke 都確認新分支的 1×1 Conv／BN gamma 與兩個 heads 確實更新。
- 只更新所選 Rep 層及 Detect／Pose heads；其他 backbone／neck、Attention、scale／bias 固定。
- 所有 BN running mean／variance 固定；可訓 Rep／heads 內 BN affine gamma／beta 可更新。固定 running statistics 不等於整個 BN 沒有學習參數。
- AdamW，Rep LR 1e-5、兩 heads LR 5e-6，warmup 1 輪後 cosine，fresh EMA；每組 5 輪，沒有實際觸發延長。
- Pose loss weight 分別校準為 Rep17 .195、Rep20 .5、Rep17＋20 .215；不是三組共用 optimize 的 .25。
- 訓練前向保留 qSiLU、B2／S1／D1。D1 反向使用 Float PWL proxy，不是先拿折疊後單 Conv 或原生 Softmax 模型訓練。
- 每輪驗證建立 EMA 的 eval 副本，再 fold 成單 Conv，評估完整 AP；不修改繼續訓練的 live／EMA。最佳三組均為 E1，完整 5 輪歷史仍保存。

來源：`yolo_achitechure/stage8/{rep,model,training,evaluate}.py`、各 `train-*/config.json`／`result.json`、`calibrate-*/result.json` 與 `smoke-*/result.json`。

## 與兩種「原本」參考的關係

| 項目 | 本輪 | optimize 的 Rep 階段 | 本地 Ultralytics RepConv 類別 |
|---|---|---|---|
| 訓練結構 | 3×3+BN、1×1+BN，相加後 activation | 相同 | 相同；identity 僅在設定開啟且 shape 允許時存在 |
| 使用類別 | 自訂 QSiLURepConv | 原生 RepConv，加初始化／部署包裝 | 原生 RepConv |
| Activation | qSiLU | qSiLU | 預設 SiLU，但可直接傳入 qSiLU module |
| 初始化 | 已訓 3×3＋新分支 gamma=0 | 相同策略 | factory 本身不規定這種預訓初始化 |
| 可訓範圍 | Rep＋兩 heads | Rep＋兩 heads | 模組本身不規定全模型凍結範圍 |
| BN running | 全固定 | 全固定 | 一般 train 模式會更新，是否固定由外部訓練程式決定 |
| 輪數／LR | 5 輪，1e-5／5e-6 | 同一初始預算與 LR | 模組不規定 optimizer、輪數或 LR |
| Pose loss weight | .195／.5／.215 | 全部 .25 | 模組不規定任務權重 |
| Attention | D1 完整 PWL＋整數 reciprocal | PWL，分母仍浮點參考 | RepConv 本身不處理 Attention Softmax |
| 推論 | 同位置單 Conv3×3+bias | 折回原 Conv 容器，以 identity BN 表示 bias | 支援分支融合 |

第 17／20 層 stride=2，因此不符合 identity 分支所需 stride=1；沒有 identity 並不是漏掉可用的同形分支。原生 RepConv 的 `act` 接受 `nn.Module`，所以採用原始 RepConv 分支／folding 實作與保留 qSiLU 可以同時成立，不必因此更換全模型 Softmax。

optimize 與本輪的 parent 不同（optimize 為 B E5、Pose-only MASF；本輪為 Stage7 qSiLU E8、沒有 MASF），seed、任務權重與選最佳輪政策也不同。不能聲稱原封不動重現 optimize 整個實驗。

## 如何解讀已完成結果

本輪已做到 RepConv 的多分支訓練與部署折疊，但訓練配方屬於 **從已訓模型出發的局部、低 LR、5 輪適應**。這與「使用完整原始模型訓練流程」不是同一個主張，也不能把 BN 凍結、訓練範圍、輪數等選擇說成 RepConv 演算法必須如此。

折疊維持同一份已訓權重的函數（容許浮點誤差），不是把權重恢復成訓練前；因此 person AP 變化不能單憑架構相同就排除。現有證據也不足以把下降單獨歸因於 qSiLU、BN 凍結或短訓練。

## 接下來的邊界

先由使用者覆核以上實際做法；若需要調整，再明確對齊原生 RepConv 實作、初始化、BN、可訓範圍、LR／輪數及驗收規則。改用原生類別本身不保證 AP 改善。需求已確認為部署端必須優化；具體實作與實驗開始仍待使用者確認。原生訓練算子可以使用，不再受舊的強制 qSiLU／D1 限制。現有來源鎖維持已完成實驗版本，Stage9 不啟動。
