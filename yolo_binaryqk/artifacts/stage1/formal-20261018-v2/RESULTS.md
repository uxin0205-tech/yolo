# 第 1 階段完成，等待使用者檢視

正式隊列於 2026-10-18 12:46～23:10（Asia/Taipei）執行，共約 10 小時 24 分鐘。三組各 10 epochs 初篩，依預定規則選 B2 接全模型恢復；恢復實際 5 epochs 後因 patience 停止，交付仍為初篩最佳權重。未啟動第 2 階段。

| 候選 | 階段 | Best epoch | 官方 AP50–95 | 球 AP50–95 | 棒 AP50–95 |
| --- | --- | ---: | ---: | ---: | ---: |
| B1 | 共同初篩 | 10 | 0.503966 | 0.512891 | 0.475569 |
| B2 | 共同初篩 | 10 | 0.509889 | 0.514560 | 0.487651 |
| B3 | 共同初篩 | 8 | 0.506355 | 0.514636 | 0.480618 |
| B2 | 選定分支恢復 | 0 | 0.509889 | 0.514560 | 0.487651 |

採用理由：依同預算初篩之官方整體 AP50–95 選分支；同分優先較少 bases。
未採用候選只有初篩結果，不能宣稱其完整恢復後仍較差。恢復保留 epoch 0 作回退。
完整八項 AP、參數契約、檔案大小、時間、顯存、權重 SHA 見 comparison.json 與 handoff.json。
scale slots 為 16／16／24；binary score 成本為兩處 heads×bases×N²×D；residual／Hadamard 成本見 protocol。
圖表由 comparison.json 重畫，不需 checkpoint 或 GPU。仍是單 seed／軟體 reference，沒有 RTL／HLS 或板端加速量測。

## 選型與精度差距

B2 的 sign＋Hadamard 以官方整體 AP50–95 勝出，比 B1 高 0.005922、比 B3 高 0.003534。B1 同樣為兩分支但精度較低；B3 有三分支、24 個 scale slots，增加 residual 形成與 score 分支成本，初篩整體精度仍未超過 B2。B3 的球 AP50–95 比 B2 高約 0.000076，這不改變事前固定的整體 AP 選型規則，也不足以在單 seed 下推論穩定優勢。

| 模型 | 官方 COCO AP50–95 | internal AP50–95 |
| --- | ---: | ---: |
| 原始 parent，完整重驗、未加訓 | 0.524686 | 0.518002 |
| B1 初篩最佳 | 0.503966 | 0.497746 |
| B2 初篩最佳／本階段交付 | 0.509889 | 0.503328 |
| B3 初篩最佳 | 0.506355 | 0.499704 |

B2 比原始 parent 的官方 AP 低 **1.48 個百分點**，沒有完全恢復原始精度。optimize A0 歷史 internal AP 為 0.506739，本輪 B2 internal 為 0.503328，仍低約 0.34 個百分點；兩者 scale、訓練方法及血緣不同，歷史結果不是本輪控制組。不能拿 B2 官方 0.509889 與舊 internal 指標直接比較並宣稱追平。

## 全模型恢復的負結果

| 恢復 epoch | 官方 AP50–95 |
| --- | ---: |
| 0：接入 B2 初篩最佳 | 0.509889 |
| 1 | 0.501803 |
| 2 | 0.492689 |
| 3 | 0.484558 |
| 4 | 0.489609 |
| 5 | 0.492613 |

依事前設定，最佳 epoch 包含 0；連續 5 epochs 未改善即停止。後兩輪雖回升，仍未超過初篩最佳，因此不採用恢復後的權重。只能結論「本配方在既定停止規則下沒有改善」，不能推論延長訓練一定無效或一定能追回；本輪不事後修改 patience 或加跑配方。

## 交付、成本與驗證

交付 [recovery/epoch-00.pt](recovery/epoch-00.pt) 的模型 state 與 [B2/epoch-10.pt](B2/epoch-10.pt) 逐 tensor 完全相同。前者記錄初篩→恢復的 immediate parent 血緣，所以檔案 SHA 不同；不是一份精度更高的新模型。正式路徑、SHA、recipe 與必要 resume 見 [handoff.json](handoff.json)。

交付權重的八項官方 AP：

| 範圍 | AP50 | AP50–95 |
| --- | ---: | ---: |
| COCO 整體 | 0.683027 | 0.509889 |
| person | 0.862384 | 0.639523 |
| sports ball | 0.718777 | 0.514560 |
| baseball bat | 0.726172 | 0.487651 |

模型共 **21,896,248 parameters**，交付檔案 **88,062,827 bytes（約 84 MiB）**。B2 初篩實測純訓練約 2.62 小時、GPU peak allocated 約 3.89 GiB；恢復純訓練約 1.76 小時、peak allocated 約 9.07 GiB。這些是訓練量測，不能代表部署推論延遲或硬體加速。完整數據見 [comparison.csv](comparison.csv)，曲線見 [curves.svg](curves.svg)。

第 1 階段保持兩個 sites、各四 heads、B2 共 16 個固定 FP32 scale slots、無 bias、一次原生 Softmax 與 P·V；QKV、P·V、projection、exp／除法與 activation 仍保留本階段的浮點／原生契約。Q10／二次冪 scale 與分解式 bias 待第 2 階段，PWL／reciprocal 待第 3 階段，qSiLU 待第 7 階段。本次不宣稱完整硬體友善主線已完成。

本輪 CPU 交付核對見 [handoff-verification.json](handoff-verification.json)：四份最佳權重 SHA、strict reload、finite state、參數與 manifest；39 次完整 val5000 評估紀錄、35 個完整 train118287 epochs、CSV／JSON 與選型一致；17 份研究程式及 11 份參考來源 SHA 不變；僅憑小型 JSON 成功重畫圖表。先前 44 項 CPU 測試及四項 GPU 短測見 [驗證摘要](../verification-v2.json)。本次收尾沒有重跑 GPU 評估或訓練。

## 邊界與待辦

使用者追問後補做 [B2 運算量核對](COST.md)：在相同 640 × 640 未 fuse 前向下，整網浮點 MAC 小計由 37.5056384 GMAC 降為 37.4646784 GMAC，減少 **0.109210%**。新增兩分支二值運算與 Hadamard／scale／合併成本另列；沒有實測推論加速結論。

第 1 階段已停止，B2 為下一階段建議 parent，尚待使用者檢視與指定下一階段。單 seed、只有 winner 接全模型恢復、停止規則及與歷史方法的差異均限制結論；尚無 RTL／HLS／板端量測。

錯誤 v1 已直接刪除，沒有封存；本次保留有效候選的最佳權重、必要 resume 與負結果資料，沒有額外刪除。`yolo_optimize`、BBAT5 v1、Pose 與其他未指定階段均未修改。完整收尾紀錄見 [工作紀錄](../../../../docs/worklogs/2026-10-18-binaryqk-stage1-results.md)。
