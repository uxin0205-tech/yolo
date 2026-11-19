# 2026-10-28：核對目前 Softmax 的 PWL 與替代方法

## 任務與範圍

使用者詢問 Softmax 是否已進行 PWL，以及有沒有嘗試其他方法。本次唯讀核對既有第 3 階段結果、比較 CSV、normalization manifest 產生程式，以及第 5 階段目前選定模型的 manifest 與實體 SHA；不啟動 GPU、訓練或下一階段。

## 核對結果

目前 `yolo_hog/result/stage5/selected.pt` 仍指向已採用的第 3 階段 KD 第五輪。實體 SHA 與 sidecar 一致，為 `fe3e84cffe41c82b9e2d2ab83f6f2fdbf5df17d9cd99527ff1ef4e0d664dd71c`。manifest 的 candidate 為 D1，exp_backend=integer-pwl-10-20、reciprocal_backend=lut256、complete_fixed_normalization=true。

實際契約是 row-max 後在 [-10,0] 使用 20 段整數 PWL exp，權重 UQ1.15，分母整數加總；256-entry reciprocal LUT 採 Q1.16 尾數與獨立 exponent，輸出 probability UQ1.24，最大 1024 tokens、分母 uint26。final_renormalization=false。訓練反向代理仍使用浮點 PWL 與精確分母，不能將它誤稱推論路徑回退浮點除法。QKV、P·V、projection、Conv、SiLU 仍浮點，未做 RTL／HLS 或板端速度量測。

第一輪 N1～N6 使用共同分母方式進行候選比較，並完成 COCO val5000：Float PWL [-8,0]/16 為 0.514165814，Float PWL [-10,0]/20 為 0.514204951，BitTrue PWL [-10,0]/20 為 0.514158557，Score-indexed LUT 為 0.514213024，Power-of-two／SHIFT 為 0.513653572，Normalized HardSigmoid 為 0.510462290。N7～N9 的 Multimax Top-1／3／5 超過事前 P·V relative L2 門檻，只完成真實資料診斷，未做完整 AP。

第二輪固定整數 PWL exp，比較 D1 LUT256、D2 reciprocal PWL16、D3 LUT64＋一次 Newton，三組均完成 val5000，AP 分別 0.514189349、0.514128963、0.514169004。依事前完整路徑規則選 D1；相對 S1 parent 0.514151658 沒有明顯掉點。這些微小差值不代表統計顯著優勢。後續 KD 第五輪 AP 0.514620791，仍保留 D1，並非切回 exact Softmax。

來源：[原始比較](../../yolo_attention/artifacts/stage3/formal-20261019-v1/comparison.csv)、[第 3 階段報告](../../yolo_attention/artifacts/stage3/formal-20261019-v1/RESULTS.md)、[目前交接](../../yolo_hog/result/stage5/handoff.json)。本次沒有修改模型或研究程式。

## 困難與解法

首次搜尋猜測的 stage3_protocol.py 路徑不存在，改用實際 stage3_checkpoint.py 的候選映射與 manifest 核對；沒有影響既有結果。其餘困難：無。

## 驗證與限制

本次是文件、程式設定與 SHA 核對，沒有重跑 AP 或 runtime forward；上列 AP 與數值測試均是既有正式結果。原始 stage3 README／報告部分段落為 KD 或後續階段開始前的歷史敘述，目前模型以 Stage5 handoff 及其實際 manifest 為準。尚未啟動第 6 階段。
