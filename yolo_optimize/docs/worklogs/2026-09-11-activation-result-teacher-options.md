# 2026-09-11：Activation 完成與大小教師／雙資料集 KD 前置

## 變更與原因

完成 SiLU／qSiLU 各 10 epoch 配對與選定 qSiLU E2 匯出重驗，集中至 `activation/bridge_v1/RESULTS.md`。使用者補充 Detect 可採 YOLO26L，主代理依 research 技能核對官方模型、Pose 與 KD 文件，保存 `kd/dual_task_v1/TEACHER_OPTIONS.md`；不使用子代理。

## 驗證與結果

qSiLU 相對配對 SiLU：COCO -0.000423、person -0.000896、ball box +0.011293、ball pose +0.004936、bat box -0.000942、bat pose -0.001166。匯出後獨立重建精確重現八項 Bit-True AP。通過 activation 淘汰線，但原融合 gate 仍有 BBAT pose／bat box／bat pose 三項未過；沒有宣稱全部恢復。

直接 FP-QK teacher 候選全量驗證八項全降，未啟動 KD。CPU `kd/dual_task_v1/preflight.py` 通過兩個 score site 的 exact-forward／eval 等價、state 不變與有限非零 surrogate 梯度；輸出 `artifacts/score-gradient-preflight-v1.json`。不是全模型 KD 訓練驗收。

核對 canonical BBAT5 YAML：ball／bat、兩個三維格式關鍵點；官方人體 17 點 Pose 不可直接作 BBAT5 輸出教師。建議先驗證大 Detect、沿用較強且已訓練的原獨立 Pose 作候選；不預設另訓大型 Pose。研究依據與限制見教師報告。

## 困難與解法

qSiLU physical32 OOM，已讓兩臂統一 physical16、logical128 不變並通過真實 smoke；正常結果保留。直接切 FP-QK 不能產生更強教師，因此停止該教師路線，不以弱教師直接排 KD。新 CPU 稽核無其他困難。

## 未解事項與目前狀態

Activation 配對及選定匯出驗證已完成；無執行中的本研究 GPU job。雙教師是原單 joint 排名蒸餾規格的修訂，需要明確確定後再實作對齊、全模型梯度與 K0／KD 配對。未下載或驗證 YOLO26L，未進行 KD 訓練、第二 seed、PTQ／QAT 或上板測試。啟動 GPU 後沿用 600 秒事件式 blocking monitor，不啟動空等待假裝訓練。
