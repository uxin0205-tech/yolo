# B1R / P2 / P3 完整實驗流程

本文件說明第二輪實驗從資料鎖定、CPU 檢查、GPU 排程、模型訓練到統一評估的完整過程。精確結果見 [`results/REPORT.md`](results/REPORT.md)。

## 1. 固定研究輸入

1. Dataset 固定使用 `bbt5-detect-baseline/dataset`，不重新隨機切分。
2. 使用 Phase 1 的 group/hash-audited manifest：train/val/test=1,987/300/291 frames。
3. 類別順序固定為 `ball=0`、`bat=1`。
4. 初始化固定為 `bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`。
5. 初始化已看過 BBT5，因此所有結論限定為同源 operational ablation。

資料設定與 manifest 請看 [`data/README.md`](data/README.md) 與 `artifacts/static-phase1/dataset/`。

## 2. CPU preflight

在不使用 GPU 的階段先完成：

- YAML/config contract 驗證。
- P2/P3 五種變體的 model build。
- forward/backward smoke。
- source initializer transfer audit。
- P3 slot 確認沒有誤插 9×9 到 Lite/Partial variants。
- queue request 與 result manifest schema 測試。

對應程式在 `masf_yolo/retest/`，測試在 `tests/retest/`、`tests/models/test_mfam.py` 與 `tests/models/test_transfer.py`。

## 3. Baseline 診斷

### B1R-A / B1R-B

1. B1R 是四尺度 P2/P3/P4/P5 Detect baseline，並保留 B0 的 P3–P5 head shape。
2. A 段凍結 backbone indices 0–10，訓練 10 epochs，lr0=0.01。
3. B 段由 A-best 初始化，解除全部凍結，重新訓練 90 epochs，lr0=0.001。
4. B 段不是 Ultralytics resume epoch counter，而是新的 90-epoch full-unfreeze stage。

### Direct control

從同一 initializer 建立 B1R，直接全模型訓練 100 epochs，lr0=0.001，用來檢查 staged freeze 是否真的有幫助。

### B0-Fair control（已實作、待 GPU 空閒）

1. 保持原始 B0 的 P3/P4/P5 三尺度 Detect graph，不加入 P2 或 MFAM。
2. 從與 P3 formal variants 相同的 `yolo11m_bat_detect_init.pt` 初始化。
3. 使用與 P3 formal 完全相同的 640、batch 16、SGD、momentum 0.937、cosine LR、AMP、seed 42、全模型 100 epochs 與 lr0=0.001。
4. Request 位於 [`../configs/retest/requests/b0_fair.json`](../configs/retest/requests/b0_fair.json)。目前只完成實作與 CPU 驗證，不在既有 queue 中，因此不會自行占用 GPU。
5. 這個對照只能修正 B0 training budget 不一致；因 source initializer 仍 data-exposed，不能宣稱 unseen BBT5 泛化。

### Head-only + full control

仿照原 `yolo_p2` 的訓練概念：

1. 先讓 P2 slot 與 P2 Detect branch 可訓練，凍結 inherited backbone/neck 與 P3–P5 towers，訓練 20 epochs，lr0=0.01。
2. 承接 head-only best，解除凍結，全模型訓練 80 epochs，lr0=0.001。
3. P2 MFAM formal variants 從 head-only/control parent 的相容 tensors 初始化，但新的 MFAM slot 不繼承。

## 4. MFAM 變體

`PaperFormulaMFAM` 的運算順序是：identity 與平行 depthwise branches 相加，經第一個 1×1，與輸入 residual 相加，再經第二個 1×1。沒有 learnable branch weighting、額外 scale/bias gate 或動態 routing。

| Variant | Kernels | Channels |
|---|---|---:|
| PaperFormula-Full | 3、5、factorized 7、factorized 9 | 100% |
| Lite-35 | 3、5 | 100% |
| Lite-35-F7 | 3、5、factorized 7 | 100% |
| Partial50-35 | 3、5 | 前 50%；其餘 identity bypass |
| Partial25-35 | 3、5 | 前 25%；其餘 identity bypass |

每種 variant 各做兩個 placement：

- P2：四尺度 Detect 的高解析 P2 slot。
- P3：維持 B0 三尺度 Detect，只在 P3 feature slot 插入 MFAM。

## 5. GPU 單工排程

GPU queue 位於 `masf_yolo/retest/queue.py`，一次只啟動一個 worker，並等待原有 `yolo-p2-study.service`。每完成一個 stage 就寫入 `queue_state.json`，有 result manifest 的 stage 不重複啟動。

執行順序：

1. 等待/承接 control-full。
2. P2 五種 smoke，各 3 epochs。
3. P3 五種 smoke，各 3 epochs。
4. P2 五種 formal，各 100 epochs。
5. P3 五種 formal，各 100 epochs。

最終 queue 狀態為 `formal_complete`，21 個列入 queue 的 control/smoke/formal 工作均在 completed 清單；另有 B1R-A、B1R-B、Direct、Control-Head 的 worker manifests。

## 6. 統一後處理

訓練結束後依序執行：

```bash
../.venv/bin/python -m masf_yolo.retest.postprocess
../.venv/bin/python -m masf_yolo.retest.profile_all
../.venv/bin/python -m masf_yolo.retest.report
../.venv/bin/python -m masf_yolo.retest.audit
```

- `postprocess`：對 14 個正式評估模型統一產生 val/test metrics，共 28 份。
- `profile_all`：產生 14 份逐模型 profile，另有 1 份 summary JSON。
- `report`：產生 comparison.csv、summary.json 與中文表格。
- `audit`：確認 metrics=28、profiles=15，最終 `ok=true`。

指標使用 faster-coco-eval 1.7.2 的 COCO-style AP，並額外保存 Ball/Bat AP、Ball AP_S、recall、false positives 與整體 AP_S/AP_M/AP_L。

## 7. 證據與發布

| 證據 | GitHub 位置 |
|---|---|
| 每模型做法與結果 | [`results/experiments/`](results/experiments/) |
| CSV / JSON 總表 | [`results/comparison.csv`](results/comparison.csv)、[`results/summary.json`](results/summary.json) |
| Profiles | [`results/profiles/`](results/profiles/) |
| Queue requests / workers | [`results/metadata/`](results/metadata/) |
| Checkpoint lineage | [`results/weights/CHECKPOINT_STATUS.md`](results/weights/CHECKPOINT_STATUS.md) |
| 發布檔案 SHA-256 | [`results/PUBLICATION_MANIFEST.json`](results/PUBLICATION_MANIFEST.json) |

Raw predictions、false-positive 圖與大型 queue logs 保留在本機 runtime，但不作為 GitHub 首頁的必要依賴。Dataset 依要求不發布。

## 8. 已知限制

- 單一 seed，沒有誤差棒。
- initializer data-exposed。
- B0 的歷史訓練預算與本輪不同。
- 本輪 profile 是靜態成本，不是目標部署硬體 latency。
- 第二輪 formal `.pt` 在發布前已從外部 runtime 消失；SHA-256 與 metrics 還在，但無法直接下載模型。第一輪權重仍完整。
