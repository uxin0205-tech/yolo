# BCND：BDCN 改良重跑工作區

此資料夾依使用者指定命名為 `BCND/`；專案程式中的正式方法名稱仍是
BDCN（Binary Distance-Codebook Normalization）。這裡只保存改良實驗的
設定、操作方式與結果索引，演算法唯一實作仍在
`src/yolo_attention/bdcn.py`，queue 唯一入口仍是
`python -m yolo_attention.cli`。

## 為什麼重跑

舊 D0 使用 16 levels、step 0.125，只能表示

$$
d_{\max}=(16-1)\times0.125=1.875.
$$

超過此範圍的 score distance 全部落入最後一格；最後權重仍約為
$e^{-1.875}=0.153$，遠大於 PWL 在 $d=8$ 的 $e^{-8}$。因此這次直接固定
$d_{\max}=8$、64 levels（$\Delta\approx0.127$），並記錄 bucket histogram、
overflow rate、last-bucket rate 與實際最大 distance。

## 資料夾內容

```text
BCND/
├── README.md
├── RESULTS.md
└── configs/
    ├── README.md
    ├── bdcn-v2.yaml
    └── bdcn-v2-r1.yaml
```

## 執行流程

```bash
../.venv/bin/python -m yolo_attention.cli queue append-bdcn-v2 --json
../.venv/bin/python -m yolo_attention.cli queue validate --json
../.venv/bin/python -m yolo_attention.cli queue run --execute --json
```

流程直接從已完成 scale/bias 補償的 A0（V1-BR）做 10 epoch global
learned-codebook recovery，最後評估 reciprocal-LUT denominator；不增加
16/32/64 screening 或 winner selection。兩個 Attention site 都由
既有 fail-closed integration 同時替換。舊 runs 不會被覆寫。

大型 checkpoint 與 metrics 仍寫到 `artifacts/runs/<run_id>/`，不放在本
資料夾。量化仍鎖定，不會由此流程啟動。

## 外部接線

`BCND/` 擁有本次 branch 的 configs、方法說明與結果；共用實作不複製：

- `src/yolo_attention/bdcn.py`：distance/codebook/fused bucket-PV 唯一 source。
- `src/yolo_attention/evaluation.py`：將全 validation diagnostics 寫入結果。
- `src/yolo_attention/queue_workflow.py`：從 A0 追加這兩個 jobs。
- `reports/COMPUTE_AND_SIZE.md`：跨方法與全模型占比的整合報告。
