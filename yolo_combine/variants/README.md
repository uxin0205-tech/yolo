# Architecture workspaces

`full35/` 與 `partial75/` 是兩個完全隔離、但介面相同的實驗工作區。兩邊以
SHA256 引用同一份唯讀來源，不複製核心實作；所有可變資料、cache、report、checkpoint
與訓練輸出都只能留在各自的 `variants/<architecture>/artifacts/`。

| Workspace | 角色 | CPU acceptance | 正式 Pose26 |
|---|---|---|---|
| `full35` | 主線 | Float/Bit-True exact、2:1 synthetic/real backward 通過 | 舊 b16/e10 provisional；新 P1 未跑 |
| `partial75` | 備案 | Float/Bit-True exact、2:1 synthetic/real backward 通過 | 尚未正式訓練 |

每個資料夾都有同一組相對路徑：

```text
README.md
variant.yaml
run.py
baselines/independent.json
baselines/independent.md
configs/training.yaml
configs/fusion.yaml
artifacts/                 # 本機產物；不納入版本控制
├── cpu-validation.json
├── joint-smoke.json
└── datasets/              # 該架構自己的 view 與 cache
```

`run.py` 從自身所在資料夾決定架構，不接受 `--architecture`，因此不能把 Full35
輸出誤寫進 Partial75。共享的 `src/yolo_combine/` 是無狀態的深 module；架構來源、
設定與所有可變輸出在 workspace seam 之後完全分開。

CPU/filesystem 稽核：

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python variants/full35/run.py audit --verify-hashes
CUDA_VISIBLE_DEVICES="" .venv/bin/python variants/partial75/run.py audit --verify-hashes

CUDA_VISIBLE_DEVICES="" .venv/bin/python variants/full35/run.py cpu-check
CUDA_VISIBLE_DEVICES="" .venv/bin/python variants/partial75/run.py cpu-check

CUDA_VISIBLE_DEVICES="" .venv/bin/python variants/full35/run.py joint-smoke --ratio 2:1
CUDA_VISIBLE_DEVICES="" .venv/bin/python variants/partial75/run.py joint-smoke --ratio 2:1
```

`cpu-check` 驗證來源 hash、BBT5 lineage、Float/Bit-True 獨立模型對共享模型逐值
相同、materialization、參數數量與 synthetic 2:1 backward。`joint-smoke` 使用各自
folder 內的 COCO 128-image view 與完整 BBT5 view；不再把 cache 寫回來源。

建立工作區與上述 CPU 命令都不會配置 GPU。正式 profile 已鎖定 batch128、P1 17、
P2 22、P3 最多100；P1/P2 必須跑滿，P3 patience 20。既有 batch16/e10 P1 只保留
為 provisional 歷史結果。

COCO cache 來源副作用與待復原事項見
[CPU validation audit](../docs/audits/2026-08-22-cpu-variant-validation.md)。
