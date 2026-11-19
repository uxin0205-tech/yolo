# Partial75 隔離備案

Partial75 保留相同程式能力與獨立輸出根目錄，但
[configs/joint.yaml](configs/joint.yaml) 固定 `enabled: false`。依使用者決策，只有明確
要求才執行；本輪沒有建立 Pose baseline、沒有訓練、沒有驗證精度，也沒有配置 GPU。

它和Full35使用完全相同的v2訓練政策：logical Detect128／physical64×2、Pose16、
Detect:Pose weight 1.0:0.25、J0=8、J1=max20/patience8、J2=max80/patience17，以及
opt-in J3=max20/patience5。共用的是程式與政策，不是checkpoint、baseline、cache、
run name或實驗結論。

可做唯讀檢查：

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python variants/partial75/joint.py preflight
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python variants/partial75/run.py audit --verify-hashes
```

Full35 的 Pose checkpoint、baseline、cache、metrics 與 seed 結論都不得複用成
Partial75 的正式結果。

程式層已共享相同的固定-seed、source-group `fraction<1` 診斷能力，但本資料夾沒有建立
30% View、沒有 cache、也沒有 run。依使用者決策，只有再次明確要求 Partial75 時才會在
本資料夾獨立 materialize 與執行。
