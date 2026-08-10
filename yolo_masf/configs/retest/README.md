# Retest 設定

`b1r_p2_p3_retest.yaml` 是第二輪實驗唯一設定入口，鎖定 BBT5 detect dataset、
pose-derived initializer、seed 42、SGD、momentum 0.937、cosine、B1R-A freeze 0–10
十個 epoch、B1R-B 九十個 epoch，以及必要時 direct 一百個 epoch。輸出只寫入
`artifacts/b1r-p2-p3-retest/`。
