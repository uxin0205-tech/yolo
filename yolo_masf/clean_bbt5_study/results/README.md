# 結果

目前沒有 Clean GPU 正式結果。本目錄不得預填舊 data-exposed 數值。

GPU 完成後，每個實驗應保存：

- 三個 seeds 的 resolved config、initializer/dataset/checkpoint hashes。
- validation 與 historical-test COCO 指標、AP_S/AP_M/AP_L、Ball/Bat 指標。
- Params、GFLOPs、peak memory、FP16 latency。
- 逐 seed 原值與 mean ± std。
- checkpoint lineage、selection freeze、strict reload 與 final audit。

預計以 `results/experiments/<experiment>/<seed>/` 保存逐 run 證據，根層再生成 comparison CSV 與中文報告；只有真實完成的輸出才能建立。
