# Retest 模組

這個資料夾只服務 B1R、P2 與 P3 第二輪實驗，不改動 Phase 1 的舊 checkpoint 或
`artifacts/static-phase1/`。

- `contracts.py`：鎖定資料來源、五個變體順序、epochs、freeze 與 hash。
- `builder.py`：建立 B1R 四尺度、P2 slot 與 B0 三尺度 P3 slot。
- `profiles.py`：建立 B1R-A/B、direct、smoke、formal 的訓練參數。
- `worker.py`：單 stage、可續跑的隔離訓練 worker。

資料固定為 `bbt5-detect-baseline/dataset`，初始化固定為
`bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`。論文公式版含 DW3、
DW5、DW(1×7→7×1)、DW(1×9→9×1) 與 identity，再接兩個 1×1 residual fusion；
不加入 branch weight、scale 或 gate。
