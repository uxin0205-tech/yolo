# Clean module

這個 deep module 將 clean initializer、公平模型矩陣、資料 split 可見性、CPU transfer inspect 與明確單 job training 隱藏在小型 interface 後方。

| 檔案 | 責任 |
|---|---|
| `contracts.py` | 鎖定 initializer SHA、dataset hash、seeds、實驗矩陣與 split 用途 |
| `builder.py` | 建立 B0/P3/P2 clean 模型並保存逐 tensor transfer report |
| `profiles.py` | 產生 strict-fair direct100 與分開排名的 P2 control profiles |
| `data_view.py` | 產生物理上不含 `test` key 的 trainer YAML |
| `inspect.py` | CPU-only 架構、參數與 transfer feasibility 稽核 |
| `plan.py` | 產生 28 個 formal/control prepared-only jobs 與依賴關係 |
| `worker.py` | 唯一會開始單一 GPU training job 的明確入口 |
| `queue.py` | 依序執行 acceptance、12 smoke、24 strict-fair 與 4 control jobs，支援 `last.pt` 續跑 |
| `postprocess.py` | 稽核 checkpoint、統一 val、硬體 profile、selection freeze、historical test 與中文報告 |

目前 CPU 稽核準備 12 個 strict-fair 架構與 2 個 P2 control，共 14 個 stage、每個 2 seeds，合計 28 個正式／控制 jobs。GPU 執行前另規劃 12 個 seed 42 smoke gates；smoke 只做工程驗證，不列入模型排名。

正式設定與研究解讀見 [`../../configs/clean/README.md`](../../configs/clean/README.md)。

訓練完成後執行 `python -m masf_yolo.clean.postprocess all`。此命令具 process lock、逐項 hash 檢查與可續跑輸出；historical test 在 `selection.json` 建立前會 fail closed。
