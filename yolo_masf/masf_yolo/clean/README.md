# Clean module

這個 deep module 將 clean initializer、公平模型矩陣、資料 split 可見性、CPU transfer inspect 與明確單 job training 隱藏在小型 interface 後方。

| 檔案 | 責任 |
|---|---|
| `contracts.py` | 鎖定 initializer SHA、dataset hash、seeds、實驗矩陣與 split 用途 |
| `builder.py` | 建立 B0/P3/P2 clean 模型並保存逐 tensor transfer report |
| `profiles.py` | 產生 strict-fair direct100 與分開排名的 P2 control profiles |
| `data_view.py` | 產生物理上不含 `test` key 的 trainer YAML |
| `inspect.py` | CPU-only 架構、參數與 transfer feasibility 稽核 |
| `plan.py` | 產生 18 個 prepared-only jobs 與依賴關係 |
| `worker.py` | 唯一會開始單一 GPU training job 的明確入口 |

正式設定與研究解讀見 [`../../configs/clean/README.md`](../../configs/clean/README.md)。
