# Artifact 管理程式

這裡是 Python 原始碼，不是實驗輸出。`io.py` 負責原子寫入與 lock；`state.py` 管理 stage hash；`checkpoints.py` 管理 canonical checkpoint；`finalize.py` 轉換 native best EMA；`strict_reload.py` 在新 process 嚴格載入。真正結果在 repo 根目錄 `artifacts/`。
