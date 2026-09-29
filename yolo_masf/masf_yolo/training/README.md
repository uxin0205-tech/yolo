# 訓練模組

`profiles.py` 定義參數；`preflight.py` 驗證 loss/backward/step；`runner.py` 保留 custom model；`worker.py` 隔離執行；`completion.py` 判斷 run 完整性；`resume.py` 安全續跑。B1、SP2、SP2P 都採凍結 backbone 0–10 共 10 epochs，再全解凍 90 epochs。GPU 工作由單一 systemd pipeline 依序執行。
