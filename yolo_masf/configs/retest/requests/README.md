# 待執行正式實驗 requests

這個資料夾保存已確認設定、但不會自行啟動 GPU 的 worker request。

## B0-Fair

`b0_fair.json` 使用未修改的三尺度 P3/P4/P5 B0 graph，從與第二輪 P3 formal 相同的 `yolo11m_bat_detect_init.pt` 初始化，並採相同的 640、batch 16、SGD、momentum 0.937、cosine LR、AMP、seed 42、全模型 100 epochs 與 `lr0=0.001`。

它只修正「B0 訓練預算不同」的公平性問題；initializer 仍已接觸 BBT5，因此不能用來宣稱 unseen BBT5 泛化。

目前只準備 request，**尚未排程、尚未訓練**。確認 GPU 空閒後才執行：

```bash
PYTHONPATH=. ../.venv/bin/python -m masf_yolo.retest.worker \
  --request configs/retest/requests/b0_fair.json \
  --output artifacts/b1r-p2-p3-retest/worker/b0_fair.json
```
