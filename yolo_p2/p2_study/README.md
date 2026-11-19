# P2 Study 訓練與操作

此目錄包含 YOLO11m-P2 的建模、權重轉移、可恢復排程、訓練、驗證、benchmark 與分析程式。

## 實驗順序

1. Preflight：環境、GPU、資料、權重與磁碟檢查。
2. Static tests：P2 尺度、權重轉移、loss、backward 與梯度檢查。
3. AutoBatch：固定所有正式實驗共用的 batch。
4. Gate/health：短訓練穩定性檢查。
5. A1：P2 Direct 全模型 fine-tune 100 epochs。
6. A2 Stage 1：只訓練新 P2 分支 20 epochs。
7. A2 Stage 2：以 MuSGD、lr0 0.001 解凍全模型，原訂 80 epochs；正式實驗在第 73 epoch 提前停止，採 epoch 39 best。
8. A0/A1/A2 完整 COCO val、RTX 5090 benchmark 與分析。

共同設定：COCO2017、imgsz 640、batch 9、seed 0、`deterministic: true`、AMP、workers 8。

## 操作

```bash
./p2_study/ctl.sh start
./p2_study/ctl.sh status
./p2_study/ctl.sh logs
./p2_study/ctl.sh stop
./p2_study/ctl.sh resume
```

控制器優先使用 transient systemd user service，否則使用 nohup。關閉 Codex 或 terminal 不會停止背景訓練。

前景除錯：

```bash
python -m p2_study.run --config p2_study/config.yaml
python -m p2_study.run --config p2_study/config.yaml --resume
```

## `artifacts` 與 `results`

- `p2_study/artifacts`：當次執行的 runtime 輸出，包括 logs、state、initial weights、runs、validation 與 benchmark。可重新產生，也可在確認封存後刪除。
- `p2_study/results`：本次已接受的正式成果，包含四權重、報告、指標、訓練歷史及 metadata。重新訓練不得覆寫。

中斷的訓練由 `artifacts/**/weights/last.pt` 續跑；正式封存只保留 best，不保留 last。

## 主要程式

| 檔案 | 用途 |
| --- | --- |
| `config.yaml` | seed、資料、GPU、epochs 與 benchmark 共同設定 |
| `coco2017.yaml` | Ultralytics COCO dataset 定義 |
| `ctl.sh` | 背景 start/status/logs/stop/resume |
| `run.py` | 階段排程、狀態與錯誤恢復 |
| `worker.py` | 訓練、驗證、benchmark workers |
| `models.py` | A0/A1/A2 建模與 Detect 權重轉移 |
| `analyze.py` | CSV/JSON 彙整、圖表與報告 |

資料見 [data/README.md](data/README.md)，正式成果見 [results/README.md](results/README.md)。
