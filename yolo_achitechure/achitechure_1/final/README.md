# Full35／Partial75 最終可攜式交付包

此目錄封裝本研究實際使用的程式碼、設定、權重、完整 AP、訓練曲線與 RAM／VRAM 證據，可整個複製到其他工作目錄。比較單位是 11 個已保存 Bit-True 候選：A0，以及 Full35／Partial75 各自的 A2、B-30%、C-30%、B-100%、C-100%。

最終結論如下：

- 完整資料 Phase C 已完成，Full35／Partial75 均 rollback 到 A2；accepted checkpoint 沒有改變。
- 依既定 overall COCO gate 與同機 latency tie-break，Partial75 是 Full35／Partial75 兩個 P3-MASF 架構中的正式工程 winner。
- Partial75 A2 沒有相對 A0 顯示實質整體 accuracy 提升，A0 latency 也較快，因此不宣稱 P3-MASF 優於 baseline。
- 使用者指定的 BBT5 `detect_dataset` 指標優先完整保存；overall／bat AP50-95 最高是 Full35 C-30%，ball AP50-95 最高仍是 A0。
- `EXPERIMENT_SPEC.md` 規劃的 winner LR tuning T1／T2／T3 尚未執行；本交付只宣告架構 winner，不虛構 tuning 結果。

## 目錄

- `code/achitechure_1/`：本實驗完整 Python package snapshot，包含 Full35、Partial75、checkpoint、驗證與 queue 邏輯。
- `code/yolo_attention/`：checkpoint 反序列化與 Bit-True PWL 所需的 attention package snapshot。
- `weights/`：唯一正式權重入口；11 個 Bit-True、10 個 Float best，以及 JSON／CSV 權重索引。
- `configs/`：attention 與兩份資料集設定。
- `reports/RTX5060TI_FINAL_0824.md`：完整結論、訓練、資源、效率與限制。
- `reports/FULL35_PARTIAL75_AP.md`：11 個候選的 COCO2017／BBT5 完整 AP、P、R、F1 表。
- `reports/training/`：8 次 B/C 訓練各自的 `results.csv`、訓練圖、四種 box curve、confusion matrix、manifest 與 telemetry。
- `reports/figures/`：跨 run 訓練軌跡、BBT5 AP 與資源比較圖。
- `models.json`：可驗證候選與權重 fingerprint。
- `MANIFEST.json`：整包檔案 SHA256 manifest；最後驗證後產生。

## 環境與資料契約

正式契約是 Python 3.12、PyTorch `2.11.0+cu128`、Ultralytics `8.4.90`、imgsz 640、validation batch 8、workers 6。可在新的 virtual environment 安裝：

```bash
python -m pip install -r requirements-lock.txt
```

兩份 dataset YAML 保留本次實驗的絕對路徑。複製到別台主機後，先修改 `configs/datasets/*.yaml` 的 `path`；不要改類別順序。BBT5 必須維持 `sports ball=32`、`baseball bat=34`。

## 驗證與重建報告

先做 CPU 結構與雜湊稽核：

```bash
python scripts/inspect_models.py
python scripts/verify_bundle.py
```

依序驗證所有模型或單一模型：

```bash
python scripts/validate_all.py --dataset all
python scripts/validate_all.py --dataset bbt5 --model partial75-a2
```

controller 每個 model／dataset 啟動獨立 child process；啟動前若 RAM 低於 0.5 GiB 或 free VRAM 低於 4 GiB 就 fail closed。這個驗證門檻和訓練 queue 的啟動門檻是不同用途。

由既有 raw evidence 重建 AP 與最終報告：

```bash
PYTHONPATH=scripts:code python scripts/build_report.py
PYTHONPATH=scripts:code python scripts/build_training_report.py
```

COCO 的 Ultralytics internal 與 canonical COCO API 不可混成同一排名欄。BBT5 valid 有 93/567 張影像的 COCO ID 出現在 COCO train2017，因此適合候選間相對比較，不是完全獨立泛化測試。
