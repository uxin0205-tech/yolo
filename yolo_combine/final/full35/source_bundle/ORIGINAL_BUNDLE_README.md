# Full35／Partial75 最終可攜式交付包

此目錄封裝本次比較實際使用的程式碼、設定、權重與 AP 證據，可整個複製到其他工作目錄。比較單位是 9 個已保存的 Bit-True 候選：A0、Full35 的 A2／B-30%／C-30%／B-100%，以及 Partial75 的 A2／B-30%／C-30%／B-100%。

`artifacts/checkpoints/` 原有 16 個 Bit-True 檔案，其中多個 `source-accepted-a2` 只是同一 A2 權重在不同 queue 中重新封裝。`models.json` 以模型 state-dict SHA256 合併這些重複項，因此不會把相同模型重測多次。所有 8 個不同的 Full35／Partial75 Bit-True 權重，加上 A0，都已在 COCO2017 與 BBT5 上完成驗證。

## 目錄

- `code/achitechure_1/`：本實驗的完整 Python package snapshot，包含 Full35、Partial75、checkpoint、驗證與 queue 邏輯。
- `code/yolo_attention/`：checkpoint 反序列化與 Bit-True PWL 所需的確切 attention package snapshot。
- `weights/bittrue/`：9 個已驗證候選；AP 報告一律使用這些檔案。
- `weights/float/`：8 個 Full35／Partial75 最佳 Float checkpoint，供接續訓練或重新 materialize；A0 沒有在此工作區保存對應 Float 版本。
- `configs/`：attention 與兩份資料集設定。
- `reports/`：人類可讀、JSON、CSV 與逐候選 raw metrics。
- `models.json`：候選、權重 SHA256、state-dict fingerprint、來源與 gate 決策。
- `MANIFEST.json`：整包檔案的 SHA256 manifest。

## 環境

正式契約是 Python 3.12、PyTorch `2.11.0+cu128`、Ultralytics `8.4.90`。可在新的 virtual environment 安裝：

```bash
python -m pip install -r requirements-lock.txt
```

兩份 dataset YAML 保留本次實驗的絕對路徑。複製到別台主機後，先把 `configs/datasets/*.yaml` 的 `path` 改成當地實際位置；不要修改類別順序。BBT5 必須維持 `sports ball=32`、`baseball bat=34`。

## 驗證權重與程式碼

先做 CPU 結構與雜湊稽核，不會跑 inference：

```bash
python scripts/inspect_models.py
python scripts/verify_bundle.py
```

單獨驗證或依序驗證所有模型：

```bash
python scripts/validate_all.py --dataset all
python scripts/validate_all.py --dataset bbt5 --model partial75-a2
```

驗證固定使用 `imgsz=640`、batch 8、workers 6。controller 每個 model／dataset 啟動獨立 child process，結束後由作業系統回收 CUDA 與 host RAM；啟動前若 RAM 低於 1.5 GiB 或 free VRAM 低於 4 GiB 就 fail closed。可用 `--workers`、`--batch` 明確覆寫，但這會形成新契約，不能直接和本報告混用。

完整 COCO validation 會產生 `predictions.json`。若需要重算 canonical COCO API AP：

```bash
python scripts/compute_coco_ap.py \
  --annotations /path/to/coco2017/annotations/instances_val2017.json \
  --predictions /path/to/predictions.json \
  --output /path/to/coco-canonical.json
```

## 結果入口

請先看 [`reports/FULL35_PARTIAL75_AP.md`](reports/FULL35_PARTIAL75_AP.md)。JSON 保留完整精度，CSV 是逐 dataset／evaluator／scope 的長格式資料。

COCO 的「Ultralytics internal」與「canonical COCO API」不可混成同一欄比較；BBT5 沒有 COCO JSON ground truth，使用 Ultralytics internal evaluator。BBT5 valid 有 93/567 張影像的 COCO ID 出現在 COCO train2017，因此只適合候選間的相對比較，不是獨立泛化測試。
