# Stage7 可直接使用的模型交接

**下一階段請使用 `qsilu_pq.pt`。** `poly_quality.pt` 是本輪綜合精度最佳的比較模型；RepConv 按研究要求固定從 qSiLU 出發。本包尚未加入 RepConv。

請整個 `handoff/` 一起交接。兩份 `.pt` 均為原選定 FP32 權重，SHA 未改，包含完整 state dict 與血緣 manifest；沒有 optimizer。這是客製雙任務圖，使用本包 `load_model.py` 載入，不能把 `.pt` 直接交給官方 `YOLO(...)` parser。

| 檔案 | 用途 |
| --- | --- |
| [qsilu_pq.pt](qsilu_pq.pt)、[qsilu_pq.json](qsilu_pq.json) | qSiLU 第 8 輪最佳模型、完整設定與 SHA；RepConv 共同 parent |
| [poly_quality.pt](poly_quality.pt)、[poly_quality.json](poly_quality.json) | PolyQuality 第 10 輪最佳模型、完整設定與 SHA |
| [load_model.py](load_model.py) | 預設載入 qSiLU，核對類別、係數、alias 與完整 Attention 契約 |
| `*.architecture.pt.gz` | 零參數骨架；模型結構與非 state dict 設定，不能單獨推論 |
| `runtime/`、[runtime-sources.json](runtime-sources.json) | 必要類別的凍結匯出與來源 SHA；不是另一套研究主線 |
| [model-config.json](model-config.json) | 輸入／輸出、Binary、scale／bias、PWL／reciprocal、activation 等交接設定 |
| [metrics.json](metrics.json) | 十項正式 AP 與完整 evaluator 口徑 |
| [manifest.json](manifest.json) | 交付檔案大小、SHA、模型角色 |
| [environment.json](environment.json)、[requirements.txt](requirements.txt) | 已測環境與第三方依賴版本 |
| [validate_package.py](validate_package.py)、[驗證結果](validation/result.json) | 獨立 CPU 程序的載入、逐值等價與單雙任務 routing 檢查 |

## 直接測試

在本機專案根目錄：

```bash
.venv/bin/python -B yolo_activation/result/stage7/handoff/load_model.py --candidate qsilu_pq --device cpu --size 160
```

交給其他人後，在具備第三方套件的 Python 環境執行：

```bash
cd handoff
python -B load_model.py --candidate qsilu_pq --device cpu --size 640
# 比較模型
python -B load_model.py --candidate poly_quality --device cpu --size 640
```

在新的 Python 程序使用 tensor：

```python
import torch
from load_model import load_model

model = load_model(device="cpu", candidate="qsilu_pq")
x = torch.zeros(1, 3, 640, 640)
with torch.inference_mode():
    outputs = model(x, task="both")
detect = outputs["detect"][0]
pose = outputs["pose"][0]
```

`task` 可選 `"both"`、`"detect"`、`"pose"`。GPU 可指定 `device="cuda:0"`；本次可攜包僅重新做 CPU 驗證，未重新占用 GPU。第三方套件版本依本機記錄提供；請依接收端硬體準備相容的 torch／torchvision。本包自帶實際驗證使用的客製 Ultralytics 8.4.90，不應以另一版本覆蓋。未做乾淨新機安裝測試。

## 輸入與輸出

正式 AP 使用 RGB、BCHW、FP32、`[0,1]`，`imgsz=640`、`rect=True` 的既有 evaluator；實際 val batch 可能是長方形，並非一律 640 × 640。上方範例用已通過 CPU 驗證的 640 × 640 輸入。實際圖片要等比例 resize＋letterbox（padding 114），OpenCV BGR 圖需轉 RGB。本入口接受 tensor，沒有自動讀圖、letterbox 或將座標映回原圖；要重現 AP，須使用原 evaluator 的完整 batch／resize 規則。

Detect decoded 為 `[B, 300, 6]`，每筆 `[x1,y1,x2,y2,score,class_id]`；Pose decoded 為 `[B, 300, 12]`，後 6 個值是兩點的 `x,y,visibility`。座標位於輸入 letterbox 空間，需扣 padding、除縮放比例再畫回原圖。請依 score 篩選；此為 E2E head，不應任意增加會改變結果的後處理。Detect 是 COCO 80 類，Pose 是 BBAT5 v1 `0=ball、1=bat`，不可混用類別編號。

## 保留的設定與精度

兩模型保留 B2 sign＋Hadamard 二值 QK、2 sites × 4 heads、16 個 S1 Q10 scale、signed16/1024 分解式位置 bias。D1 使用整數 PWL exp `[-10,0]`／20 段、256-entry reciprocal LUT、UQ1.16 reciprocal、UQ1.24 probability，max_tokens=1024，沒有最後浮點除法重正規化。每個 site 的 H、W 上限均為 32。

qSiLU 的類別、固定係數、187 個註冊位置、30 組共享 alias 與 163 個實際部署位置都保留；另 24 個位置是 Pose RealNVP 訓練 loss 專用。完整位置與係數見 `.json`，不能只依 state dict 猜 activation。

| 模型 | COCO AP50–95 | BBAT box AP50–95 | BBAT Pose AP50–95 |
| --- | ---: | ---: | ---: |
| qSiLU E8 | 0.513951 | 0.613904 | 0.900762 |
| PolyQuality E10 | 0.515610 | 0.616505 | 0.903934 |

正式評估為 COCO val 5,000、BBAT5 v1 val 683、FP32 E2E；上述是本輪已有的完整任務結果。交付後 CPU 160／640 合成輸入的全部輸出與原模型逐值相同，並拒絕錯誤 activation／PWL 常數。這項交付驗證不等於重新跑 AP。

模型仍有 Conv、QKV、P·V、projection 與 activation 浮點運算；D1 是整數 reference，尚無 RTL／HLS 或板端量測。兩候選通過本輪 activation 相對門檻，原獨立模型的四項融合目標仍未全部達成。

## 接續研究與可攜邊界

本包可在不讀研究程式、舊 checkpoint 或資料集的情況下做 tensor 推論。骨架使用 Python 物件序列化，僅載入本次可信任交付並核對 SHA；`load_model.py` 會先驗證檔案與完整設定。

接續 RepConv 時，在原專案使用 `stage7.model.load("yolo_activation/result/stage7/handoff/qsilu_pq.pt")`，沿用正式研究程式。可從所選權重開始新的訓練段；本包沒有 optimizer／scaler，不能精確續跑已結束的第 8 輪。研究改動請回 canonical 專案，重新匯出本包，不直接修改 `runtime/`。

qSiLU SHA：`2f91ec1862ad62304f1e2727e9b9ee64a27c5a46a45137f83a74dc7f14c46a69`。

PolyQuality SHA：`447b06043eae81cc4422c4b1a05f40d7843526f2ac3eec15860e4b454d2aac7d`。
