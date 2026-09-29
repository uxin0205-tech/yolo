# 2026-08-31：YOLO 架構、RepConv 與 BinaryQK 精度恢復研究

## 任務與範圍

盤點 `/home/uxin/yolo` 目前 YOLO26、Full35 Detect＋Pose shared model、MASF、BinaryQK、activation 與 quantization 工作線，研究以下問題：

1. RepConv 是否值得先只替換 P3／P4 Neck 的少數 `3×3 Conv`。
2. MASF 是否已有足夠證據保留或移除。
3. BinaryQK 大幅掉點的原因與可優先驗證的恢復方法。

本輪只做文件研究、既有結果／checkpoint 唯讀稽核與 CPU 診斷；沒有執行 GPU training、validation、PTQ、QAT、export 或外部狀態修改。

## 變更內容與原因

1. 新增 `docs/research/2026-08-31-repconv-binaryqk.md`：

   - 畫清 Full35 shared YOLO26m layers 0–22、雙任務 heads、兩個 BinaryQK sites 與 P3 MASF 的目前關係。
   - 分開報告 YOLO11m 與 YOLO26 BinaryQK lineage、evaluator 與恢復階梯。
   - 以 RepVGG、YOLOv7、YOLOv6、BinaryAttention、Bi-ViT、QARepVGG 等一手來源，區分文獻事實、本地證據、工程推論與待驗證假說。
   - 制定 RepConv R0–R3、MASF M0–M2、BinaryQK B0–B9 及同 lineage interaction 矩陣。

2. 新增 `scripts/audit_accuracy_regressions.py`：

   - 只讀取現有正式 CSV。
   - 對 MASF strict-fair、YOLO11 BinaryQK sign-only 與 YOLO26 A-FINAL 設 fail-closed 回歸訊號。
   - 列出 YOLO11／YOLO26 BinaryQK 的恢復階梯，讓後續代理能快速重現問題。

3. 新增 `scripts/audit_repconv_seams.py`：

   - 驗證 YOLO26 YAML 的 layer 17／20 確實為 `Conv[256,3,2]`／`Conv[512,3,2]`。
   - 驗證原 Conv／BN 搬到 RepConv `3×3` branch、`1×1` BN 歸零後保持初始函數。
   - 驗證 `fuse_convs()` 輸出誤差與訓練期額外參數量。

4. 新增子專案 `README.md`、研究索引及本工作紀錄索引，提供資料集規範、報告與稽核入口。根層 `/home/uxin/yolo/README.md` 原本已持續提供 canonical BBAT5 與全域工作紀錄入口，本輪未改動。

## 主要研究結果

### 現行正式模型

- authoritative model 是 `yolo_combine/final/full35` 的 Full35 J3 `best_joint`。
- shared layers 0–22 只執行一次，再分成 COCO80 Detect 與 BBAT5 Pose26 heads。
- shared parameters `26,529,701`；兩個獨立模型合計 `45,580,762`，少 `41.796%`。
- Bit-True COCO overall、BBAT box、BBAT pose mAP50-95 為 `0.498022`／`0.630036`／`0.903717`，八項 gate 通過。

### MASF

- YOLO11m BBAT5 strict-fair：B0 `0.460696`，最佳 MASF `0.452507`，差 `-0.008189`。
- YOLO26m COCO：A0 `0.506737`、Full35 `0.506391`、Partial75 `0.506754`，都在 `0.001` tie band，無 material gain 證據。
- Full35 J3 checkpoint 的 `graph.model.16.p3_masf.alpha=0.1106591076`；目前模型已適應該分支，移除前先做 `alpha=0` full validation。

### RepConv

- 首輪最乾淨 seams 是 `model.17`（P3→P4）與 `model.20`（P4→P5）兩個 outer stride-2 Conv。
- 兩者自然沒有 identity branch；局部替換符合 YOLOv7 planned re-parameterization 的風險控制方向。
- 它是 training-time over-parameterization；部署 fuse 後與原本一樣是單一 `3×3`，不降低相對原 Conv 的部署 MACs。
- layer 17／20 訓練期合計增加 `329,216` parameters，fuse 後消失。

### BinaryQK

- YOLO11 COCOeval zero-train：E0 `0.510850` → E1-S `0.459297`，差 `-0.051554`。
- YOLO11 QAT/KD：T0 `0.512671`，最佳 binary T6-F/A `0.511841`，差縮至 `-0.000830`。
- YOLO26：B26-FP `0.517998`，W-DIR `0.507457`，A-FINAL `0.506357`；正式 final 仍差 `-0.011641`。
- W-PROG `0.502866` 低於 W-DIR；PWL／SHIFT 幾乎無損，所以不優先重跑 progressive 或 normalization。
- 下一輪順序：one-site isolation → W-DIR 低 LR full-model recovery → output/feature KD 或 ranking-aware KD → per-head threshold/temperature → full residual dual-basis。

## 驗證方式與結果

### RepConv CPU 稽核

命令：

```bash
/home/uxin/yolo/.venv/bin/python scripts/audit_repconv_seams.py
```

結果：

- layer 17／20 YAML seam：通過。
- identity-safe transfer：`max_abs=0`。
- fuse 後相對原 Conv：`max_abs=1.073e-6`，通過 `1e-5` gate。
- 額外 train-time parameters：`66,048 + 263,168 = 329,216`。
- 最終狀態：`[GREEN]`。

### 精度回歸稽核

命令：

```bash
python3 scripts/audit_accuracy_regressions.py
```

結果：兩次執行均得到相同三個預期紅燈，exit code `1`：

- MASF strict-fair：`-0.008189`。
- YOLO11 sign-only：`-0.051554`。
- YOLO26 A-FINAL：`-0.011641`。

exit code `1` 是刻意的 fail-closed 結果，表示正式 evidence 仍有 accuracy regression，不是程式錯誤。

### 語法與 checkpoint 稽核

- `python3 -m py_compile scripts/audit_accuracy_regressions.py scripts/audit_repconv_seams.py`：通過。
- `/home/uxin/yolo/.venv/bin/ruff check ...`：兩個腳本最終 `All checks passed!`。
- `/home/uxin/yolo/.venv/bin/ruff format --check ...`：兩個腳本均已格式化。
- `torch.load(..., weights_only=True)`：正式 Full35 inference checkpoint schema 可讀；MASF alpha 與 layer 17／20 tensor shape 均核實。
- 本輪未用 `weights_only=False` 反序列化 checkpoint。
- 5 份 Markdown 的相對連結檢查：`broken_links=0`。
- 尾端空白與 merge-conflict marker 掃描：無命中。

## 困難與解法

1. 直接檔案 patch 多次遭 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`：先以標準 `patch` 完成必要修改，之後定位可用的系統 `apply_patch`，改在核准的 escalated PTY 中精確套用後續 patch。早期 `patch` 產生的 `audit_accuracy_regressions.py.orig` 已用 `apply_patch` 移除。
2. 系統沒有 `python` 指令：改用 `python3`；需要 torch／Ultralytics 的診斷固定使用 `/home/uxin/yolo/.venv/bin/python`。
3. 初版 RepConv scalar 診斷觸發 requires-grad 轉型警告：改用 `detach().item()`，重跑後無警告且結果不變。
4. 廣域 `rg` 曾產生大量歷史 log 命中：改以正式 README、REPORT、CSV、config 與 checkpoint schema 為範圍，避免把歷史或無效 run 混入結論。
5. 直接執行 `./scripts/audit_repconv_seams.py` 會使用系統 `python3`，該環境沒有 torch：README 與正式驗證固定使用 `/home/uxin/yolo/.venv/bin/python scripts/audit_repconv_seams.py`，最終重跑為 `[GREEN]`。
6. `py_compile` 產生兩個可重建 `.pyc`：完成驗證後已刪除兩個檔案及空的 `scripts/__pycache__/`，沒有刪除任何使用者資料或模型產物。
7. Ruff 驗證產生可重建的 `.ruff_cache/`：最終檢查前已刪除該 cache；未移除任何原始碼、資料集、權重或實驗結果。

## 資料集紀錄

- 本輪沒有建立 runtime View、沒有執行資料載入實驗，也沒有改動任何影像、label、split 或 registry。
- 後續 BBAT5 工作仍只允許 `/home/uxin/yolo/original/pose/derived/bbat5-v1/` 的正式 Pose／Detect Task View。
- `/home/uxin/yolo/original/pose/dataset/` 與 `detect_dataset/` 本輪只作規範閱讀，沒有作為訓練入口。

## 未解事項或風險

1. 尚未執行 MASF `alpha=0` 的 Full35 八項 validation，因此不能宣稱目前正式模型可無損移除 MASF。
2. 尚未實作 Full35 RepConv graft／parser tests，也未跑 R0–R3 GPU 訓練；CPU 結果只證明 seam 與數值等價可行。
3. 尚未跑 BinaryQK one-site、low-LR recovery、KD、threshold 或 ranking loss；所有增益仍是待驗證假說。
4. 現有多數 BinaryQK／MASF結果只有一至兩個 seeds；千分位差異不足以支持穩定優越主張。
5. YOLO11 與 YOLO26 evaluator、attention sites、task lineage 不同，不能把前者的恢復幅度當成後者保證。
6. RepConv 的 post-fuse PTQ/QAT 尚未測；QARepVGG 顯示結構重參數化可能放大量化誤差，必須另過 matched quantization gate。
7. 目前 BinaryQK 主要是 fake quant／accuracy simulation；沒有 bit-packed/custom kernel 與端到端 profiling 前，不宣稱實際加速。
