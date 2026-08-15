# 正式訓練稽核

稽核日期：2026-08-15

> 本稽核在 artifact 清理前完成。17 個 accepted checkpoint 當時均通過 load/finite/config 檢查；清理後只保留五個階段 winner checkpoint，詳見 `CLEANUP.md`。

## 結論

目前 Queue 接受的 17 個 training run 通過機械完整性檢查：全部為 `succeeded`，每個 run 都有 manifest、variant/training YAML、`results.csv`、formal metrics、analytical profile、`best.pt` 與 `last.pt`；所有 CSV 數值及 checkpoint 浮點 tensor 都是有限值。17 個 `best.pt` 均可在 CPU 載入，且兩個固定路徑都是 `HardwareFriendlyAttention`：

- `model.10.m.0.attn`
- `model.22.m.0.1.attn`

兩處 Attention 的 embedded `VariantConfig` 都與 run 的 `variant.yaml` 一致；全部 BDCN checkpoint 也保留一個跨兩處 Attention 共用的 codebook bank。

這代表目前接受的 checkpoint 沒有缺檔、NaN 權重、單點 Attention 替換或 variant mismatch。它不代表所有方法已充分收斂，也不代表 seed 穩定、官方 COCO API 已驗證或硬體效能已實測。

## 共用訓練條件

- 模型：YOLO26m，官方初始權重 SHA-256 `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`。
- 資料：完整 COCO2017 train2017，118,287 張；val2017，5,000 張。
- 實際設定：imgsz 640、batch 16、workers 8、AdamW、AMP、`fraction=1.0`、deterministic。
- Seed：除 `d1-seed1` 外全部為 0；`d1-seed1` 為 1。
- 裝置：單張 NVIDIA GeForce RTX 5090。
- Python 3.12.3、PyTorch 2.11.0+cu128、Ultralytics 8.4.90。

實際 import 的 Ultralytics 來源是 `/home/uxin/yolo/yolo_p2/ultralytics`。該 tree 相對 `v8.4.90` 只修改 Detect 可選的 `cls_channels` 與 P2 YAML；標準 YOLO26m 沒有提供第二個 Detect argument，因此這個分支不啟用。不過既有 manifest 只記錄版本字串，沒有記錄 `ultralytics.__file__` 或 runtime source commit，這是 provenance 限制，未來 manifest 應補記。

## Run-by-run 稽核

「完成」以目前接受的 `results.csv` 行數計算；W-DIR/W-PROG 因 `patience: 5` 正常 early stop，並非 crash。

| Run | Stage | Seed | 完成/規劃 epoch | Best epoch | 訓練內 best mAP | Formal mAP | 稽核 |
|---|---|---:|---:|---:|---:|---:|---|
| I-SCR | screening | 0 | 10/10 | 7 | 0.492950 | 0.492482 | pass |
| H-SCR | screening | 0 | 10/10 | 10 | 0.495390 | 0.495243 | pass |
| T5-SCR | screening | 0 | 10/10 | 9 | 0.492940 | 0.492907 | pass |
| W-DIR | recovery | 0 | 26/40 | 21 | 0.507540 | 0.507457 | pass，early stop |
| W-PROG | recovery | 0 | 11/40 | 6 | 0.503260 | 0.502866 | pass，early stop |
| V1-B0 | bias | 0 | 5/5 | 3 | 0.505380 | 0.505416 | pass |
| V1-BD | bias | 0 | 5/5 | 3 | 0.506430 | 0.506877 | pass |
| V1-BR | bias | 0 | 5/5 | 4 | 0.506720 | 0.506658 | pass |
| N1-PWL | normalization | 0 | 5/5 | 2 | 0.506180 | 0.506268 | pass |
| N1-SHIFT | normalization | 0 | 5/5 | 2 | 0.506320 | 0.506357 | pass |
| D1-SHARED | codebook | 0 | 5/5 | 5 | 0.499050 | 0.498984 | pass |
| D1-PATTN | codebook | 0 | 5/5 | 5 | 0.499060 | 0.499126 | pass |
| D1-PHEAD | codebook | 0 | 5/5 | 5 | 0.498770 | 0.498880 | pass |
| D1-SHARED-10 | staged extension | 0 | 5/5 | 5 | 0.499790 | 0.500098 | pass；總設計為 staged 5+5 |
| D1-PATTN-10 | staged extension | 0 | 5/5 | 5 | 0.499750 | 0.500055 | pass；總設計為 staged 5+5 |
| D1-PHEAD-10 | staged extension | 0 | 5/5 | 5 | 0.499670 | 0.499842 | pass；總設計為 staged 5+5 |
| D1-SEED1 | codebook confirmation | 1 | 10/10 | 10 | 0.474700 | 0.475014 | pass，但 seed gap 很大 |

## 訓練量

正式接受 histories 合計 132 epochs、15,613,884 次訓練影像 presentation、975,876 個 mini-batch 與 660,000 次 epoch-level val image evaluation。各 CSV `time` 最後值加總為 95,343.2 秒，即約 26.48 單卡程序小時；它包含資料載入、每 epoch validation 與 checkpoint I/O，不是純 CUDA kernel 時間或耗電量。

## 需要揭露的問題

1. W-DIR 在 26/40、W-PROG 在 11/40 epochs early stop。兩者都有完整 best checkpoint，但不能寫成跑滿 40 epochs。
2. D1-SHARED seed 0/1 formal mAP 相差 0.025084，即 2.508 AP。這表示 codebook-only branch 的 seed 穩定性不足；只做兩個 seed，不能宣稱穩健統計結論。
3. N1-SHIFT formal mAP 比 zero-train N0-SHIFT 低 0.000390。N1 recovery 沒有帶來提升，故不能宣稱 PMP recovery 必要。
4. 歷史 worker log 曾包含 NaN/exception run；它們屬 rewind 前失效流程。清理前已確認目前接受 run 的 CSV/checkpoint 為有限值；之後 invalidated artifacts 與 worker logs 已依核准永久刪除，queue events 與正式報告仍保留 provenance。
5. `i-scr` manifest 的研究程式 revision 是 `7660e7c...`，其餘 16 個 training run 是 `949d819...`。I-SCR 仍通過 checkpoint/config 稽核，但跨 revision 是比較限制。
6. 現有 manifest 未保存 runtime Ultralytics source path/commit、GPU 型號、CUDA driver 與資料 cache hash。版本可重建性尚未達到完整封存等級。
7. Evaluate-only run 沒有和 training run 相同完整的 manifest/variant copy；原始 variant 位於 `artifacts/queue/generated/`。Queue state 足以追溯本次結果，但 artifact contract 後續應補強。

## 尚未完成

- 官方 COCO API/faster-coco-eval canonical evaluation。
- 至少三個 seeds 的正式重現性統計。
- A-FINAL 最新 BN-fold/export/bit-true artifact。
- GPU latency、VRAM benchmark、FPGA/HLS/板端 latency、energy、BRAM/DSP。
- Optional Phase Q 的 PTQ/QAT 與 mixed precision。
