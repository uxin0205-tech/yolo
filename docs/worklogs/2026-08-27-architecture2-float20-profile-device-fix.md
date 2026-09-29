# 2026-08-27：architecture_2 Float20 成果匯出與 profiling 裝置修正

## 任務與範圍

整理 Full35 J3 C0～C3 Float20 實驗成果，修正正式 GPU 成本 profile 的裝置不一致問題，補齊結果匯出、執行防護、中文文件與驗證紀錄。本次不啟動 C2／C3 完整訓練、PTQ 或 QAT-lite，也不修改任何資料集、標註或模型權重。

正式資料來源維持不變：

- COCO Detect：`/home/uxin/yolo/coco2017/`。
- BBAT5 Pose／Detect：`/home/uxin/yolo/original/pose/derived/bbat5-v1/`。
- Full35 handoff：`/home/uxin/yolo/yolo_combine/final/full35/` 的既有 accepted J3 revision。

## 已完成成果

`artifacts/runs/full35-j3-float20-seed0/matrix-complete.json` 顯示 C0、C1、C2、C3 均已完成20 epochs與1,860個macro steps；正式結果已匯出至 `results/full35-j3-float20-seed0/`，manifest狀態為`completed`、`c_best=null`、`selection_status=pending_user_decision`。

| 候選 | COCO mAP50-95 | Pose box mAP50-95 | Keypoint mAP50-95 | Macro F1 | Params降幅 | GFLOPs降幅 | latency降幅 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.604258 | 0.832275 | 0.981708 | 0.959143 | 0.00% | 0.00% | 0.00% |
| C1 | 0.022093 | 0.251601 | 0.456716 | 0.435785 | 7.61% | 5.51% | 1.24% |
| C2 | 0.383706 | 0.655288 | 0.892492 | 0.839558 | 4.45% | 3.19% | 2.53% |
| C3 | 0.217756 | 0.500147 | 0.775751 | 0.714643 | 3.95% | 2.83% | 0.50% |

C2是三個簡化候選中表現最佳者，但其COCO、Pose box、keypoint與Macro F1相對C0分別下降0.220553、0.176987、0.089216與0.119586，全部遠高於0.008門檻。C1與C3更差。因此依目前規格沒有簡化候選取得完整訓練或量化資格，C0仍是精度最佳模型。

## 問題根因與修正

首次正式 profile 在 PyTorch CUDA forward 失敗，錯誤指出同一運算同時收到`cuda:0`與`cpu` tensor。逐步縮小範圍後確認：

1. shared Detect head 的`anchors`與`strides`是在第一次 inference 動態建立的普通屬性，不是 registered buffer。
2. 原流程先在CPU執行GFLOPs推論，因而留下CPU版動態cache。
3. 後續`model.to(cuda)`只搬移參數與registered buffers，不會搬移這兩個cache。
4. GPU latency forward因此混用CUDA feature tensors與CPU cache而失敗。

`src/achitechure_2/profiling.py`新增`_prepare_profile_model()`，先把model搬到最終裝置，再執行Detect／Pose／both GFLOPs，使動態cache直接建立於正確裝置。`tests/test_profiling.py`新增未註冊cache的回歸測試；修正前可重現失敗，修正後通過。

`scripts/continue_full35_j3_float20.sh`也改為可重跑且不覆寫歷史證據：可讀取已封存的成功training state，完整archive直接略過，partial archive拒絕執行。失敗證據保存於：

- `artifacts/queue/history/full35-j3-float20-seed0-attempt2-profile-device-mismatch.json`
- `artifacts/queue/history/full35-j3-float20-seed0-attempt2-profile-device-mismatch.log`

## 後續暫停防護

`configs/runs/full35-c2-c3-auto-continuation.yaml`的狀態改為`paused_pending_user_review`。除了YAML狀態，`FullRunConfig.require_execution_enabled()`也在full matrix、單一full candidate、quant matrix與單一quant candidate入口fail closed；即使服務或命令被誤啟動，也會先拒絕執行。

`README.md`、`configs/README.md`、`RUNBOOK.md`、`TRAINING_GUIDE.md`與`results/README.md`已統一為：Float20完成、C2最佳但未過gate、C2／C3 full與量化等待使用者決策。

## 驗證方式與結果

- 原始GPU最小重現：CPU GFLOPs後再CUDA forward回傳exit 42，`anchors/strides`位於CPU。
- 修正後同一GPU重現：exit 0，`anchors/strides`均位於`cuda:0`。
- 正式postprocess service：`yolo-architecture2-full35-j3-float20-postprocess-retry1.service`成功結束，profile與匯出manifest完整。
- 精準回歸：`13 passed`，涵蓋profiling與暫停後續執行。
- 完整CPU測試：`92 passed, 1 skipped`；skip為需2.35 GB handoff的opt-in integration，先前已獨立驗收。
- `config-check`：valid，Ultralytics固定為8.4.90，顯示C2/C3後續已建置但暫停。
- Ruff：`ruff check src tests`通過，無錯誤。
- 資料變更：無。

## 清理

已刪除16組`/tmp/architecture2-profile-device-*`診斷state／log／lock與臨時重現程式。這些只是可重建的暫存診斷，不含使用者資料；正式失敗證據已保留在`artifacts/queue/history/`。專案內先前patch過程留下的`*.orig`與`*.rej`另於本次整理移除，不影響正式程式、結果或歷史manifest。

## 困難與解法

困難是錯誤只出現在「先做CPU GFLOPs，再做GPU latency」的順序，單獨把模型搬到GPU則正常。透過最小化真實CUDA重現與逐一列印parameter／dynamic cache裝置，確認問題不是checkpoint或一般`model.to()`失效，而是未註冊inference cache。將profile初始化順序固定後解決。

## 未解事項或風險

- Float20是固定20% train-only screening，不使用COCO val2017或BBAT5 formal val，不能直接宣稱正式部署winner。
- C1～C3的best epoch皆接近最後epoch，完整資料或更長預算可能改善；但現有下降遠大於0.008，若要探索C2應先明確建立新的探索規格，不能把原gate改寫成已通過。
- PTQ與QAT-lite尚未執行；目前不存在量化相容性結論。
- QAT-lite即使未來完成也只是W8A8 fake-quant simulation，不等於Bit-True或部署INT8。
- C2／C3後續服務目前保持inactive；只有使用者檢視後明確重新授權，才能改回可執行狀態。
