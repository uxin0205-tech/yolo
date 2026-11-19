# 2026-10-18：Binary 狀態、0.51～0.52 精度與恢復配方稽核

## 任務與範圍

回應使用者要求參考 optimize，並核對其記憶中的初始精度 0.51～0.52。採單一主代理，只修改 `yolo_binaryqk` 與根層交接文件；`yolo_optimize` 及歷史 Attention 來源唯讀。未啟動其他研究階段。

## 變更內容與原因

- 確認 B1 於 05:07:39 完成 10 epochs，官方 AP50–95 0.440541、internal 0.428949；B2 於 05:07:59 因其他 GPU compute process 在 preflight 退出，尚未開始訓練，B3 未啟動。原 failed 狀態與 log 保留，不改寫成成功。
- 核對初次二值化與 optimize 後期 A0 恢復的來源。舊 H-SCR 為 Attention 適應、LR `5e-5`、10 epochs，W-DIR 接全模型恢復、LR `5e-5`、26/40 epochs early stop，internal 0.507457。Optimize 已訓 A0 為 0.506739，其後期 Q/K `5e-7` 還搭配 head `2.5e-5`。本輪把該後期配方限縮為 Q/K-only 並套用初次替換，引用範圍不當。
- 確認舊 H-SCR/W-DIR 使用 dynamic scale 與可學 gamma；本輪一開始固定校準係數。保留此差異，尚不把精度落差全歸因於 LR 或 scope。
- 在 GPU 空閒時只執行原始 parent 的完整驗證。重現 internal **0.518001594**、官方 COCO bbox **0.524685533**，確認使用者記憶及本輪 evaluator 能重現 parent 精度。沒有新增 baseline 加訓。
- 停止延續舊隊列，新增 `STOP` 與 `continuation-decision.json`；保留 B1 最佳權重、resume、曲線。修正版三候選需共同 parent、共同配方與新 run ID，不能混用多訓 10 epochs 的 B1 作公平比較。
- 修正排程控制：GPU 占用用可辨識的 `GPUInUseError`／exit 75；等待每 600 秒重查，STOP 可中斷；`--continue-queue` 驗證完成結果及 checkpoint，未完 run 走 trainer resume 驗證；file lock 防雙重排程；處理 child 啟動競態與舊 status mtime。訓練數學與舊 `RECIPE` 本次未更動，沒有重啟長訓。
- 保存 [稽核報告](../../yolo_binaryqk/docs/protocol-audit-2026-10-18.md)、精確 CSV、曲線及來源 SHA；同步根 README、專項 README、整合計畫、cleanup 接續狀態、舊工作紀錄更正與本索引。

## 驗證方式與結果

`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=yolo_binaryqk .venv/bin/python -m pytest yolo_binaryqk/tests -q`：**38 passed in 3.85s**。Ruff 檢查 `yolo_binaryqk/binaryqk`、`yolo_binaryqk/tests` 全部通過。新增 5 個測試涵蓋資源等待、STOP、結果完整性、重試及排程鎖。另以實際 B1 最佳權重確認 strict reload，可識別為已完成候選。

文件收尾檢查：本次 10 份文件的 122 個本機連結全部有效；稽核 JSON 所列 8 份唯讀方法／結果來源 SHA 全部維持一致。

原始 parent 使用 `binaryqk.model.load_parent`、`binaryqk.data.dataset('val', 8)` 與 `binaryqk.evaluate.evaluate`，E2E FP32、640、全 5,000 張，約 33 秒模型評估；只佔用本次短評估，程序已結束。完整版本、時間、指標見 [parent-evaluation.json](../../yolo_binaryqk/artifacts/stage1/protocol-audit-20261018/parent-evaluation.json)。未聲稱精度問題已修復，未驗證新版恢復訓練效果。

資料為 `/home/uxin/yolo/coco2017` 的 COCO80 Detect，val2017 5,000 張；runtime View 保留完整 membership、labels 與來源 hash。沒有修改 split、影像、標註或 BBAT5 v1。原始 parent SHA 與既定來源一致。

## 困難與解法

1. 使用者提到的 0.51～0.52 與本輪先前回報的 0.4405 可能混用 internal／official。改為分欄，並實測共同 parent；確認口徑差異存在，但不能解釋 B1 的大部分落差。
2. 低 LR 來源是已恢復 A0，而非首次轉換。補查實際 YAML、freeze scope 與上游血緣，停止盲目延續，將舊結果標為受限診斷。
3. 舊 supervisor 將 GPU 占用當成訓練失敗。新增等待與續接控制，測試通過；不因修好排程便重啟已不適用的配方，也不終止他人的 GPU 程序。
4. 一般 sandbox 仍有 `bwrap` loopback 權限問題，透過工具核准的工作區指令完成讀取、測試與短評估，未修改系統網路設定。

## 未解事項或風險

尚未拆出 scope、LR、固定 scale 各自造成的精度損失；新版訓練配方、短測與正式恢復仍未完成。38 項測試確認程式契約，不能證明 AP 恢復。舊結果只有單 seed，歷史模型未全部用本輪 evaluator 重跑。

目前無背景長訓，舊 ETA 失效。第 1 階段尚未完成，B2／B3 沒有正式訓練結果，未選 winner，沒有啟動第 2 階段。後續應從已訓 parent 做恢復適應，不能保證達到 0.51；RepConv 仍須繼承 qSiLU 與完整硬體友善契約。本次沒有額外清理資料夾、commit 或發布。

## 12:19 GPU 狀態查詢

2026-10-18 12:19:06（Asia/Taipei），依使用者詢問唯讀查詢 `nvidia-smi`：RTX 5090 使用率 0%，顯存 611／32,607 MiB，compute process 清單為空；原 supervisor 與已知訓練 PID 均不在執行。舊隊列仍為 B2 preflight failed，沒有重啟長訓。此為查詢當下狀態，不能保證之後其他工作不占用。困難：無；未修改模型或資料。
