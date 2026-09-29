# 各階段精度與部署成本比較

本區以既有 checkpoint 補測推論成本，不重新訓練、不更改資料集，不把訓練 peak memory 或 macro 時間冒充推論量測。

輸入由 `manifest.json` 固定 26 組代表與配對，包括 FP／A0／B100、HOG、RepConv、P3／P2 MASF、舊新融合、activation、KD 及推論分支。每筆保存來源 SHA、既有 AP50–95、指標來源與角色。輸出為 `artifacts/comparison-v1/` 的 JSON、CSV／報告及失敗 log；結果不覆寫，已完成 job 不重跑。

## 固定量測契約

- 單張 `1×3×640×640`、FP32、合成 uniform 輸入、seed 50900912。CPU 固定 4 threads／interop 1；eager、eval、inference_mode，不使用 autocast、compile 或額外 fuse，TF32 關閉。
- Params 是重建後模型的唯一 registered parameters；包含仍保留但未執行的 one2many 權重。另列 tensor bytes、buffer bytes 與原 checkpoint bytes，訓練快照不稱部署檔案大小。
- HOG 輔助頭移除後量測；RepConv 折疊後量測且做 CPU160 數值比對。舊 A0／B100 保留 checkpoint 原有 backend；其餘依各階段驗證方法重建。
- MAC 使用實際 640 forward 的 Conv 與浮點 mm／bmm／addmm 計数；FLOPs subtotal = 2×MAC subtotal。BinaryQK 的位元乘積另列，不假設為浮點 MAC。BN、activation、PWL、reciprocal、pooling、排序與記憶體流量不在此 subtotal，因此不是全算子 FLOPs。
- CPU 暖機 2 次後量 10 次；GPU 暖機 10 次後量 50 次，使用 CUDA events 及同步 wall time，保存每次樣本、median、mean、P90。包含模型內部 decode／top-k，不包含資料讀取、H2D 或外部 NMS。
- GPU peak allocated／reserved 在暖機後重置計數，包含 resident 模型及輸入。CPU RSS 為整個 worker 的 process 高水位，含模型載入，不稱純 activation peak。
- GPU energy/frame 使用 NVML 累計 mJ 差除以實際完成 frame 數，3 組各至少 2 秒、至少 10 frames。這是整張 GPU 的遙測能量，包括 idle／桌面其他活動；不是整機或目標 FPGA 能耗，也不使用 TDP 推算。

## 執行與監測

`build_manifest.py` 建立一次性來源清單；`measure.py` 執行單一案例；`run_queue.py` 順序執行並以 600 秒等待週期監測。正常不讀 log、不反覆查 GPU。錯誤停止於該 job，修正後重啟 queue 只執行尚未成功者。

正式結果引用完整 COCO／BBAT AP，合成輸入只用於成本測試，不是重切資料或以少數影像取代精度評估。Detect、Pose 與共享雙 head 的時間不可混作相同工作量。one2many 另需外部 NMS，core-only 時間不能代表其完整部署延遲。

目標板卡未指定或沒有可連線設備時，target hardware latency 與 target energy/frame 留空並記錄原因；不複製 GPU 欄位充數。硬體量測是否需加入由使用者設備資訊決定。

## 官方方法依據

[PyTorch peak memory 定義](https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html)限定 tensor allocator，不能代替整機顯存；[NVML energy counter 定義](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html)是 driver 啟動以來的 GPU 累計 mJ，因此採差值並註明遙測邊界。

Git 只發布量測程式、manifest、JSON／CSV 與報告；不包含 checkpoint、資料集或環境 binary。原始失敗 log 本機保留。

## 完成狀態

26 組正式量測及 4 組受干擾案例的隔離補測均完成，沒有 GPU 工作待續。MAC 正文統一使用 accounting-v2；初版 inference_mode 計數保留供追溯但不採用。量測工具的預設計數也已改用修正版。

[完整結果](<../../reports/performance/README.md>)／[原始 queue 狀態](<artifacts/comparison-v1/summary.json>)／[隔離補測](<artifacts/isolated-recheck-v1/summary.json>)。
