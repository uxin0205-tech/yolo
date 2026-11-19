# 2026-10-31：第六階段清理與第七階段準備

## 任務與範圍

依使用者「整理好、刪除沒用的內容、準備下一階段」要求，清理第六階段已完成研究的中間產物，保留唯一選定模型與必要證據，並在 `yolo_activation/stage7/` 準備 activation 研究。只進行 CPU 檢查，沒有啟動 GPU、正式 AP、訓練或 RepConv。

沿用 `finish-work` 與 YOLO 工作流程，使用者已明確授權直接清理，不另外封存或重複要求確認。全程單一主代理，未 commit／push。

## 清理內容與原因

清理前 Stage6 有 18 份實體 `.pt`，合計約 2.809 GiB。先檢查各 supervisor 均已停止，搜尋 optimize、activation、architecture、quantize、attention、binaryqk、masf 的程式及配置，未找到這些 Stage6 中間 checkpoint 的外部使用者。另檢查目前使用者的開啟檔案，沒有程序持有刪除目標。

刪除前將 19 個項目的完整路徑、大小、SHA、原因與可恢復性寫入 [cleanup-20261031.json](../../yolo_combine/result/stage6/cleanup-20261031.json)。實際直接移除 17 份中間權重／optimizer 與 2 份 runtime labels cache，釋放 **2.7115 GiB**；所有刪除目標都只有一個 hard link。未建立 archive／backup，沒有移到別處假裝釋放空間。

移除範圍包括 H0／H1／P／M0／J1 已停止的 resume、各中間最佳、三個 rho 初始化、J1 重現／回退權重、KD E2，以及最終 head 的 E5 resume。E5 不是選定 E1 的 exact resume，下一階段從 E1 建立新 optimizer，無需保留它。未選權重位元組不可直接恢復，若要重建須依保留配方重訓；不宣稱仍能從舊 optimizer 接續。

唯一保留實體模型是 `yolo_combine/result/stage6/pose-head-recovery/train/best-e001.pt`，由 `selected.pt` 入口引用。SHA 前後均為 `e9b9fd7ccc88f10bbd170835138ebde5c7810db0726424496e2cac2aa462ffe7`。保留各段 AP、history、來源鎖、配方、sidecar、圖表與重畫程式，小型證據沒有當成另一套可執行權重保存。

`handoff.json` 已將 `resume` 設為空，另外保存原 E5 的 SHA 與移除原因；新增清理入口及下一階段準備狀態。歷史 run JSON 及內嵌 lineage 原樣保留，其中舊 checkpoint 路徑代表來源歷史，不表示清理後仍存在。未修改選定 checkpoint 或其 sidecar。

受保護 Detect／Pose 原來源、MASF 結果、BBAT 正式影像與標註，以及 optimize 的舊 runtime 依賴均保持。只刪除 Stage6 runtime View 的兩份 cache，未改 `/home/uxin/yolo/original/pose/derived/bbat5-v1/` 或 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。

## 第七階段準備

先唯讀參考 optimize activation bridge 的原始實作、實際 resolved config、10 輪 qSiLU summary、零樣本結果、上游 J3 與完整 YAML，再保存 [reference-review.json](../../yolo_activation/stage7/reference-review.json)。其選定 qSiLU E2 SHA 為 `1bbbbe0f87cb853827af1bb625e7066178e3fb2c7e14c410d997efb24e94834a`；已核對檔案，不將舊結果冒稱本輪結果。

保留 qSiLU 固定分段係數、先診斷再必要恢復的流程，以及 fresh optimizer／EMA、warmup／cosine 的分段方式。本輪不重跑 SiLU control，不新增 ReLU；正式候選只有 qSiLU、Hardswish、PolyShift。訓練 scope、BN、task weights、LR 與預算需依新 parent 診斷後固定，不直接照搬 optimize 的 Attention 小 LR／MASF 設定。

新增 `prepare.py`、`check.sh` 與中文 README，只讀 Stage6 parent 並盤點 activation references、alias、真實前向位置及硬體契約。CPU 160×160 FP32 結果：187 條 SiLU 註冊路徑、30 個不同物件；both／Detect／Pose 分別執行 163／115／127 個位置。24 個 RealNVP SiLU 為訓練 loss 專用，eval 均未執行。qSiLU 類別及固定係數已核對，但尚未安裝到模型。

Stage6 loader 會拒絕 activation 變更，已記明 Stage7 必須建立新 manifest／載入契約，不可竄改 Stage6 sidecar。繼承 B2、S1 Q10／bias、D1 完整整數 PWL＋LUT256 reciprocal；RepConv 階段所有候選都必須由已驗證 qSiLU parent 出發。

activation 的舊目錄另有歷史結果與共用程式；此次沒有將未遷移的整個資料夾清空。`activation_lab` 目前仍被 optimize 直接 import，正式 Stage7 重建前須按依賴清理，不能破壞唯讀 optimize 的執行能力。此次 CPU 準備程式放在獨立 `stage7/`，不改舊來源。

## 驗證方式與結果

- 清理前已載入選定模型，strict state／manifest／SHA 通過；確認模型重建不讀取 Stage6 中間權重。
- 清理後重新從 `selected.pt` 載入，並執行 CPU 單／雙任務前向，核對輸出逐值相同、有限值及硬體契約。
- 逐檔核對 19 項移除，剩餘實體 Stage6 checkpoint 僅 1 份；未產生 checkpoint 副本。
- 新準備腳本 Ruff、格式及 shell 語法檢查通過；29 個文件連結、15 個來源 SHA 與原 AP 證據核對通過。Stage6 實體內容約剩 111.45 MiB，含唯一權重與小型證據。正式資料 registry 及 GPU 狀態核對結果存於 [清理後驗證](../../yolo_combine/result/stage6/cleanup-verification-20261031.json)，沒有 GPU compute process。
- AP 沿用上一輪已通過的完整重載結果，因本次沒有修改權重，不重新消耗 GPU 評估。

## 困難與解法

初次 activation 盤點假設所有 SiLU 都在 Conv.act，檢查主動拒絕 Pose26 RealNVP 的 Sequential SiLU。唯讀核對本地 `RealNVP` 實作後，補入來源 SHA；Conv 使用唯一 owner 的 hook，RealNVP 的獨立 SiLU 直接加 hook。再次檢查通過，避免把訓練 loss 算進推論成本或漏掉註冊位置。

先前 `handoff.json` 仍指向 E5 resume；此次清理同步移除活動入口並記錄舊 SHA。舊歷史結果不回寫，避免破壞來源鎖與實驗可追溯性。

## 未解事項或風險

Stage6 原四項精度目標仍未達，沒有因清理或準備下一階段就改成通過。Stage7 尚需目標資料夾依賴遷移、候選 manifest／loader、完整數值診斷、正式 AP 與必要恢復訓練。qSiLU 與 RepConv 尚未加入目前模型；本次未做硬體綜合、板端量測或長方形特徵圖研究。
