# 2026-08-27：architecture_2 C2／C3完整訓練與量化自動接續

## 任務與範圍

- 依使用者明確授權，讓既有Float20完成後自動判定C2／C3，接續完整資料訓練、PTQ與簡單QAT。
- 正式資料仍是完整COCO2017 Detect與不可變BBAT5 v1 Pose；沒有重切split、抽樣或修改影像與標註。
- 本輪簡單QAT定義為200-step QAT-lite fake-quant simulation；完整Q2與部署INT8不在授權範圍。

## 變更內容與原因

- 新增closed正式設定`configs/runs/full35-c2-c3-auto-continuation.yaml`及schema／catalog驗證，集中保存授權、資料、100 epochs、patience20、0.008 gate、校正與queue參數。
- 新增full runner：逐檔驗證Float20匯出manifest；只讓四項主要精度下降均不超過0.008且至少一項成本改善的C2／C3進入full。
- 每個full候選都從同一J3 C0-Handoff fresh建立，不承接Float20權重或optimizer；使用完整COCO與BBAT5 v1、Pose=true，支援每epoch exact resume及共同32→16 OOM fallback。
- 新增Q0／Q1／Q2L runner：Q1各用固定Float20 View的64個Detect與64個Pose batches校正，Q2L固定200 steps、AMP=false、前50 steps更新observer、每50 steps正式validation。
- PTQ與QAT-lite checkpoints延伸完整Float lineage，保存spec、架構／訓練／資料YAML hashes、J3架構parent、直接full parent、handoff與量化policy hash。
- 新增W8A8 route wrapper、fuse等價性檢查、fake-quant scope、逐stage AP／mAP／F1 gap與simulation-only標記。
- 新增最終匯出器：完整Float joint差距不超過0.008時，再依latency、GFLOPs、Macro F1與candidate ID決勝；另列可用q1／q2l或float-only。
- CLI的`full-run`與`quant-run`只接受持鎖`yolo_combine.gpu_queue`；互動shell不能直接解鎖CUDA。
- 新增`scripts/continue_full35_c2_c3_auto.sh`，依序等待Float20 postprocess、判定、排full、排quant並匯出最終中文報告；任一契約漂移都fail closed。
- 啟動`yolo-architecture2-full35-j3-c2-c3-auto-seed0.service`；目前為active／running，不逐epoch輪詢。
- systemd設為`Restart=on-failure`、5分鐘後重試、每小時最多3次，讓full checkpoint可在主機層中斷後resume，同時避免程式錯誤無限重跑。
- 同步更新README、configs README、RUNBOOK、TRAINING_GUIDE與results README，區分Float20歷史報告與run-specific自動授權。

## 驗證方式與結果

- CPU-only完整pytest：90 passed、1 skipped；skip是需另設`ARCHITECHURE_2_HANDOFF=1`的2.35 GB重型materialization，先前正式handoff驗收已完成。
- `ruff check src tests`：全部通過；自動腳本`bash -n`：通過。
- `config-check`：valid=true、Ultralytics 8.4.90、C2／C3 full與Q2L顯示已授權等待gate；完整Q2仍未授權。
- 真實Full35 C2 CPU fake-quant graph smoke已通過：197個quantized conv、6個排除節點，Detect／Pose輸出、strides與kpt_shape契約正確。
- 新測試覆蓋closed config、0.008邊界、成本gate、route wrapper、cycling batches、CLI及最終near-tie選擇。
- systemd確認：`Restart=on-failure`、`RestartUSec=5min`、`StartLimitIntervalUSec=1h`、`StartLimitBurst=3`。
- 正式full、PTQ與QAT-lite GPU數值尚未產生；服務正在等待既有Float20與postprocess完成，故不宣稱accuracy或量化結論。

## 困難與解法

1. 內建`apply_patch`持續因主機bwrap loopback的`RTM_NEWADDR`失敗。
   - 解法：改用受控權限下的最小standard patch，逐段核對並重跑完整pytest、ruff與config-check。
2. 現行Float20 process會在候選啟動時重算`EXPERIMENT_SPEC.md`hash，途中升版會讓尚未開始的候選失敗。
   - 解法：維持2.3.0與目前正式YAML hash不變，將使用者的新執行授權鎖在獨立closed continuation YAML、catalog與本工作紀錄；Float20歷史結果保留原hash，不回溯改寫。
3. PTQ／QAT-lite正式validator與200-step訓練尚不能在GPU繁忙時提前實跑。
   - 解法：完成官方route介面、真實Full35 CPU fake-quant graph與單元測試，正式數值只由通過idle gate的queue產生；若實際契約失敗則有限重試後停止並保留log，不偽造完成。

## 未解事項或風險

- Float20與後處理仍在執行；C2／C3是否通過、full最佳epoch、PTQ／QAT-lite gap與C_best都尚無正式結果。
- 接續服務是目前user manager中的transient unit；本次不中斷即可自動接續，若主機重開機則需依RUNBOOK重新啟動服務。
- `EXPERIMENT_SPEC.md` 2.3.0因active run hash暫時凍結，仍保留舊review gate文字；本次較新的執行授權以closed continuation YAML與本紀錄為準，待既有Float20 lineage封存後再升版，不能回寫舊結果hash。
- Q1／Q2L是fake-quant simulation，不能據此宣稱部署INT8、Bit-True或實機量化latency；完整Q2未排隊。
- 本輪只有seed0與既定預算，最終C_best僅在本次C2／C3比較範圍成立；額外seed仍需未來授權。
- 本次沒有修改BBAT5／COCO來源、影像、labels、split或manifests，也沒有執行commit／push／pull／reset。
