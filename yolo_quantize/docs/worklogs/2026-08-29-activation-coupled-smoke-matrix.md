# 2026-08-29：Full35 activation耦合adapter與30格無訓練smoke

## 任務與範圍

依使用者確認重新以現有yolo_activation與yolo_combine/final/full35為來源規劃及測試；沒有yolo_activation2。先分析activation與quantization耦合，再實作可測的Full35 activation-output adapter與A3至A8 GPU smoke。本次依使用者指示不進正式validation、QAT或長epoch訓練，也不實作weight量化實驗。

## 變更內容與原因

1. 將intake拆成screening與finalization兩層。上游finalist中斷只阻擋正式結論，不再阻擋已取得completed evidence後的無訓練screening。
2. 將qSiLU、poly_quality、poly_shift、Hardswish依使用者接受的三任務0.04 absolute mAP gate列為screen candidates；保留SiLU control。qSiLU通過上游舊0.015 gate，但不預設為量化winner。
3. 新增CoupledExperimentPlanner，policy ID同時包含activation與A-bit；只有完整activation policy可進未來weight matrix，禁止事後拼接獨立winner。
4. 新增ActivationOutputAdapter，以clone-by-default、完整preflight與transactional wrapping包裝已審核activation leaf；支援observe、disable quantization、freeze observers、LSQ+ fake quant與state-dict roundtrip。
5. 新增Full35ActivationAdapter，重用yolo_activation的190-site replacement manifest與accepted Full35 loader，保護Binary Q/K、MASF、attention PWL、RealNVP及Detect／Pose heads。
6. 依真實inference路徑把output quantization縮成124個deployment sites；66個detect_one2many、pose_one2many及pose_flow training-only sites不量化。
7. 新增無訓練GPU smoke runner。每格先用canonical train exemplar觀察range，再對canonical val exemplar比較matched FP與fake quant，分開保存aggregate、decoded、one-to-one raw及TopK overlap。
8. 執行5個activation乘A3至A8共30格；輸出原子JSON與CSV摘要，支援resume。
9. 新增machine-readable smoke config、中文分析報告，並同步README、實作計畫及工作紀錄索引。

## 驗證方式與結果

- TDD：先為intake gate、coupled matrix、LSQ+ range初始化、transactional adapter、state-dict roundtrip與Full35真實契約建立失敗測試，再補實作。
- Full35 adapter契約測試通過：124個部署quantizers、66個training-only exclusions、受保護class signature不變；graph仍為23 shared layers、三尺度heads、reg_max 1、end2end true與RealNVP。
- GPU smoke：NVIDIA GeForce RTX 5090、PyTorch 2.11.0＋CUDA 12.8；30 planned、30 passed、0 failed。
- 所有格的Detect／Pose輸出結構相同、數值有限，124個observer都取得有效range。
- A8 worst raw NRMSE與minimum TopK pair overlap：SiLU .0911／.680、qSiLU .1019／.613、poly_quality .0816／.693、poly_shift .0989／.687、Hardswish .1194／.617。
- resume smoke：5個A8格全部skip既有passed結果，最後5 passed、0 failed。
- 完整CPU suite：34 passed in 8.00s。
- ruff check通過；ruff format --check回報16 files already formatted。
- compileall、兩份YAML解析、smoke JSON 30格不變量與JSON／CSV SHA-256比對全部通過。
- 真實intake CLI：screening gate退出碼0；finalization gate退出碼2，符合目前上游狀態。
- BBAT5 assignment_changed=false；沒有建立split、30% view、dataset副本或改動label。
- 沒有執行optimizer、backward、epoch、正式validation、QAT或weight quantization。

## 困難與解法

- 困難：使用者先前提到yolo_activation2，但工作樹不存在該目錄。
- 解法：取消該依賴，只以可驗證的yolo_activation evidence與Full35 bundle重建計畫。
- 困難：第一次CLI收尾驗證漏帶PYTHONPATH=src，兩個命令都因套件匯入失敗回傳1。
- 解法：改用README記錄的完整命令重跑，取得預期screening 0與finalization 2。
- 困難：內建apply_patch及一般sandbox命令持續遇到bwrap loopback RTM_NEWADDR錯誤。
- 解法：先嘗試apply_patch並保留失敗證據；經核准後使用受控的精確字串替換與新檔寫入，只修改yolo_quantize。
- 困難：PyTorch 2.11的reset_peak_memory_stats不接受torch.device，且Full35 forward參數是task而非tasks。
- 解法：以實際runtime signature與最小GPU smoke確認，分別改用device index 0與task參數。
- 困難：190個observer中有24個pose_flow在inference沒有執行，後續又確認one-to-many同樣是training-only。
- 解法：把deployment output quantization明確縮為124個位置，66個training-only位置保留activation函數但不插入部署quantizer。
- 困難：decoded NRMSE因TopK index切換而不隨bit單調。
- 解法：升級為v2比較契約，分開量測decoded、one-to-one raw、selected pair overlap與anchor Jaccard。

## 未解事項或風險

- finalist queue仍pending／interrupted，finalization snapshot與正式activation結論未完成。
- smoke樣本非常小且採min／max初始化，只能作工程與敏感度proxy，不能代表完整mAP。
- qSiLU與poly_quality都不能據此宣稱winner；A3／A4雖可執行，但TopK幾乎崩潰。
- 尚未比較MSE observer初始化、完整validation、QAT、INT8／INT4／SD4／ternary weights、export或板上硬體。
- 0.04 gate是每task absolute mAP門檻；本次NRMSE不可代替。
- 正式訓練依使用者指示保持未啟動。
