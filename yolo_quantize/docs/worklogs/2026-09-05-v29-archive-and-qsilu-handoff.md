# 2026-09-05：V29安全封存與qSiLU接手

## 變更內容與原因

- 依使用者新決定，停止原V29 blocking monitor與舊deferred handoff，不讓兩個supervisor重複啟動。
- 等Detect predictor Fixed-SD4 matched-sham完成epoch 1 checkpoint並建立epoch 2 control後，對精確V29 process group發送SIGINT。原Queue、checkpoint、metrics與stale execution-state沒有搬動或改寫。
- 建立artifacts/archives/v29-poly-shift-paused-for-qsilu-v1/archive-handoff.json，記錄source、state、status、第一組完成checkpoint及第二組resume checkpoint的SHA-256。封存manifest SHA-256為ce73d390f2c62421d7c98a26e5b46e980d89bc5443641bb7956e87031cdffb17。
- V30 qSiLU plan改以此不可變archive handoff為dependency，required status是archived_for_qsilu_handoff；runtime新增dependency SHA比對，避免把V29偽標為complete或接受被改寫的handoff。
- 新增V29七組來源與qSiLU實驗報告，以及Q3證據約束的regional Hardswish條件計畫。Hardswish不混入目前qSiLU主線。

## 驗證方式與結果

- V29 supervisor PID與所有GPU compute process已消失；沒有使用SIGKILL。
- 第二組sham的qat-epoch-controls已包含epoch 0、1、2，last.pt在停止前已存在；封存標示completed epoch 1，可由同一reviewed Queue命令resume。
- 第一組Pose tower exact ternary paired QAT合法完成：QAT 7 epochs、patience 5 early stop。mAP50 worst每epoch仍在−0.015內，但mAP50-95 worst為−0.0701至−0.0804，沒有best_joint checkpoint。
- archive dependency hash drift regression test已加入；qSiLU Queue定向測試14 passed in 6.17s。
- 全專案CPU測試272 passed in 53.02s；Ruff、compileall與git diff --check通過。
- V30 plan SHA-256為1ae2965674a91b5e75def5c7c3d18d75d2067d0762e669916a3c4548fe10512a；V32 Hardswish條件plan SHA-256為dfb0bc0ad3a3d78e0f772f0d55599788586836de3a60c279da11fea5aa1aa48e。唯一YAML鍵、archive、qSiLU checkpoint及三份Q3證據SHA均通過。
- V30 qSiLU Queue已啟動，初始事件為q1_started、candidate qsilu-all-ten-regions-w8、attempt 0；supervisor session為83729，事件監測檔為artifacts/queues/v30-qsilu-complete-quantization-lane-v1/status.json。
- 未重新切分或改動BBAT5；未啟動formal、multi-seed、長epoch或硬體量測。

## 困難與解法

- 困難：V29 execution-state保留先前失敗後續跑造成的stale failed/running欄位，不能把它直接改成complete。
- 解法：另建references-only archive handoff並hash pin；舊state保持原貌，可追查也可續跑。
- 困難：第一組QAT沒有best_joint.pt，最初封存器依假設尋找該檔而fail closed。
- 解法：唯讀核對qat-experiment.json後，改封存實際存在的best_detect、best_pose與last，並明記未通過joint gate；沒有捏造checkpoint。
- 困難：初次更新後的full-lane CPU test仍寫舊dependency status，造成fixture等待。
- 解法：停止CPU-only pytest，修正fixture為archived_for_qsilu_handoff並重跑通過；GPU未受影響。
- 困難：apply_patch持續因bwrap loopback namespace失敗。
- 解法：先依規範嘗試後，使用限定workspace且每個sentinel必須唯一的精確替換；新artifact採atomic temporary replace。

## 未解事項或風險

- qSiLU all-ten W8歷史上只有九區通過；backbone-attention isolated W8為recover，Q1可能不會直接green。
- 若Q2沒有任何best_joint同時通過雙門檻，Queue必須停止，不可只看mAP50而忽略mAP50-95。
- Hardswish的Q3優勢只有neck-attention +0.000229，未證明超過量測波動；因此不自動執行其完整weight matrix。
- V29剩餘工作未完成但可續跑；目前只改優先順序，不宣稱poly_shift整體失敗。
