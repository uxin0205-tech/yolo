# 2026-09-09：第一輪需求與部署邊界稽核

## 變更與原因

依原計畫逐項核對，不把已完成 queue 當成精度恢復成功。使用 finish-work 技能區分實驗、報告、部署、清理與發布的授權邊界；沒有執行刪除、commit 或 push。

新增 `scripts/build_round1_evidence.py` 與 `optimizations/integrated-roadmap/results/`，从真實 summary 建立 6 組訓練、25 個完成 epoch、50 筆 EMA／live 八 AP CSV 和 JSON 證據索引。每筆保留原 summary、parent差值，EMA另列共同epoch native差值。不把不同長度或live/EMA混在一起。

新增 `optimizations/integrated-roadmap/round1-audit.md`，覆蓋 BEST／batch／B／HOG／RepConv／MASF／QK／scale／梯度投影／MuSGD／視覺／部署／監測／延期項目。重寫根 README 及 master plan 過期開頭／結尾，保留原實驗門檻、架構、研究索引與資料規範，詳細階段歷史仍在原工作紀錄。

## 實際驗證

執行 `PYTHONDONTWRITEBYTECODE=1 /home/uxin/yolo/yolo_combine/.venv/bin/python scripts/build_round1_evidence.py`，exit0。每個完成 epoch 的 inference／完整快照存在且非空，八 AP 有限且在0–1、parent路徑／warmup1／physical32吻合。原 BEST SHA256 重算與原紀錄一致。6組最佳 EMA 均未通過相對parent的數字增準門檻，與既有結論一致。

沒有重跑已通過的GPU AP／訓練／無關tests。本輪generator的檔案存在檢查不宣稱重載全部完整快照，數字gate不代替paired seeds或視覺／硬體驗收。新文件及JSON另作路徑／格式檢查。

最後連結檢查發現根 README 的 HOG 研究檔名少了 `to-`，已修正並重跑相關連結／JSON檢查；不重跑模型驗證。

## 部署與下一階段

第三次同一決策阻礙核對：本輪計畫仍為 awaiting_next_phase_decision_no_active_gpu_job，無 recovery／supervisor 程序；量化 phase-hold 仍明確禁止未重新授權的新 GPU／formal／deployment 工作。既定候選已完成或依 gate 停止，沒有可安全自行接續的既有 queue；未收到新的 BinaryQK challenger 階段決策。依連續三回合阻礙規則，將目標標為 blocked（等待使用者決策），不是 complete；精度恢復未達成，所有既有產物與原 BEST 保留。程序篩選 exit1 表示沒有匹配程序，不是 GPU 執行錯誤。其他困難：無。

續行核對（第二次同一決策等待）：實際 `ps` 未找到 recovery／quiet supervisor／MuSGD probe 程序；本專案 active_run 仍為 null。另讀量化 `phase-hold.json` 確認 `state=deferred_by_user`、`new_gpu_jobs=false`、`formal_validation_now=false`、`deployment_optimization_now=false`、`resume_requires_explicit_user_request=true`。這是決策等待，不是 verified GPU wait；沒有新授權，不建立替代 queue、不重跑已完成工作。目標仍未標為完成。

來源 `source_bundle/code/yolo_attention/export.py` 只匯出2個attention site state，並非整個Detect＋Pose模型。不能用它冒稱完成export。

另以唯讀方式確認 `yolo_quantize/README.md` 和 `docs/reports/2026-09-08-quantization-phase-handoff.md`：使用者與老師於2026-09-08決定延後量化，恢復需重新確認；該專案V36 parent亦不同於本輪PSEL。没有恢復其PTQ／QAT，也未將別專案的收益搬入本報告。

目前第一輪既定候選已完成或依事前gate停止，精度恢復仍未達成。若要繼續新的BinaryQK STE／蒸餾訓練契約，或恢復延期量化，應作新的階段決策；依技能要求，不以文件整理代替方法變更授權、不擅自跨過guard。此為首次明確列出需要新階段決策，未將goal標成complete／blocked。

## 清理盤點與風險

只讀盤點：`artifacts/direction1-20260908` 約39G，`artifacts/direction1-20260909` 約17M；全列Keep，因checkpoint、負結果、trace、逐圖資料均被引用。沒有刪除候選或刪除請求。父工作樹有大量無關未追蹤資料，保持原樣，不stage。

困難：README原本疊加多段互相矛盾的「最新」進度，已整合；工作區沒有實體本地AGENTS.md，遵循使用者提供的專案規則及實際全域AGENTS。其他困難：無。未解：精度回升、真實應用案例／人工驗收、新QK契約、延期部署階段與實測硬體效果。
