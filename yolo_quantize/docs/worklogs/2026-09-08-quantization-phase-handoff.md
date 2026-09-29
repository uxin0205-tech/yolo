# 量化延後：安全階段收尾

## 變更內容與原因

使用者依老師建議，將量化延後至模型最後階段；本輪既定 PTQ 完成後收尾，不再執行先前考慮的額外兩組 QAT。依 finish-work 技能將執行完成、研究未完成、保留證據與後續授權分開，不為封存擅自刪檔或發布。

已讓原有有限 PTQ 自然完成，monitor 回傳 `decision_required,current_index=11,current_candidate=select_max_two_cumulative_qat,current_arm=analysis,completed_jobs=12,error=null`；supervisor session 25632 確認 exit code 0，monitor 自然退出。未停止 GPU child，未啟動下一批；不宣稱其他使用者的 GPU 程序不存在。

新增階段成果報告與 `phase-hold.json`；同步 README、CURRENT_PLAN、報告／工作紀錄索引、設定／腳本／產物及兩個 queue 入口。原始 `plan.json`、`execution-status.json`、結果 JSON 和程式皆不改寫。新 hold 是人／agent 的交接決策，不是 CLI 強制鎖；原 runner 已退出，沒有自動下一批 QAT。

## 已完成結果

592 單層輸出 probes、40 區域 PTQ、六組各 5-epoch QAT，以及本日 12 累積 PTQ 均有實際結果。最後五個追加候選中，Pose tower SD4 接受；Detect predictor SD4、Pose predictor Paper-TWN、W6／W5 廣泛追加未接受。Pose predictor 的 mAP50 尚可但 mAP50–95 超標，不能僅用 mAP50 判斷。

保留兩個已測取捨點：early／neck SD4＋MASF TWN（最差總下降 1.403／1.553 pp，估算減少 4.61%）；再加 Pose tower SD4（1.403／2.954 pp，估算減少 9.40%）。成本只含 codes＋scales，非檔案大小或硬體加速；兩者都不是已選定的最終部署模型。

## 驗證方式與结果

- 只讀 Python 稽核：43 個既有來源 hash、六組 completion 的 5 epochs、30 回合 480 個有效指標。
- 只讀累積稽核：13 個 stage／plan／reference hash、parent manifest hash、12 個不同原始 metrics report hash；12 候選均 148 權重／124 activation quantizers。
- 192 項相對 accepted 的差值獨立重算一致，最差值及雙門檻一致；沒有 NaN／Inf／缺項。
- `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/uxin/yolo/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_cumulative_ptq_queue.py`：10 passed in 0.03s。
- 先前完整 351 tests 通過；本次只有文件及交接決策修改，不重新宣称全域 lint／format 舊問題已修。
- 最終文件／hold 檢查：14 份 Markdown、112 個本機連結，無失效連結或尾端空白；12 列累積結果逐列比對原始數值通過；7 個 blueprint 來源 pin 及原始 plan／status／ledger hash 未變。交接檔明定新增 GPU=false、額外 QAT=0、需使用者明確要求才恢復。
- 所有入口修改後再跑同一 CPU 回歸：10 passed in 0.02s。沒有重啟 monitor 或 GPU queue。

## 困難、解法與未解事項

研究負結果不是執行故障，未為提升分數改門檻或重跑。舊 runner 終態文字仍要求挑 QAT，故保留原始終態並另建明確延後決策、同步人讀入口，避免誤啟動；未修改 source-pinned runner。其他困難無。

未完成：新增累積 QAT、完整逐層 mAP、finalists、export／formal、真實整數部署與硬體測量。全部延後，需使用者明確要求再接續；若最終 parent 改變，既有結果不可直接當新模型精度。本次不更換主線 checkpoint，也不將封存解讀為撤銷已有 activation／量化配置。

清理與發行：保留全部現行依賴與原始證據；未執行待核准舊報告清除，未刪除任何新檔案，未 commit／push。
