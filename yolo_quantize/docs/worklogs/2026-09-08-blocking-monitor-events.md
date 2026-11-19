# 600 秒 blocking monitor 與完成事件

## 變更內容及原因

依使用者要求接回既有 shell monitor。等待期間只等待原 shell 的事件輸出，未讀取 source／training log，未輪詢 GPU、未啟停訓練。正常狀態維持靜默，狀態改變後才進行本次整理。

原 monitor 回傳 `status=running_qat`、`current_index=3`、`current_candidate=masf-paper-twn`、`current_arm=qat`、`completed_jobs=635`、`error=null`。635 包含之前 CPU／PTQ 工作，不能說是 635 組 QAT。六組短 QAT 的完成數是 3；MASF exact ternary 結束後，既有 supervisor 已自動啟動 Paper-TWN。

## 結果與推理

既有 CPU 彙整器確認三組完成。MASF exact ternary 共 5 epochs，五個回合的 16 項搜尋指標均通過門檻；第 5 回合最差總下降 mAP50 為 0.949 百分點、mAP50–95 為 1.069 百分點。相對同配置 PTQ，COCO mAP50–95 回升約 0.147 百分點，但 BBAT box mAP50–95 約下降 0.020 百分點，不能宣稱 QAT 全面改善。第 4 回合最差下降 0.829／1.007 百分點，優於末回合的這兩項最差值，但尚未据此選定 export 或 final winner。

本次只更新同 parent 結果報告與 README 快照。Paper-TWN、filterwise TWN 與 Detect predictor LS-SD4 恢復試驗仍由原 queue 依序執行，不插隊、不另建 baseline。六組完成後才分析累積混合方案。

## 驗證、困難與未解項

彙整器輸出 `completed_jobs=3,status=partial`，讀取已完成產物與其雜湊，不讀正在訓練的 log。原 monitor 正常退出碼 0。CPU 回歸驗證結果另補記。

CPU 補驗：`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 ... pytest -q -p no:cacheprovider tests/test_continuous_qat_summary.py`，5 passed（0.01 秒），沒有新增測試快取。原 monitor 已因完成事件退出，接續重啟同一 600 秒監測入口，不新增 GPU 工作。

困難：彙整器沒有 argparse，傳入 `--help` 仍執行預設彙整；本次為已授權的 CPU 報告更新，未影響訓練，後續直接使用無參數命令。其他錯誤無。未完成 formal、export 重驗與累積配置；監測異常或整批結束才進下一次分析。

## 第二次完成事件：Paper-TWN

2026-09-08 06:56（Asia/Taipei）彙整：monitor 正常退出，`current_index=4,current_candidate=masf-twn,completed_jobs=636,error=null`。本批 QAT 已完成 4/6，filterwise TWN 由原 queue 自動接續；本次未啟停 GPU。

Paper-TWN 五個回合的 16 指標皆通過搜尋雙門檻。末回合最差 mAP50／mAP50–95 總下降 0.874／1.064 百分點；相對其 PTQ，COCO mAP50–95 +0.206 百分點，BBAT box mAP50–95 +0.123 百分點，但 COCO Person mAP50 -0.071 百分點，不能說全面改善。

與 exact ternary 的末回合相比，Paper-TWN 的 COCO overall mAP50 略高（66.197% 對 66.123%），BBAT box mAP50–95 略低（83.088% 對 83.273%）；尚無一致支配關係，先保留兩者至 filterwise TWN 完成。單 seed、末回合比較不是 best checkpoint／formal 結論。

變更原因：完成事件出現才更新既有 CPU 報告及 README，沒有讀訓練 log 或重新分析執行器。彙整器驗證完成數 4，status=partial，無錯誤；回歸測試沿用同一五項檢查。困難無。剩餘第 5、6 組及累積配置未完成，接回單一 600 秒 monitor。

本次 CPU 回歸結果：5 passed（0.01 秒）。

## 第三次完成事件：filterwise TWN

2026-09-08 10:47（Asia/Taipei）彙整：monitor 回傳 `running_qat_preflight,current_index=5,current_candidate=detect-predictor-ls-sd4-recovery,current_arm=cpu_preflight,completed_jobs=637,error=null`。本批 5/6 完成，最後一組正在原 queue 的前置檢查；不能把 preflight 稱作訓練已開始。

Filterwise TWN 的五個回合、每回合 16 指標皆通過搜尋雙門檻。末回合最差 mAP50／mAP50–95 總下降為 0.826／0.981 百分點；相對其 PTQ，COCO mAP50–95 +0.226 百分點，BBAT box mAP50–95 +0.131 百分點，COCO Person mAP50–95 則 -0.019 百分點。

| MASF 方法（第 5 回合） | 最差 mAP50 下降 pp | 最差 mAP50–95 下降 pp | COCO mAP50–95 % | BBAT box mAP50–95 % | COCO Person mAP50–95 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact ternary | 0.949 | 1.069 | 48.733 | 83.273 | 61.180 |
| Paper-TWN | 0.874 | 1.064 | 48.738 | 83.088 | 61.168 |
| filterwise TWN | 0.826 | 0.981 | 48.821 | 83.297 | 61.149 |

解讀：filterwise TWN 在上述最差下降與整體 COCO／BBAT box 指標較好，可列為下一階段的優先候選；但 Person 指標較低，且尺度 metadata／精確成本及 export 未完成比較，因此不宣布全面 winner。三元在 MASF 的探索到此收斂，不把這三組訓練再複製到所有區域。

使用者本次詢問覆蓋與次數，已說明 148 部署權重路徑有量化不等於所有算子整數化；592 probe 與 40 PTQ 不含訓練，也不等於 148 層逐層 mAP。這批六組各 5 epochs，後續最多再兩組，不以此次詢問為由停止或擴張 queue。

驗證：CPU 彙整器確認 completed_jobs=5、status=partial；只讀完成產物，未讀 active log／source 或修改訓練。更新本輪結果與 README。困難無；第六組及累積配置仍待完成。回歸測試結果另補記。

本次報告回歸驗證 5 passed（0.01 秒），接續重啟 600 秒 shell monitor。
