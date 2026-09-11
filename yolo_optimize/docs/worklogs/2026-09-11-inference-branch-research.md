# 2026-09-11：推論分支重組研究與完整驗證

## 變更與原因

使用者要求研究目前推論能如何處理。使用 research 技能查核本機 Pose26／NMS 原始碼及官方文件，由主代理完成，不啟用子代理。新增 `inference/pose_branch_v1/probe.py` 與研究報告；不重新訓練，嘗試以原 qSiLU E2 框分類加 KD E2 關鍵點，避免直接接受 KD 框精度下降。

## 驗證與結果

確認來源 hash，所有非 Pose state 相同；只置換 96 個關鍵點分支 state。Float／BitTrue CPU 接線精確通過，候選 weights_only 匯出 roundtrip 通過。兩後端完整 COCO 5,000 張與 canonical BBAT Pose 683 張驗證完成；imgsz 640、batch 32／16，沒有訓練或資料修改。

BitTrue COCO／person／ball／bat 框 AP 與原模型精確不變；整體 Pose 從 0.891328882 到 0.890906327，bat Pose 從 0.923008323 到 0.922163213。未保留完整 KD E2 的 Pose 增益，因此不升版、不替換原 best。完整數據與下一步順序見[研究報告](<../../inference/pose_branch_v1/README.md>)。

## 困難與解法

v1 發生 KeyError：BitTrue normalize.endpoint_table 為後端轉換新增常數，不能用 Float state 全量覆蓋。改成共享來源先驗證相同，只載入 Pose head，保留後端轉換；v2 CPU 與完整 GPU 驗證均成功。舊失敗產物與原權重全部保留。sandbox bwrap 不可用，必要命令經權限審核執行。

## 未解事項與風險

本次推論重組無改善，不推論為 KD 全面無效。分類排序／候選選取與關鍵點 AP 有耦合，未量化各因素占比。沒有完整原生對照與獨立 test。one2many + NMS 比較、實際部署座標流程與板端效能尚未驗證；共享層訓練未啟動。目前本次監測工作已 JOB_DONE，無後續 GPU queue。舊 Pose head KD 五輪亦已完成，更新狀態，避免沿用「背景執行中」敘述。
