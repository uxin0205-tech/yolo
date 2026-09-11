# 2026-09-11：取消原生對照，直接 KD 並結束模型等待

## 變更與原因

使用者明確表示不需要原生對照，要求直接修正，並要求排好queue後終止working。已核對PID／PGID1157731為native Pose-only訓練，再SIGTERM終止該process group；原queue收到exit -15停止，這是使用者取消而非訓練數值錯誤。已產生驗證／log全部保留，不把未完成native當完整對照；未刪除或覆寫權重。尚未保存的當前更新不會作為新起點。

新增 `kd/pose_focus_v1/direct_queue.py`：從原qSiLU E2執行head KD五輪，成功後跑`summarize_direct.py`整理逐epoch與起點比較。不是從已取消native接續。原兩個macro native smoke只是安全檢查，保留不重跑；不再做native正式五輪。

## 驗證與監測

沿用已通過的KD校準、真實更新、教師不變、非Pose state與Detect输出精確不變前置。啟動前檢查新增脚本AST及沒有既有KD run。背景queue每次child.wait最多600秒，不呼叫模型API；job失敗會保留log並停止，不宣稱程式會自主分析／修復未知錯誤。

## 困難與未解

無。使用者允許停止working後，不再由模型反覆續接等待；GPU訓練與腳本可在背景接續，但未承諾工作完成時模型自動喚醒。共享特徵解凍仍需看KD結果後決定，尚未把未驗證的共享層程式排入queue；獨立匯出重驗也未包含於自動結果整理。因取消完整native對照，只能比較與起點差異，不能把所有提升歸因為KD。
