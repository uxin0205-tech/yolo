# Pose 專項與共享特徵受控接續

最新結果：head KD 已完成 5 輪，E2 關鍵點小升但框下降，沒有合格 best_joint；正式 native 對照由使用者取消。原框＋KD keypoints 推論重組也沒有增益，均不升版；目前沒有 GPU queue。完整結果與保存入口見[總報告](<../../reports/consolidated-20260911/README.md>)。下方為執行前／歷史計畫，共享層試驗仍未實施。

最新指示：已取消正式 native 對照並停止該工作，保留其已完成產物。使用 `direct_queue.py` 只執行head KD五輪，再自動整理到 `artifacts/direct-kd-result-v1.json`。排好後結束模型working，背景程式每次wait最多600秒；錯誤停止並留log，沒有自動喚醒模型／自主修錯保證。共享特徵仍待本輪結果，以下两臂配對描述為取消前歷史。

使用者已核准執行 Pose-only，並允許安排共享特徵處理。第一階段從 qSiLU E2 起點做 native／head KD 配對，固定全部非 Pose state；不從退化的雙教師 KD E5 接續。完整背景見 [前一輪決策](<../dual_task_v1/POSE_ONLY_NEXT_PLAN.md>)。

## 第一階段

AdamW、Pose head LR1e-5、各5 epoch、warmup1、physical batch16、原生 Pose loss；共享 backbone／Neck／Detect／MASF／BN／QK／PWL全部固定。資料為 canonical BBAT5 v1 train5964／val683，不改 split，COCO仍全量驗證，不參加此階段訓練。

head KD 使用原獨立 Pose 教師的 cv2、cv4、one2one_cv2、one2one_cv4，三尺度各一個 hidden feature tap。teacher eval/no_grad；channel-mean square spatial transfer，不比錯位的 decode detections，不使用人體17點教師。μ 由 train-only 原生與 KD 的 head feature 梯度比5%校準，不能沿用前一輪 Neck μ。教師不進學生 state／EMA／export。

`run_queue.py --preflight`：calibrate→native smoke→KD smoke，每個輸出獨立。`run_queue.py`：前置通過後 native→KD各5輪。正常只 child.wait(timeout=600)，錯誤停止接續並只修復失敗 job。沒有宣稱外層模型等待零 token。

CPU imports、設定序列化與 AST 已通過；真實 GPU 前置及正式訓練結果以 artifacts/events 與 summary 為準，不把本說明當已完成。

## 第二階段：共享特徵授權已取得，條件式執行

先分析 head-only 六項 BBAT、COCO與匯出驗證。若 head-only 有足夠收益，先保留可用候選，不急著破壞共享特徵；若仍有 head難以補回的差距，安排同起點的小規模共享層對照。

擬先只解凍 P3 輸出融合 Conv（檢查實際 model.16.cv2 接線與參數後鎖定），不解凍整個 backbone。S0維持head-only，S1增加該小段共享權重，其他硬體常數、共享BN統計與Detect／MASF固定。兩臂都只餵BBAT5，head LR1e-5，新增共享Conv LR初探2e-7，fresh optimizer，先各3epoch、warmup1；實作前驗證更新範圍與安全界線，另開run並保留全部parent。這是後續設計值，不是已校準配方。

共享層一旦改動，Detect數學函數不再保證不變，所以每輪完整驗證COCO overall／person並檢查ball／bat，保留相對parent0.001精度保護線。出現明確取捨時不自行宣布犧牲person；先保存、分析，再決定是否調整。不得把S1與未追加同樣訓練預算的舊head-only結果比較，避免把額外訓練誤認為共享層收益。

兩阶段皆不新增部署層，不改activation或PWL [-10,0]20段。若仍需私有Pose adapter或大規模backbone重訓，需先提出收益／成本與更具體證據，不先堆進本次queue。
