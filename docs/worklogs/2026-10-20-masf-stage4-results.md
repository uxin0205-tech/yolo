# 2026-10-20：第四階段 MASF 延長完成與交付

## 任務與範圍

使用者於 2026-10-19 授權延長訓練與不同方法，並要求每階段完成後先停下來檢視。本輪於 2026-10-20 09:50:32（Asia/Taipei）完成，延長流程耗時約 11 小時 52 分鐘，15／15 jobs 成功。10:39 狀態核對確認 supervisor／worker 已退出，GPU 無 compute process；第五階段沒有啟動。

本次依 finish-work 規範由單一主代理核對既有訓練與交付，沒有新增訓練、GPU 驗證、commit 或 push。事前 optimize 參照與七份來源 SHA 沿用 [延長紀錄](2026-10-19-masf-stage4-extension.md) 與 extension-reference.json；本次沒有修改方法或已鎖定的 Python 來源。

## 結果與建議

| 任務／方法 | 追加輪數 | 累計輪數 | 最佳累計輪數 | 最佳 AP50–95 | 相對原 parent |
|---|---:|---:|---:|---:|---:|
| Detect L1 | 8 | 13 | 8 | 0.514675024 | +0.000054233 |
| Detect L2 | 15 | 20 | 17 | 0.514728468 | +0.000107677 |
| Pose L1 | 6 | 11 | 6 | 0.920154750 | −0.000554639 |
| Pose L2 | 5 | 10 | 5 | 0.919994127 | −0.000715261 |

兩方法同任務從同一原 E5 完整 live／EMA／optimizer／scaler 狀態獨立出發。L1 延續末端 LR，L2 降低 head LR、提高 context LR。Detect L2 用完 15 輪追加預算，其餘依 patience 停止；Pose L2 最佳仍為續訓起點。累計輪數包含原共同五輪，不表示各候選重新訓練原 parent。

Detect 原 parent AP 為 0.514620791。L2 最佳整體增加約 0.0108 個百分點，sports ball 從 0.513224007 升到 0.513849131，但 baseball bat 從 0.487594089 降到 0.485704876，下降約 0.1889 個百分點，超過 .001 保護門檻。L1 也未通過球棒保護。最佳 L1／L2 的 alpha-on 減 alpha-off 整體 AP 分別為 −0.000007983／−0.000005570，不能把整體小幅提升歸功 MASF。

Pose 原 parent keypoint AP 為 0.920709389，兩組均未超過。L2 保留的原 E5 bbox AP 有提升，但 keypoint 下降，且同權重開關效果極小。兩任務都依事前規則建議保留原 parent，完整候選權重仍供使用者覆核。這是本輪配置與訓練範圍的結論，不能推論 MASF 在所有設定下無效。

## 交付與文件

- [完整正式結果](../../yolo_masf/artifacts/stage4/extension-20261019-v1/RESULTS.md)：Detect 八項主要 AP 加三項尺寸 AP、Pose 十二項 AP，包含兩組 on/off 與 parent。
- [完整指標 CSV](../../yolo_masf/artifacts/stage4/extension-20261019-v1/metrics-comparison.csv)：92 筆 task／arm／開關／指標比較。
- [訓練曲線](../../yolo_masf/artifacts/stage4/extension-20261019-v1/extension-curves.svg)、[逐輪數據](../../yolo_masf/artifacts/stage4/extension-20261019-v1/comparison.csv)：既有 plot 只需小型 JSON，不需要 GPU 或 checkpoint。
- [Detect 交接](../../yolo_masf/result/stage4/detect/README.md)、[Pose 交接](../../yolo_masf/result/stage4/pose/README.md)：原 parent 與四份候選均有 SHA；各候選正式 resume 保留。
- [最終稽核](../../yolo_masf/artifacts/stage4/extension-20261019-v1/final-audit.json)：來源、全量重載 AP、候選與 resume SHA、曲線格式及清理核對。

同步更新根層 README、MASF README、正式研究計畫、yolo_cleanup.md、run README 及 working-handoff.json 的完成狀態，保留歷史紀錄。沒有將研究候選覆蓋 optimize 正式模型。

## 驗證方式與結果

CPU 最終稽核通過：15 個 jobs 均成功；四份候選 SHA 與 handoff 相符，候選 sidecar 存在，原訓練 checkpoint 連結指向實際交付權重；四份 GPU 正式重載結果與各自訓練最佳 AP 完全相同；原 parent SHA 正確，四份必要 resume 亦記錄 SHA。verify_source 通過，SVG XML 解析通過，完整 CSV 存在。

沿用已完成的 18 項 CPU 延長測試、四組 GPU smoke，以及兩個任務的 alpha 零值等價、raw P3／P4／P5 不變、bridge 梯度、qSiLU 重載與非零 BN folding 診斷；本次文件與數據整理沒有重跑無關測試。Detect L1／L2 formal AMP retry 共 1／3 次，均重算原 macro 後完成；Pose 兩組沒有 retry。資料 assignment、labels、evaluator 與硬體契約保持原設定。

## 清理與成本

背景流程依既有刪除授權移除 8 份 smoke 權重，合計 1,223,126,450 bytes，約 1.139 GiB；本次核對皆已不存在。清單在 run 的 smoke-cleanup.json。正式負結果、唯一原 E5 resume、四份候選及必要續訓權重保留，沒有 archive／backup。本次狀態核對沒有再刪除其他檔案。

每個 MASF 新增 75,777 參數、475,136,000 Conv MAC（640 輸入）及額外 alpha 乘加／qSiLU。這些是解析成本，沒有硬體綜合或實板加速量測。推薦原 parent 不新增 MASF 成本；Detect 仍保有 B2／S1 Q10／integer PWL exp＋D1 reciprocal。

## 困難與解法

無新訓練或交付異常。背景流程已完成，但 README、研究計畫與 working-handoff.json 尚保存執行中狀態，已依實際結果同步。既有大量工作樹變更屬先前清理與研究內容，本次只修改上述文件及小型交付數據，沒有重置、暫存或提交。

## 未解事項或風險

單 seed、沒有未改方法 control 加訓，且兩組早停預算不同，不能宣稱統計顯著或完整因果歸因。ball／bat 類別 AP 不等於專屬小物件 AP，目前未追加正式尺寸分組錯誤分析。第四階段已停止，下一階段及任何追加實驗待使用者指定。
