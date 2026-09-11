# 方向 1 候選完整驗證與階段整理

## 變更與原因

依使用者要求先完成方向1再處理融合。新增候選驗證腳本與 `reports/direction1-20260910/`，集中結果、架構、超參數、來源及未解事項；原始權重與所有對照不搬移、不覆寫。

## 監測恢復

對話中斷後原工具 session 消失；只核對原 PID 200228，確認訓練仍存活、原 supervisor 已不在。新增 `watch_existing_masf.py`，以 pidfd 固定程序身分，每次最多阻塞等待600秒，正常不讀log或GPU。沒有重啟訓練。

程序退出後核對 summary 為 complete、E6–E10 各118287張／925次更新、配對trace一致，保存恢復監測與比較結果。非本監測器child，exit code未知，不能虛構exit0；完成判斷來自完整摘要與後續獨立驗證。

## 驗證結果

新增 `verify_direction1_candidates.py`，從無MASF control E8及MASF bridge E8的EMA各自匯出Bit-True權重。兩者各完成COCO5000重驗，四項AP與原epoch紀錄差值皆0，CPU160重載相同；各完成固定8張圖片推論。

驗證工作於UTC 2026-09-09 18:16:36正常退出0，summary passed。control overall/person=0.508267092/0.627699455，MASF=0.508211955/0.627664127。兩候選仍未回復FP，MASF額外增量未達門檻；不自動升格。

整理後檢查通過：13 個報告相對連結、2 個新增腳本 AST、已調和的 queue 狀態、2 份完整 COCO 候選紀錄與權重路徑。未重跑已通過的 GPU 驗證，也未擴大至無關測試。

## 困難與解法

原supervisor因中斷消失：用pidfd只接回監測。圖片目視工具因sandbox bwrap失敗：明記未完成目視，不以生成圖片代替視覺驗收。其餘完整AP驗證困難：無。

## 狀態與風險

本輪方向1預定實驗及數值驗收完成，但增準目標未完全達到、整體長期目標仍未完成。目前本研究沒有GPU工作在執行；combine、activation、方向2均未啟動。先向使用者確認下一階段是否保留MASF，不把略高分的無MASF候選擅自作融合來源。

使用交付整理技能核對來源、數值與未驗證範圍；所有清理候選均保留，不做刪除或Git發布。總結與權重入口見[方向1報告](<../../reports/direction1/README.md>)。
