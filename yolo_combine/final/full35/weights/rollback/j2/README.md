# J2 rollback

> **Cleanup-v1 發行範圍：** 本分支保留程式、設定、報告、統計圖表與 manifests；checkpoint、資料集影像／標註、逐筆預測、batch 預覽和冗長執行日誌已從此發行快照排除。下文的歷史權重路徑、數量、checksums 與實驗結果仍保留作研究紀錄；訓練、推論、完整交付驗證及資料重建需要另行提供原始資產，不代表 clone 後即可直接重跑。 詳見[清理範圍與資產需求](../../../../../../docs/cleanup-v1/README.md)。


這裡保留升格J3前的完整J2四種selector權重。正式主權重已是J3；只有`weights/combined/*/best_detect.pt`仍沿用J2，因J3沒有刷新Detect selector。
