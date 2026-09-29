# Field Check

> **Cleanup-v1 發行範圍：** 本分支保留程式、設定、報告、統計圖表與 manifests；checkpoint、資料集影像／標註、逐筆預測、batch 預覽和冗長執行日誌已從此發行快照排除。下文的歷史權重路徑、數量、checksums 與實驗結果仍保留作研究紀錄；訓練、推論、完整交付驗證及資料重建需要另行提供原始資產，不代表 clone 後即可直接重跑。 詳見[清理範圍與資產需求](../../docs/cleanup-v1/README.md)。


本目錄保存 Clean 公平研究外的探索性檢查，例如 `context_rf_cpu/` 的 receptive-field/context CPU 實驗。這些結果不是正式排名證據，也不能取代 Clean initializer、多 seed 與一致 training budget 的實驗。新探索應放在獨立子目錄；若要進入正式結論，須先納入鎖定設定、selection freeze 與 final audit。
