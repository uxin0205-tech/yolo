# 本次發布保存盤點

無建議刪除候選；全部 Keep，不進行刪除或覆寫既有結果。

| ID | 確切範圍（相對本目錄） | 類型 | 依賴／風險 | 決定 |
| --- | --- | --- | --- | --- |
| K01 | artifacts/cpu-preflight-v1/cpu-test-resume.pt | CPU 測試快照 | 完整恢復測試來源；不是訓練起點，不上傳 Git | 保留本機 |
| K02 | artifacts/cpu-preflight-v1/summary.json | CPU 證據 | 報告引用，不能刪 | 保留並發布 |
| K03 | figures/ | 圖與可編輯來源 | 完整架構、推导必要；preview 是本機檢視副本 | 正式 DOT／SVG／PNG 發布，preview 保留本機 |
| K04 | proposed-training.json | 被取代設計 | A/B 歷史依據，非現行設定 | 保留並標明 superseded |

獨立 Git 臨時工作區僅包含本次發布副本，完成 push、驗證乾淨後移除本次建立者；既有工作區不動。原始 checkpoint、cache 與其他研究全部不在清理範圍。
