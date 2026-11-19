# 跨階段選定成果

依 2026-10-19 使用者要求，完成選型後將選定資料夾移到 `result/stageN/`，供下一階段直接接續。權重只保留一份實體；必要歷史入口可以是符號連結。下一階段使用自己的程式與輸出目錄，不把新研究一直追加到舊 run。

- [第 1 階段 B2](stage1/README.md)：第 2 階段共同 parent。
- [第 2 階段 S1 Q10](stage2/README.md)：第 3 階段共同 parent，已搬移並核對全部 SHA。
- [搬移與完整 SHA 驗證](stage1-relocation.json)：14 個既有檔案均未改變，沒有複製大型產物。

小型設定與報告可提交 Git；checkpoint／resume 不提交。未被選中的正式比較與負結果留在相應階段 artifacts，不作下階段 parent。
