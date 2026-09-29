# Full35 final 專用來源子集

> **Cleanup-v1 發行範圍：** 本分支保留程式、設定、報告、統計圖表與 manifests；checkpoint、資料集影像／標註、逐筆預測、batch 預覽和冗長執行日誌已從此發行快照排除。下文的歷史權重路徑、數量、checksums 與實驗結果仍保留作研究紀錄；訓練、推論、完整交付驗證及資料重建需要另行提供原始資產，不代表 clone 後即可直接重跑。 詳見[清理範圍與資產需求](../../../../docs/cleanup-v1/README.md)。


此處不是原始 9 候選 Full35／Partial75 研究 bundle 的完整複本，而是供
`final/full35` 重建 graph 所需的受控子集。

保留內容：

- `code/achitechure_1/` 與 `code/yolo_attention/` 的完整 Python 模組；
- Full35-A2 Float／Bit-True checkpoint 各一份；
- Float／Bit-True attention YAML 與精確 dependency lock；
- 原始來源與驗證說明。

刻意排除 Partial75、A0、B/C ablation、報表與其他淘汰權重。原始 bundle 說明另存
`ORIGINAL_BUNDLE_README.md`，只作血緣證據；本子集的實際內容以此目錄
`MANIFEST.json` 為準。
