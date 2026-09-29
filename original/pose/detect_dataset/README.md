# Ball／Bat Detection Dataset

> **Cleanup-v1 發行範圍：** 本分支保留程式、設定、報告、統計圖表與 manifests；checkpoint、資料集影像／標註、逐筆預測、batch 預覽和冗長執行日誌已從此發行快照排除。下文的歷史權重路徑、數量、checksums 與實驗結果仍保留作研究紀錄；訓練、推論、完整交付驗證及資料重建需要另行提供原始資產，不代表 clone 後即可直接重跑。 詳見[清理範圍與資產需求](../../../docs/cleanup-v1/README.md)。


此目錄由相鄰的 `dataset/` pose 資料衍生，來源檔案不會被修改。

- `data.yaml`：2 類別 detection 契約，`ball=0`、`bat=1`。
- `coco80/data.yaml`：既有 COCO80 detector 的驗證契約，`sports ball=32`、`baseball bat=34`。
- 影像使用逐檔相對 symlink；標註已移除 keypoint，只保留 `class x y w h`。
- 來源沒有實際 `test/` 內容，因此只提供 train／valid。

驗證集有部分 COCO train2017 ID 重疊；精確數量與雜湊請見 `manifest.json`。
