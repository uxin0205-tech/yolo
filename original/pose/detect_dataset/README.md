# Ball／Bat Detection Dataset

此目錄由相鄰的 `dataset/` pose 資料衍生，來源檔案不會被修改。

- `data.yaml`：2 類別 detection 契約，`ball=0`、`bat=1`。
- `coco80/data.yaml`：既有 COCO80 detector 的驗證契約，`sports ball=32`、`baseball bat=34`。
- 影像使用逐檔相對 symlink；標註已移除 keypoint，只保留 `class x y w h`。
- 來源沒有實際 `test/` 內容，因此只提供 train／valid。

驗證集有部分 COCO train2017 ID 重疊；精確數量與雜湊請見 `manifest.json`。
