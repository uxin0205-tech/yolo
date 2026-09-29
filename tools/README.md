# 儲存庫維護工具

`check_repository_hygiene.py` 檢查 Git 暫存區中的檔案，阻止 checkpoint、分片權重、資料集本體、執行期垃圾、逐筆預測、LFS 指標與超過 25 MiB 的單檔進入 `Cleanup-v1`。只使用 Python 標準函式庫及 Git，不會啟動訓練或下載資料。

在儲存庫根目錄、選定要提交的檔案後執行：

```bash
python3 tools/check_repository_hygiene.py
```

輸入是 Git 暫存區；輸出是終端結果與 exit code，沒有產生新檔案。工具及本 README 提交 Git；環境、資料集、權重與快取保留在本機。完整政策見 [Cleanup-v1 說明](../docs/cleanup-v1/README.md)。
