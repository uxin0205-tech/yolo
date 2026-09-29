# GitHub 自動檢查

`workflows/repository-hygiene.yml` 在 push 與 pull request 後執行 `tools/check_repository_hygiene.py`，檢查已追蹤檔案與大小，避免大型訓練產物再次進入 Git。只有讀取程式庫權限，不下載 LFS、不安裝模型套件、不啟動 GPU。

輸入為提交內容，結果呈現在 GitHub Actions。工作流程檔案提交 Git；執行輸出保留在 Actions。
