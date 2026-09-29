# generated variants

每個子目錄由 `queue_workflow.py` 根據實際 parent winner 建立，內容通常是 `<job-id>/variant.yaml`。這些 YAML 保留 parent 的 basis、scale、bias 等欄位，只改該階段允許變動的設定。

- 輸入：成功 parent 的 variant 與 selection policy。
- 輸出：下一個 queue job 的不可手改 variant snapshot。
- Git：generated artifacts 不提交；本 README 提交。
