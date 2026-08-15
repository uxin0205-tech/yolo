# artifacts/queue

這是 persistent research queue 的唯一狀態目錄。

~~~text
queue/
├── queue.json      # 唯一目前狀態來源
├── events.jsonl    # append-only 歷史事件
├── worker.lock     # 防止兩個 worker 同時執行
└── generated/      # selection winner 派生的 variant YAML
~~~

使用 `python -m yolo_attention.cli queue status|validate|next` 讀取；`run-next` 不加 `--execute` 只預覽。不得手改 job 狀態或複製 checkpoint 假裝成功。queue artifacts 不提交 Git；本 README 提交。
