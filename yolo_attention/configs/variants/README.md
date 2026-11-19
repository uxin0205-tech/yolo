# attention variants

每份 YAML 對應一個可驗證的 `VariantConfig`，只描述 Q/K basis、scale、relative bias、normalization、BDCN 或 optional quantization。正式 queue 的 winner-derived YAML 寫入 `../../artifacts/queue/generated/`，不覆寫這些模板。

新增方法時先擴充 typed enum、驗證與測試，再加入 YAML。使用：

~~~bash
python -m yolo_attention.cli validate-config configs/variants/h-screen.yaml
~~~

本目錄提交 Git；`q1-*` 是鎖住的 optional quantization 設定，不是已完成結果。
