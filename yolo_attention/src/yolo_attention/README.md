# yolo_attention package

production package 的公開入口是 `python -m yolo_attention.cli`。各模組維持單一責任：設定在 `config.py`，數學方法在 basis/bias/normalization/BDCN 模組，兩處模型轉換在 `integration.py`，訓練與 queue side effect 只在 runner/backend/executor。

資料流：

~~~text
VariantConfig → HardwareFriendlyAttention → convert_yolo26_model
              → two-site fail-closed model → trainer/evaluator

queue_workflow → queue_store → queue_executor → queue_backend
~~~

不得複製整個 Attention class、不得在 CLI 寫演算法，也不得讓單一 Attention 靜默通過。新增方法遵循根 `AGENTS.md` 的 test-first、factory 接線與文件同步規則。本目錄提交 Git。
