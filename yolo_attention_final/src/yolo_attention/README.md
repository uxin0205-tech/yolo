# `yolo_attention` 套件

正式套件的公開入口是 `python -m yolo_attention.cli`。各模組維持單一責任：設定集中在 `config.py`；數學方法位於 basis、bias、normalization 與 BDCN 模組；`integration.py` 負責兩處模型轉換；`pwl_validation.py` 與 `pwl_experiment.py` 分別處理 PWL streaming diagnostics 與正式報告；訓練及 queue side effects 只允許出現在 runner、backend 與 executor。

資料流如下：

~~~text
VariantConfig → HardwareFriendlyAttention → convert_yolo26_model
              → 兩個 site 的 fail-closed 模型 → trainer/evaluator

queue_workflow → queue_store → queue_executor → queue_backend

pwl-score-analysis → adaptive range gate → Float/Bit-True PWL compare
                   → PWL results CSV/SVG/report
~~~

不得複製完整 Attention class、不得把演算法直接寫進 CLI，也不得讓只轉換單一 Attention site 的模型靜默通過。新增方法必須同步更新測試、factory 接線與文件。本目錄需納入 Git。
