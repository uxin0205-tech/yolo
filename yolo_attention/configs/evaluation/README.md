# evaluation

`coco2017.yaml` 只描述共同 COCO2017 evaluator 資源，不包含 epochs、optimizer 或 LR。Queue 的 baseline、N0、BDCN zero-train 與 A-FINAL evaluation 都讀取這份設定；只有 `queue run-next --execute` 或 `queue run --execute` 才會呼叫 Ultralytics `val()`。

相對 `data` 路徑以 repository root 解讀。正式比較不得在不同 job 偷換 `imgsz`、split 或 evaluator 設定。
