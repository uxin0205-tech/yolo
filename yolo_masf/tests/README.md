# 自動測試

目錄對應 `masf_yolo/` 的 data、models、training、evaluation、artifacts；根層測 CLI、workflow、runtime、reporting、cleanup 與 contracts。

```bash
env PYTHONPATH=. ../.venv/bin/pytest -q
```

`__pycache__`、`.pytest_cache` 可刪；fixtures 與測試程式不可清除。
