# 測試

`test_p2_study.py` 覆蓋 A0 三尺度、A1/A2 四尺度與 P2 stride 4、Detect 權重轉移、A2 Stage 1 freeze、階段順序及 portable pretrained 路徑。

```bash
source ../.venv/bin/activate
python -m pytest tests/test_p2_study.py -q
```

其他檔案為 Ultralytics 上游測試；修改上游核心程式時再執行所需 test suite。
