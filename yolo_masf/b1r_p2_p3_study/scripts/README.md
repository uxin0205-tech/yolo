# 執行與重建入口

## GPU queue

```bash
PYTHONPATH=. ../.venv/bin/python -m masf_yolo.retest.queue
```

Queue 一次只跑一個 worker；requests 與 worker manifests 位於 [`../results/metadata/`](../results/metadata/)。

## B0-Fair（目前不要執行）

B0-Fair 已完成設定與 worker 支援，但尚未排入 GPU queue。確認 GPU 空閒後，再依 [`../../configs/retest/requests/README.md`](../../configs/retest/requests/README.md) 執行單一正式工作。

## 統一後處理

```bash
PYTHONPATH=. ../.venv/bin/python -m masf_yolo.retest.postprocess
PYTHONPATH=. ../.venv/bin/python -m masf_yolo.retest.profile_all
PYTHONPATH=. ../.venv/bin/python -m masf_yolo.retest.report
PYTHONPATH=. ../.venv/bin/python -m masf_yolo.retest.audit
```

## GitHub 發布包

```bash
python scripts/build_retest_publication.py \
  --source artifacts/b1r-p2-p3-retest \
  --repo .
```

發布腳本把 runtime metrics、profiles、queue metadata 與 lineage 實體化到 `b1r_p2_p3_study/results/`；不複製 dataset、smoke/last checkpoints、大型 predictions 或 queue logs。

完整流程請看 [`../EXPERIMENT_PROCESS.md`](../EXPERIMENT_PROCESS.md)。
