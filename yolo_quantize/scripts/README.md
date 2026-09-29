# 操作入口

從專案根目錄執行。先看[目前計畫](../docs/CURRENT_PLAN.md)，不要重複啟動正在執行的 GPU queue。

2026-09-08 已在十二個累積 PTQ 完成點收尾，量化延後；目前沒有要接的新 QAT，monitor 已退出。下列 GPU／monitor 命令僅保留作日後明確恢復時的操作說明，不應現在執行。

## 日常 CPU 操作

| 腳本 | 輸入／輸出 | GPU |
| --- | --- | --- |
| `summarize_continuous_qat.py` | 本輪已完成 QAT → 逐回合／逐指標報告 | 不用 |
| `audit_continuous_evidence.py` | PTQ/probe/QAT metadata → 同源稽核 | 不用 |
| `inventory_project.py` | 目錄與 JSON/YAML 引用 → 分類／待刪清單；不刪檔 | 不用 |
| `profile_continuous_distribution.py` | parent checkpoint → 兩種 view 分布 | 不用 |
| `trace_continuous_precision.py` | parent → CPU 算子精度清單 | 不用 |

```bash
/home/uxin/yolo/.venv/bin/python scripts/summarize_continuous_qat.py
/home/uxin/yolo/.venv/bin/python scripts/audit_continuous_evidence.py
/home/uxin/yolo/.venv/bin/python scripts/inventory_project.py
```

報告與 metadata 可供 Git 發行；checkpoint、cache、runs 不因生成報告而自動上傳。上述腳本需要本機來源，不能宣稱乾淨 clone 能重建所有實驗。

## 執行及監測

| 腳本 | 角色 |
| --- | --- |
| `run_cumulative_ptq_queue.py` | 已完成的十二個累積 PTQ 入口；無參數只做 CPU 準備，目前不重啟 GPU |
| `prepare_continuous_qat.py`、`extend_continuous_qat.py` | 生成不可覆寫的本批計畫，不自行訓練 |
| `run_continuous_qat_queue.py` | 六組 CPU 預檢→GPU 串行訓練；本批已完成，不重啟 |
| `run_continuous_special_queue.py` | 已完成的 parent 重驗＋40 組 PTQ，不是目前啟動入口 |
| `run_continuous_layer_probe.py` | 已完成的 592 組 GPU 輸出探測，不是逐層 mAP |
| `validate_continuous_parent.py` | GPU 搜尋精度重驗，需授權及空閒 GPU |
| `preflight_continuous_parent.py` | parent reload；GPU export 的 EMA 投影需明確 `--projection-device cuda:0`，不可用 CPU 差異誤判血緣 |

```bash
PYTHONPATH=src /home/uxin/yolo/.venv/bin/python -m yolo_quantize.blocking_monitor artifacts/queues/full-model-cumulative-0908/execution-status.json
```

monitor 每 600 秒比較固定欄位，未變則靜默等待，有事件輸出後退出。它不是自動修復模型；supervisor 異常時保留 run，需診斷後明確 resume。

## CPU 回歸

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/uxin/yolo/.venv/bin/python -m pytest -q
/home/uxin/yolo/.venv/bin/ruff check src tests scripts
/home/uxin/yolo/.venv/bin/ruff format --check src tests scripts
```

歷史 activation、CPU profile 與舊 PTQ 命令見[整理前首頁快照](../docs/archive/project-readme-before-2026-09-08.md)及[歷史腳本說明](../docs/archive/scripts-before-2026-09-08.md)，不要把舊 `--execute-reviewed-plan` 命令當作本輪待辦。歷史图表使用 `matplotlib==3.11.1` 與 Noto Sans CJK；不需重繪來配合目前結論。
