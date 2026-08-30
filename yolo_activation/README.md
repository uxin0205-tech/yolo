# YOLO Activation 研究

本子專案研究 YOLO detection／pose 共用模型的 activation 選擇、受約束數學候選、逐層 policy 与
後續硬體／量化實作。Full35 activation-only 階段已在 2026-08-29 收尾：以
`yolo_combine/final/full35` accepted checkpoint、完整 COCO2017 + Canonical BBAT5 v1 完成 profiling、
zero-shot、11-region sensitivity、10-epoch recovery 与 matched finalist control。唯一通過八項
10-epoch gate 的非 SiLU 候選是 `qsilu_pq`；其 20-epoch finalist 依使用者要求停止，先交给量化工作列
分析 activation × quantization 耦合。尚無板上硬體結果。

## 文件入口

- [原始 activation 任務規格](docs/Activation_research.md)
- [Phase 0 研究索引](docs/research/README.md)
- [硬體友善 activation 原型與驗證計畫](docs/research/phase0-hardware-activation-prototype.md)
- [SIPA–BCSP 學術主張與完整實驗計畫](docs/research/sipa-bcsp-experiment-plan.md)
- [SIPA–BCSP 新穎性 prior-art 稽核](docs/research/sipa-bcsp-novelty-audit.md)
- [SIPA-BCSP 完整訓練架構](training/README.md)
- [Full35 全量正式實驗配置](training/full35/README.md)
- [已完成 Activation 权威整合报告](reports/completed-activation-integrated-report.md)
- [Full35 Activation 最終整合分析](reports/full35-activation-final-analysis.md)
- [六種 activation 完整数学推导](docs/research/full35-activation-mathematical-derivations.md)
- [Machine-readable 结果](reports/full35-activation-results.json)
- [可重建 SiLU／qSiLU 权重](release/weights/README.md)
- [中文工作紀錄](docs/worklogs/README.md)
- [全域 BBAT5 資料集規範](../docs/agents/bbat5-datasets.md)

## 資料集入口

| 任務 | 正式入口 | 邊界 |
| --- | --- | --- |
| BBAT5 v1 ball/bat Detect | `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml` | 二類 Detect Task View；固定 assignment，不得重切、抽樣或改標註 |
| COCO2017 COCO80 Detect | `/home/uxin/yolo/coco2017.yaml` | 80 類通用 Detect；不得改接 BBAT5 二類 head/YAML |

兩個資料集分開訓練、驗證與報告；只以各自 SiLU baseline 的 delta 評估跨資料集穩健性，不合併
raw mAP。任何 runtime dataset view 都必須可從正式入口重建，不得形成新資料版本。

## 本地原型驗證

```bash
/home/uxin/yolo/.venv/bin/python -m pytest
/home/uxin/yolo/.venv/bin/python scripts/validate_phase0.py
/home/uxin/yolo/.venv/bin/python scripts/activation_training.py toy-dry-run
```

預設 registry 包含 SiLU、Hardswish、ReLU diagnostic、`qsilu_pq`，以及 SIPA 的
`poly_shift`／`poly_quality`。ReLU 不進昂貴 recovery；`poly_shift` prerequisite 失败后，14 个
region／mixed jobs 依 frozen queue 正确封锁。完整数值、停止点及下一步见
[最终报告](reports/full35-activation-final-analysis.md)；硬体效能仍需 target profile、synthesis 与板上量测。
