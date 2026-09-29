# YOLO Activation 文件導覽

這個目錄同時保存目前結論、數學與研究過程。為避免把歷史狀態誤認成現在狀態，閱讀時請依下列順序。

## 目前應先閱讀

1. [已完成 Activation 權威整合報告](../reports/completed-activation-integrated-report.md)：目前完成範圍、六種 activation、訓練設定、完整結果與量化交接。
2. [六種 activation 可讀版數學推導](research/full35-activation-mathematical-derivations.md)：逐步說明每個符號、限制條件、係數來源與硬體含意。
3. [Machine-readable JSON](../reports/full35-activation-results.json)與[八項指標 CSV](../reports/full35-activation-results.csv)：可供程式或試算表核對的正式數值。
4. [可重建權重](../release/weights/README.md)：accepted SiLU 與完成 10-epoch recovery 的 qSiLU 權重。
5. [Full35 正式實驗配置](../training/full35/README.md)：資料、epoch、batch、學習率、queue 與 gate。
6. [中文工作紀錄](worklogs/README.md)：按日期保存修改、驗證、困難與未解風險。

## 目前狀態摘要

- Activation-only 階段已停止，不再繼續調參或訓練。
- 六種 activation 都已有明確定位；唯一通過 10-epoch 八項 gate 的非 SiLU 候選是 `qsilu_pq`。
- SIPA 已完成數學、實作與 Full35 實驗；BCSP 只完成 manifest、region、queue 與 gate 支援，沒有完成 mixed-policy 搜尋。
- 量化、QAT、Q8／Q12／Q16 全碼枚舉與 FPGA／ASIC 實測尚未在本子專案完成。

## 歷史研究文件

下列文件保留當時的研究問題與決策過程。文件中的「尚未收到模型」或「等待實驗」代表該日期的真實狀態，不代表現在仍缺少同一項目。

| 文件 | 定位 |
| --- | --- |
| [原始 Activation 任務規格](Activation_research.md) | 使用者最初需求與研究方向，僅供追溯 |
| [Phase 0 activation 版圖](research/phase0-activation-landscape.md) | 早期文獻與候選篩選 |
| [Phase 0 數學設計](research/phase0-mathematical-design.md) | SIPA 前期數學草案 |
| [Phase 0 硬體原型](research/phase0-hardware-activation-prototype.md) | 純函數、fixed-point 與 ONNX 原型 |
| [qSiLU 硬體近似研究](research/qsilu-hardware-approximation.md) | qSiLU prior art、命名與硬體邊界 |
| [SIPA–BCSP 實驗計畫](research/sipa-bcsp-experiment-plan.md) | 預註冊計畫；不等於所有工作都已執行 |
| [SIPA–BCSP 新穎性稽核](research/sipa-bcsp-novelty-audit.md) | 學術主張與先前技術邊界 |

## 資料與發布邊界

- 新實驗固定使用 Canonical BBAT5 v1 與 COCO2017，不重切、不抽樣、不改標註。
- 本機 `artifacts/` 保存約 29 GB 原始 run、queue、OOM 與驗證證據，受 Git ignore 排除。
- GitHub 保存來源碼、設定、測試、整理後結果，以及量化需要的兩組可重建權重分片。
- 歷史失敗權重、中止的 qSiLU finalist `last.pt` 與 optimizer checkpoint 不發布。
