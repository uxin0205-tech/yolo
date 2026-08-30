# Activation 研究索引

## 目前結論與數學

- [六種 activation 可讀版完整數學推導](full35-activation-mathematical-derivations.md)：所有符號、限制條件、係數來源、分段公式、fixed-point 不變量與實驗定位。
- [qSiLU 硬體近似與名稱邊界](qsilu-hardware-approximation.md)：prior art、dyadic 係數、誤差與硬體主張限制。
- [SIPA–BCSP 新穎性稽核](sipa-bcsp-novelty-audit.md)：哪些組合可能構成研究貢獻，哪些內容已有先前技術。
- [Activation 權威整合報告](../../reports/completed-activation-integrated-report.md)：Full35 實驗結果與量化交接。

目前只有 `qsilu_pq` 通過完整 COCO2017 + Canonical BBAT5 v1 的 10-epoch 八項 gate。SIPA 的數學、
實作與 Full35 實驗已完成；BCSP 支援架構已完成，但 mixed-policy 搜尋沒有執行完成。

## 歷史研究與預註冊計畫

| 文件 | 內容 | 閱讀方式 |
| --- | --- | --- |
| [Phase 0 activation 研究版圖](phase0-activation-landscape.md) | 早期 primary-source 文獻與候選篩選 | 歷史研究狀態 |
| [Phase 0 數學設計草案](phase0-mathematical-design.md) | SIPA 的早期推導與假說 | 以可讀版數學文件為準 |
| [Phase 0 硬體原型](phase0-hardware-activation-prototype.md) | 純函數、Q16.10 與 ONNX 原型 | 不是 detector AP 或板上結果 |
| [SIPA–BCSP 實驗計畫](sipa-bcsp-experiment-plan.md) | 預註冊的完整理想實驗 | 計畫不等於全部已執行 |

舊文件中的「尚未收到 baseline」或「等待正式實驗」是文件建立當時的真實狀態。現在是否完成，請以
[文件導覽](../README.md)與[權威整合報告](../../reports/completed-activation-integrated-report.md)為準。

## 學術聲明邊界

- `qsilu_pq` 是本專案凍結的 qSiLU 候選，不宣稱所有分段二次 SiLU 近似都是新方法。
- SIPA 的 individual ingredients 有 prior art；主張必須建立在組合限制、bit-exact 證據與完整模型結果。
- BCSP 尚未產生 normalized-regret 搜尋結果，也沒有完成 random、greedy 或 ActNAS-style 比較。
- 目前沒有 target FPGA／ASIC synthesis、latency、power 或資源量測，不能宣稱已證明硬體加速。
