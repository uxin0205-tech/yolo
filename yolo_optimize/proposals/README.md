# 優化假說與原始計畫

本區保存各方向最初的推導、對照設計與停止條件，不作現在的 queue 狀態。實測結果以[總報告](<../reports/final/README.md>)及各實驗目錄為準；子計畫的 proposed／未開始可能是歷史文字。

| 方向 | 原始計畫 | 現行判斷 |
| --- | --- | --- |
| BinaryQK | [accuracy-recovery](<binaryqk-accuracy-recovery/README.md>) | 已做正式梯度修復與分段訓練，未完全追回 FP |
| 固定 scale／codebook | [scale-codebook](<binaryqk-scale-codebook/README.md>) | 固定尺度等價改善有證據；新8選1 selector未實施 |
| HOG | [HOG 計畫](<p3-hog-companion-training/README.md>) | 兩條研究線都有試驗，目前版本未採用 |
| P3 MASF | [Detect entry](<p3-masf-detect-entry/README.md>) | P3／P2／bridge有結果；使用者選bridge，不代表通過額外增準gate |
| 訓練衝突 | [conflict-safe](<training-conflict-safe/README.md>) | 投影啟動條件沒有足夠支持，未實施 |
| 已訓模型恢復 | [recovery](<trained-model-recovery/README.md>) | 原始設計保留；後續新combine與KD結果另列 |
| 第二輪創新 | [round2](<round2-innovation/README.md>) | 普通KD已測，區域／排序創新未完成 |
| person-only | [專用head](<coco-person-specialized-head/README.md>) | 依使用者要求延期，不建立新資料 |
| 整合計畫 | [integrated-roadmap](<integrated-roadmap/README.md>) | 早期融合後計畫與實測稽核，不冒充現在主線 |

新想法另建資料夾並清楚標記 proposed；程式與大型產物放實驗所屬階段。不要直接修改舊方法／parent來掩蓋失敗。整理前完整索引見[歷史入口](<../docs/history/README.md>)。
