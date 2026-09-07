# 2026-09-04：Paper-TWN 第一手文獻與逐區三元實驗契約

## 變更內容與原因

- 新增 `docs/research/2026-09-04-paper-twn-primary-literature.md`。
- 查核 TWN v2／v3、TTQ、INQ、Optimal Scaled Codebook、layer-wise TNN、MLQ 與 Reg-PTQ 的原始論文／作者正式頁。
- 原因是釐清 Paper-TWN 固定 PTQ、原論文 QAT、exact ternary、TTQ 與 progressive ternary 的方法邊界，並為「backbone 通過後固定，再調 neck，最後調 head」建立公平的逐區實驗契約。

## 驗證方式與結果

- 逐項對照原始 PDF 的公式、Algorithm、實驗 protocol 與表格。
- 確認 TWN v1／v2 使用 `0.7E|W|`，目前 v3 使用 `0.75E|W|` 且逐 filter；兩者不可共用未版本化名稱。
- 確認 TWN、TTQ、INQ 的主要成果均包含訓練／retraining，不能直接替 fixed PTQ 結果背書。
- 確認 Optimal Scaled Codebook 可對 `{-1,0,+1}` 作任意分布下的全域 weight-MSE 最優 PTQ，但不保證 task loss。
- 確認沒有第一手來源提供 Full35 backbone→neck→head 的安全排序；報告已把它明確列為待實測專案設計。
- 未使用 GPU、未執行訓練、calibration 或 validation，也未中止目前執行中的工作。

## 遇到的困難及解法

- 困難：TWN arXiv 不同版本的 threshold 與 granularity 有實質差異，摘要頁與最新版 PDF 內容也不同。
- 解法：直接比對版本化 PDF `1605.04711v2` 與 `1605.04711v3`，在報告中要求保存 `paper_version`、`threshold_multiplier` 與 `granularity`。
- 困難：部分 CVF PDF 入口回傳 403。
- 解法：同時使用 CVF 會議正式 HTML／官方 poster，以及作者大學頁保存的論文全文交叉核對。

## 未解事項或風險

- 既有 `paper_twn` artifact 是否完整保存版本與 granularity，仍需主工作流做本機 config／hash 稽核。
- Full35 各區 ternary 適合度仍需後續逐區 validation；文獻無法替代實測。
- exact ternary、TTQ 與 INQ-style recovery 的正式 GPU queue 尚未執行，本次只完成研究與規格。
