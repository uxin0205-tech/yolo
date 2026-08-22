---
status: accepted
date: 2026-08-22
---

# 只承接融合 winner，依 winner graph 解析候選區域

## 決策

architecture_2 不再重做融合、雙權重切換或來源架構選擇。它只接受 `yolo_combine` 的單一正式 Handoff Revision，將該模型原樣定義為 C0-Handoff，再依 handoff 宣告且通過 graph audit 的 Candidate Region 建立 C1、C2、C3。

共享 winner 在共享 trunk 套用因子；路由 winner 分別產生 D-C* 與 P-C*；部分共享 winner依實際區域產生 S-C*、D-C*、P-C*。任何實際 module path 都來自 handoff，不在本專案預先寫死 layer index。

## 理由

- 融合 winner 尚未產生，預先假設 shared 或 routed 會讓候選比較失去對象。
- 硬寫原始 YOLO layer 6、8、13、19 無法證明它們仍是 winner 中可比較的相同區域。
- 將融合與簡化分成兩個 workspace，才能把性能差異歸因於 C1、C2 或 C3。

## 後果

- 沒有正式 handoff 時只能完成設定、fixture 與 CPU contract tests，不能宣稱候選已建立或可正式訓練。
- C0-Handoff 與 C0-Control 必須分開報告。
- C_best 與量化候選都由使用者在 Float 結果後決定，程式不得自動選擇。
- 若上游變更 winner，必須建立新的 Handoff Revision；不同 revision 的結果不得混用。
