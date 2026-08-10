# Domain docs

本文件說明 engineering skills 探索本 repo 時，應如何使用 domain 文件。

## 探索程式碼前

依序讀取：

- 根目錄的 `CONTEXT.md`。
- 若根目錄存在 `CONTEXT-MAP.md`，依其指向讀取與工作相關的 `CONTEXT.md`。
- `docs/adr/` 中與工作範圍相關的 ADR。

若文件不存在，直接繼續，不需回報缺少文件，也不要預先建立空白文件。
需要整理術語或確立決策時，再由 domain-modeling 流程建立。

## 文件結構

本 repo 採 single-context：

```text
/
├── CONTEXT.md
├── docs/
│   └── adr/
└── masf_yolo/
```

`CONTEXT.md` 記錄 domain glossary、邊界與核心概念；
`docs/adr/` 記錄已確立的架構決策。

## 使用 domain vocabulary

issue 標題、規格、重構提案、假設與測試名稱，應使用 `CONTEXT.md`
定義的術語，不任意改用同義詞。

若需要的概念不在 glossary：

1. 先確認是不是正在創造專案沒有使用的語言。
2. 若確實是 domain 缺口，記錄並交由 domain-modeling 流程處理。

## ADR 衝突

若工作內容與既有 ADR 衝突，必須明確指出，不得默默覆蓋既有決策。

例如：

> 與 ADR-0007 的既有決策衝突；建議重新討論，原因是……
