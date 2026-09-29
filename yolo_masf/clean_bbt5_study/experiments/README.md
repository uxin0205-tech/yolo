# 實驗

## Strict-fair

12 個模型均由同一 clean initializer 開始，使用相同 direct full-model 最多 100 epochs、`patience=30` early stopping、資料、augmentation、optimizer 與 seeds 42/43。模型清單與唯一差異以 [`../../codex_plan.md`](../../codex_plan.md) 為準。

- B0：三尺度基準。
- P2：Direct 加上 PaperFormula、Lite35、Lite35-F7、Partial50、Partial25。
- P3：M7、Lite35、Lite35-F7、Partial50、Partial25；全部不含 F9。

## 分開報告

- P2-Control：head20 + full80，用來診斷新 head 最佳化，不混入 strict 排名。
- SP2/SP2P：主矩陣 validation selection freeze 後才執行。
- Smoke：只驗證訓練穩定性，不是實驗結果。
