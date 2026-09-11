# 2026-09-01：P3 MASF 終端架構圖報告

## 工作目標

回應「是否不該直接融合」的疑問，製作可直接在 terminal 閱讀的架構報告，清楚區分目前 shared-seam、
第一階段 Detect-entry 修改，以及只有 C2 驗證成功後才考慮的模組升級。

## 變更內容與原因

1. 新增 `optimizations/p3-masf-detect-entry/architecture-report.md`，以等寬文字圖呈現三個架構階段。
2. 報告補上目前與建議架構的 forward equations、MASF 梯度路徑推導、state owner、CPU dependency
   證據、A/B 差異表與工程 gate。
3. 更新方向 README、執行計畫、優化方向契約與根 README 的導覽，讓終端報告可從主要入口找到。

這次只有文件變更，沒有修改 production model、checkpoint、dataset 或訓練設定，也沒有執行 GPU
training／validation。

## 驗證方式與結果

- 以 `sed -n '1,360p'` 在 terminal 完整輸出報告；改成精簡圖後目前共 282 行，目前 shared-seam、第一階段
  Detect-entry 與條件式 C3 三張等寬文字圖均可閱讀。
- 掃描本次相關 Markdown 的 34 個本機相對連結，結果 `missing_local_links=0`；移除唯一一處非必要
  行尾空白後再次檢查。
- 內容 assertion 確認三個階段、`model.16`／`model.23` state owner 與
  `∂p4/∂θ = ∂p5/∂θ = 0` 隔離性質都有記錄。
- 報告明記 CPU prototype 只證明 routing 隔離，尚未證明 AP 提升，沒有把工程驗證誤寫成精度結論。

## 困難與解法

無。

## 未解事項與風險

- Detect-entry 仍是 `proposed`，尚未 production 實作。
- CPU probe 只證明 P4/P5 可被隔離；C0/C1/C2 matched GPU experiment 尚未執行，不能保證 AP 提升。

## 後續格式調整

依使用者指定，將目前 C1 與預計 C2 兩張主圖統一改成
`layer：feature ─┬─> downstream` 的精簡 layer-flow 格式。現況使用 `p4_affected/p5_affected`
明示跨尺度影響；修改後使用 `p4_raw/p5_raw` 明示未經 MASF 的原始 pyramid route，並補充
Detect-entry 是 `model.23` 內部 adapter，不新增 YAML layer index。

以 terminal 顯示兩張新圖後，檢查以下六個必要元素全部通過：目前 shared path、目前
`p4_affected`、目標 Detect-entry fork、目標 `p4_raw`，以及兩個版本各自的 Detect input list。
行尾空白檢查沒有發現問題。
