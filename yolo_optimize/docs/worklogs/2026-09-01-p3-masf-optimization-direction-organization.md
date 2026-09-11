# 2026-09-01：P3 MASF 優化方向與計畫整理

## 工作目標

把「P3 MASF 應移到 Detect entry」的優化方向與可執行計畫放在同一個獨立資料夾，並建立可供未來
多個優化方向沿用的索引與文件契約。

## 變更內容與原因

1. 新增 `optimizations/README.md`，定義一個方向一個資料夾、`README.md` + `plan.md` 的最低契約、
   lifecycle 狀態與共通實驗規則。
2. 新增 `optimizations/p3-masf-detect-entry/README.md`，集中記錄現行 shared-seam 問題、原本／建議
   計算圖、既有 metrics、CPU prototype 證據、範圍與相容性風險。
3. 新增同目錄 `plan.md`，把 production graft、routing tests、checkpoint lifecycle、訓練 scope、
   C0/C1/C2 matrix、精度 gate 與停止條件拆成可勾選步驟。
4. 更新子專案 README 與研究報告的導覽連結，讓研究證據、方向與執行計畫互相可追溯。
5. 修正子專案 README 的 accuracy audit 說明：現行腳本預期是四個 fail-closed 紅燈，不是三個。

這次只整理文件與計畫，沒有修改 `achitechure_1` production model、checkpoint 或實驗結果，也沒有啟動
GPU training／validation。

## 驗證方式與結果

- 以唯讀腳本掃描本次新增與修改的 7 份 Markdown，共解析 29 個本機相對連結；結果
  `missing_local_links=0`。
- 檢查 `optimizations/p3-masf-detect-entry/` 同時包含 `README.md` 與 `plan.md`；方向 ID、
  `proposed` 狀態、C0/C1/C2 arms、`+0.001` material-gain gate 與停止條件一致。
- 執行 `python3 scripts/audit_accuracy_regressions.py`，結果如預期回傳 `1`，並明確輸出
  `[RED] 偵測到 4 個目前精度回歸訊號`；因此 README 的四個 fail-closed 紅燈說明正確。

## 資料集影響

無。未讀寫或重建任何 dataset、split、影像與標註；計畫明確要求後續 BBAT5 實驗只能使用不可變的
`/home/uxin/yolo/original/pose/derived/bbat5-v1/` 與正式 configs。

## 困難與解法

既有環境的預設 `apply_patch` sandbox 曾出現 `bwrap: loopback: Failed RTM_NEWADDR`；改用相同
`apply_patch` 並移除 sandbox 注入的 network-disabled 環境變數後完成文件修改，沒有繞過檔案範圍限制。

## 未解事項與風險

- 尚未完成 production Detect-entry graft 與 regression tests。
- 尚未執行 M0/M1 full validation 或 C0/C1/C2 matched GPU experiments，不能宣稱此方向會提升 AP。
- 新舊 seam 的 checkpoint lineage 必須分開，避免把 shared-seam trained weights 當成新位置的正式結果。
