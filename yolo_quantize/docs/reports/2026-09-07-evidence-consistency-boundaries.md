# 比較條件與證據層級：本輪為準，歷史結果保留但不混用

## 結論

使用者指出的兩項限制成立：舊 V35 CPU profile 與較早 qSiLU recovery PTQ 不能稱為同 parent 隔離比較；1,332 筆 CPU 權重投影誤差不等於 148 層逐層 mAP 驗證。

本輪已改用同一個 `v36-qsilu-a8-full-coverage-epoch1` 起點。CPU 稽核實際產物確認：40 組區域 PTQ、592 組單層輸出探測、6 份 QAT 計畫，以及目前已產生的 3 份 QAT 執行圖，均指向同一 parent 血緣。這項稽核只確認來源與覆蓋，不是新增 mAP 測試，亦不追溯修復舊實驗的比較條件。

## 證據必須分層閱讀

| 證據 | 起點／數量 | 回答的問題 | 不能宣稱 |
| --- | --- | --- | --- |
| 歷史 CPU profile | V35；148 × 9＝1,332 筆 | 格式投影後的權重 NRMSE、cosine、估計成本 | 已完成逐層任務精度驗證；等同本輪 V36 排序 |
| 歷史 PTQ | 部分較早 qSiLU recovery checkpoint | 該起點與路由的任務精度；可提出排序遷移假說 | 與 V35 CPU profile 構成隔離實驗；已證明排序可遷移 |
| 本輪 CPU 分布 | V36；148 層 × EMA shadow／deployed 兩種 view | 近零集中程度、尾部、幅值分布 | 296 筆等於格式 mAP 測試；兩種 view 可以混為同一數值 |
| 本輪單層輸出探測 | V36；148 × 4＝592 組 | 一次替換一層時的輸出敏感度；固定 Detect／Pose 各一張樣本 | 148 層完整 mAP 矩陣；已找到逐層最優格式 |
| 本輪十區 PTQ | V36；10 × 4＝40 組 | 同起點、一次只替換一區的完整搜尋任務指標 | 所有單層候選已逐一做 mAP；PTQ 差就一定不能 QAT |
| 本輪短 QAT | V36 EMA 起跑；6 組排程、各最多 5 epochs | 特定區域格式改變後，模型與量化參數適應的結果 | 只更新替換層的隔離測試；沿用外部 sham 等於同 parent sham 對照 |

四種 probe／PTQ 格式為 Fixed-SD4、exact ternary、Paper-TWN、filterwise TWN。LS-SD4 的尺度學習效果另由同區域 Fixed／LS-SD4 QAT 配對檢驗，不能把靜態 Fixed-SD4 投影直接叫作已驗證 LS-SD4。

## 本輪血緣定位

- Parent manifest SHA-256：`f25a3caaebe6b7d51663d615c0202be7dedec431683cb0ad0c8fe9dccfc3d48a`。
- 部署 inference SHA-256：`8c1f3652fd21c0e38221cfc3acba7ee013c5e88df641b9331c33796f6f11b66c`。
- QAT full-resume SHA-256：`f83e1ba22adc737ed05abf68152f621a67915a2104ace3bfd5a3b630d66dcb78`，載入 `ema_state`。

inference 與 full-resume 是不同用途的產物，不要求兩個檔案雜湊相同；要求同一 manifest 的 checkpoint 血緣，以及既有 export 重载／EMA 投影驗證相符。本輪 parent 搜尋精度重驗亦已有紀錄，不重新訓練對照模型。

重建稽核：`/home/uxin/yolo/.venv/bin/python scripts/audit_continuous_evidence.py`。
機讀結果：`artifacts/queues/full-model-continuous-0907/evidence-consistency-audit.json`。

## 三個容易誤讀的舊欄位

1. `isolated-special-report.json.reference` 保留歷史 accepted／matched 參照。它不是本輪權重替換的實際起點；隔離比較請用各 `results.*.build` 及 `incremental_reference` 指向的 V36。相對 accepted 的總下降仍需計入 activation 替換。
2. 該檔 `selection` 是舊 mAP50-only 邏輯；本輪部署候選判定使用 `special-dual-summary.json` 的 16 項雙門檻。QAT recovery admission 只代表允許測恢復，不代表精度達標。
3. 執行圖 `warm_start.activation_calibration` 尚有 `reused_locked_v19_learned_state` 舊字串。已核對其 parent、checkpoint、state_key 與雜湊實際為 V36；舊字串不是從 V19 載入的證據。保留既有產物不回寫，以本稽核補充正確解讀。

## 尚需完成

目前已完成的 Fixed／LS-SD4 五回合結果見[恢復報告](2026-09-07-continuous-qat-recovery-results.md)，不能據此宣稱所有層都已做 QAT。下一步仍是有限候選 QAT、必要的單層／子區域 mAP 確認、backbone→neck→head 累積配置與交互影響驗證，之後才鎖定 finalists 做 export 重驗與 formal。

同起點只是可比較的必要條件，不是充分条件：粒度、scale 方法、選層集合、訓練範圍與驗證資料也必須揭露；單 seed 結果不作統計顯著性宣稱。全模型 148 權重路徑已有量化配置，也不等於所有算子已完成純整數部署。
