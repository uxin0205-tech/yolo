# 2026-08-24 Pose physical batch／nbs 128 修正工作紀錄

## 變更內容與原因

正式 Pose P1／P2／P3 原已設定 physical batch 128，但 inherited `nbs` 仍為64。在目前固定的
Ultralytics 8.4.90 中，optimizer 建立前會以
`batch * accumulate / nbs` 縮放 weight decay；128／64會讓 YAML 中的
`weight_decay=0.00027` 實際變成0.00054。

本次將正式 P1／P2／P3 的 `nbs` 設成128，確保：

- physical batch=128；
- accumulate=1；
- configured與effective weight decay都是0.00027；
- 不改動 MuSGD、lr0=0.00038、lrf=0.5、momentum=0.948或其他來源超參數。

歷史 provisional P1 的physical batch16／nbs64仍保留於稽核欄位，沒有回溯改寫。
Full35與停用中的Partial75相容設定同步更新，但Partial75沒有執行。

## 驗證方式與結果

採 test-first：

1. 先加入 P1／P2／P3 `nbs == 128` 防回歸斷言，舊程式如預期失敗：`64 != 128`。
2. 修正公開 Pose stage policy 後，相關測試 `3 passed`。
3. 最終在 `CUDA_VISIBLE_DEVICES=-1` 下跑完整套件：
   `87 passed, 3 skipped in 109.04s`。
4. 三個 skipped 均為既有 CUDA integration test，沒有測試失敗，也沒有使用 GPU。

## 遇到的困難及解法

困難：若只看 YAML 的 weight decay 數值會漏掉 Ultralytics 的 nbs 縮放。
解法：以目前實際版本的 trainer 行為推導effective值，並以測試同時鎖定physical batch相關
的 nbs，避免日後再次靜默放大。

workspace既有檔更新仍遇到 `bwrap: loopback: Failed RTM_NEWADDR`；依既定方式先保存舊檔到
`/tmp`，再用 `apply_patch` 重建。無資料遺失。

## 未解事項與風險

physical batch 128 的RTX5090完整AMP memory feasibility仍待GPU空閒時驗證；本修正只確保
optimizer超參數語意正確，不代表記憶體已驗收。除此之外，無新增未解問題。
