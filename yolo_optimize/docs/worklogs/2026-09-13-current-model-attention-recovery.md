# 2026-09-13：白話模型說明、開關診斷與原生 Attention 重訓分流

## 變更內容與原因

使用者無法區分目前選用技術與歷史候選。新增 reports/current-model 作為白話入口，說明權重、E2、BinaryQK 補償分支、PWL、MASF alpha／bridge、qSiLU 與未採用 HOG／RepConv／KD，提供 ASCII 架構圖並更新根 README 入口。依 finish-work 技能核對實際權重、架構與來源，不只重排舊文字。

使用者後續明確更正要移除 BinaryQK 補償分支後恢復 Attention 並重訓，因此另立 experiments/attention_recovery_v1，不把直接 FP-QK 切換稱成新重訓結果。Softmax 是保留 PWL 或恢復原生版本尚待確認，故沒有啟動訓練。

## 驗證方式與結果

CPU 重建同 SHA 權重，核對兩個 Attention 位置、16 個固定尺度 slots、qSiLU、無 HOG／RepConv。MASF alpha=0 只改一個 state；與 Identity 旁路 Detect 輸出一致；Pose 輸出完全一致；materialization 後開關仍有效；FP QK 公式精確一致；PWL 契約通過。

新增兩組完整 COCO val5000／BBAT5 v1 val683 GPU 驗證，兩組既有 MASF on 結果重用。兩組均成功；新結果 BBAT 六項與相應 MASF on 精確重現，權重 SHA 保持不變。數值與來源 hash 保存於 reports/current-model/switch-comparison.json。無重新訓練、無 split 變更、無來源模型覆寫。

## 困難與解法

sandbox helper 的 bwrap 無法啟動，讀取與既有文件編輯採經授權工具；修改仍使用 apply_patch。專案部分 AGENTS/domain 檔位於父層，改讀父層有效文件。

使用者改變方向時 queue 已在執行；加入 STOP_AFTER_CURRENT 防止未來重啟，但已載入程式不會因磁碟更新即時改變，两組皆正常完成。保留有效診斷，明確標示非重訓。沒有終止或重跑其他正常工作。

## 未解事項與風險

等待 Softmax 契約選擇後才啟動新訓練。移除 BinaryQK 不能只換 score；還需移除補償分支、正確映回 QKV／BN 並防止舊驗證器重建回二值化。恢復精度沒有保證，須同資料、同指標獨立驗證。沒有 commit、push 或刪除舊结果。
