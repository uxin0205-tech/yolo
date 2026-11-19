# 2026-11-04：更正 Rep 訓練方式解讀

## 任務與範圍

使用者要求依 RepConv 原本方式訓練，詢問代理原先實際怎麼做。先回答實際程式與紀錄，維持等待確認，不啟動訓練。

## 變更內容與原因

新增 [實際做法核對](../research/repconv-training-clarification-2026-11-04.md) 與 [來源／刪除紀錄](../research/repconv-training-audit-2026-11-04.json)。撤回並直接刪除代理自行推導、未執行的原生 SiLU／Softmax 訓練提案及其獨立參考 JSON；必要錯誤原因及來源 SHA 已提取。修正根層／架構 README、正式計畫、研究索引與 AGENTS 的錯誤方向，保留「確認前不開始」邊界。先前工作紀錄以更正註記標為歷史，不保留另一套實驗方案。

## 驗證方式與結果

唯讀比對 Ultralytics RepConv、optimize from_conv／configure／folded、實際 config，以及本輪 rep／model／training／evaluate 和三組已完成 result。確認 live 與 EMA 使用多分支訓練，副本才 folding；本輪使用自訂類別，但兩分支線性結構與原實作一致。兩個位置 stride=2，不適用 identity 分支。原生 RepConv 支援自訂 activation，採原始分支方式不要求還原 SiLU／Softmax。

實際配方為 Rep＋兩 heads、BN running 固定、AdamW、Rep LR1e-5／heads5e-6、每組5輪；Pose weight為 .195／.5／.215。三組新分支更新與固定 state guard 均通過。這是 optimize 式局部適應，不能泛稱完整原版訓練流程。

本輪只修改文件，程式／protocol／模型權重未改，沒有 CPU 模型實驗或 GPU 工作。檢查文件入口與原來源鎖 SHA，確認原實驗不受影響。

## 困難與解法

前次代理過度推論使用者要切換全模型非線性；本次明確更正，依實際程式區分 Rep 結構、初始化與訓練配方，不再把尚未確認的解讀當成使用者意圖。

## 未解事項或風險

後續調整原生模組、BN、可訓範圍及訓練預算需先呈現具體方案並由使用者確認。現有證據無法把 person 微降歸因於單一選擇；目前沒有新增實驗結論。
