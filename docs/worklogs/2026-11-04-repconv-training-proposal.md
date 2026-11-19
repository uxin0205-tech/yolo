# 2026-11-04：Rep folding 說明與訓練算子調整提案

> 後續更正：使用者要求核對 RepConv 本身的原本訓練方式。本文中的 SiLU／Softmax 切換是代理過度推論，未獲確認、未執行，提案已直接刪除。以 [訓練方式澄清](../research/repconv-training-clarification-2026-11-04.md) 為準。

## 任務與範圍

使用者詢問 Rep 是否維持推論架構，提議訓練改用原本算子，明確要求「先幫我調整等我確認再開始做」。本輪只整理可覆核提案，保持 Stage8 完成／停止與 Stage9 未啟動，不將提案當成執行授權。

## 變更內容與原因

新增 原算子切換提案（已撤回刪除） 與 原參考 SHA 紀錄（已由 [現行核對與刪除清單](../research/repconv-training-audit-2026-11-04.json) 接續），同步正式計畫、根層／架構資料夾 README、研究索引及工作紀錄索引。提案暫按原生 SiLU／exact Softmax 訓練、部署轉回 qSiLU／D1 理解，仍待使用者確認；第一輪建議只做 Rep17＋20，5 輪，不自動展開全部候選。

先唯讀核對 optimize Rep 的方法、設定、結果及現行實作。確認 optimize 原本也使用 qSiLU／PWL；本提案是新方法。Rep folding 保證同一已訓模型在相同 activation 下的線性折疊等價，不能保證重新訓練後 AP 不變，也不能保證 SiLU／Softmax 替換為近似函數後等價。

## 驗證方式與結果

查閱 RepVGG 原論文與本地 folding、activation、D1 前向／proxy 來源。原論文網頁初次直接開啟回應 403，改用搜尋取得 CVPR 官方索引內容，不依賴非官方轉述。確認 supervisor 為 complete、worker_pid 為 null、下一階段未啟動；GPU compute process 清單為空。

現行程式、protocol、權重與來源鎖未改。只做文件連結／格式與來源 SHA 核對，409 個已鎖定程式／配置及 26 項事前參考 SHA 均一致，沒有 CPU 模型實驗、正式 AP、訓練或 GPU 工作。

## 困難與解法

「原本算子」尚未明確指涉範圍；已用簡短選項釐清，文件以暫定理解標為待確認，沒有默認回退浮點 QK 或更換 parent。訓練與部署使用不同非線性時精度可能改變，因此提案分開列切換前後 AP，最終以部署版本選型。

## 未解事項或風險

待使用者確認原生算子訓練範圍與第一輪提案。原生版本是否較準、換回 qSiLU／D1 後是否保留收益尚無本輪證據；不能保證 person 回升。使用者確認之前不實作或啟動實驗。
