# 2026-09-08 方向1區域排序蒸餾完整規格

## 變更內容與原因

- 依使用者要求完整列出方向1，新增round2-innovation/region-ranking-full-spec.md：目標、joint老師／學生、資料、hook、梯度、配額、loss、實驗順序、成本與停止條件。
- 把流程收緊為先K0/K-UNIFORM，普通KD有訊號才K-REGION；有訊號後補K-FG確認是否超過普通foreground KD。不把joint合併損失誤稱可由同joint老師自動補回。
- 提出首版512 pairs、COCO／BBAT5配額、row標準化、teacher margin與μ校準規則，全部標為未實測提案；teacher／checkpoint／scope／recipe尚待第一輪，不假稱ready。
- 使用者指定Luna/max代理只同步5個既有入口／摘要；主代理負責規格與分析。僅Markdown，未實作模型或loss。

## 驗證方式與結果

- 唯讀核對第二輪既有計畫與attention.forward：score依序經既有progressive、relative bias再normalize；`last_scores`實際detach，因此不能作student可微KD輸入。
- 釐清Q/K權重凍結但STE／上游可學時，score仍可能適應；只有下游heads/P/V可學不能改進其上游ranking。此為梯度路徑推導，不是本次測試結果。
- 主代理以python3標準函式庫唯讀檢查本次7份Markdown、62個本地連結：路徑、fences、行尾空白、檔尾換行錯誤均0。子代理完成5份既有入口同步，其初檢的5個pending引用已由主代理新建正文後消除。未做CPU模型／數值測試、GPU、訓練、AP validation、calibration、checkpoint讀寫或profile。

## 困難與解法

- 預設sandbox曾有bwrap loopback初始化限制，唯讀exec與必要文件更新使用經審核的操作，不修改安全設定；仍只用apply_patch編輯。
- 既有方向只寫固定B／區域KD，欠缺eligible pool、tiny-object／空區域回退與task scaling；本次補為明確提案並保留未測標示。
- 本地診斷buffer已detach，直接沿用會讓KD無梯度；在規格中指定live-score tap與KD-only update驗證，未擅自改程式。
- 其餘：無。

## 未解事項或風險

- teacher是否有互補訊號、第一輪權重與可訓練scope尚未驗證；硬體凍結契約不得繞過。
- 固定pair上限不等於完整teacher等算力，區域mask不能增加已丟失的空間解析度，attention排序亦不是因果真值。
- 新超參數只有設計理由，沒有本地收益證據；不同KD normalization不可直接重用舊控制結果。
- person-only暫緩、BBAT5不改；無commit／push／刪除。只產生必要Markdown，沒有本次cache或模型可清理。
- 沿使用者額度限制不擴展新題目；本端無法讀帳戶週剩餘比例，未宣稱自動到73%停止。

返回[工作紀錄索引](<README.md>)或[完整規格](<../../optimizations/round2-innovation/region-ranking-full-spec.md>)。
