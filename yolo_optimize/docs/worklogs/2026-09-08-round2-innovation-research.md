# 2026-09-08 第二輪創新候選研究

## 變更內容與原因

- 因使用者要求更好的第二輪方向並希望有創新，新增round2-innovation入口／計畫與創新推導報告；不覆寫第一輪、不啟動person-only。
- 保留兩個獨立假說：任務保護的固定dual-basis選擇、固定token-pair預算的區域ranking KD。明確指出signed Hadamard、旋轉二值化與foreground KD已有先例，不冒稱全球首次。
- 指定Luna／max執行一手來源摘錄與本地實作事實盤點；主代理负责代數推導、創新界線與對照設計。
- 同步根README、優化／研究／工作紀錄索引。僅新增／修改Markdown，未修改production程式或資料。

## 驗證方式與結果

- 唯讀核對AGENTS、CONTEXT、canonical dataset ADR、joint配置／loss與BinaryScore的sign／coefficient實作。
- 原始論文／作者source核對ITQ、QuaRot、SpinQuant、FGD、CrossKD、LD、YOLOv10與YOLO26官方文件；不同架構與版本不可直接外推AP。
- 發現本地sign(0)=+1，因此sign(Dx)=Dsign(x)不能無條件套用；在推導中保留零值／INT8飽和例外。
- 本地盤點確認Pose26實際2×3、DFL-free、已有E2ELoss／TAL及RLE、Q/K為無activation affine且key_dim=32；hardware contract凍結Q/K與係數。據此區分研究主題優先與當前可執行優先，R2-BASIS未取得新契約前不標ready。
- 最終以python3標準函式庫唯讀檢查本輪9份Markdown：77個本地連結，路徑／fences／行尾空白／檔尾換行錯誤均為0；包含5份新增與4份索引更新。本次沒有CPU模型數值測試、cache replay、AP或latency結果。
- GPU工作0：沒有nvidia-smi、torch import、forward/backward、training、模型validation、calibration、checkpoint內容讀寫。

## 困難與解法

- sandbox bwrap loopback初始化失敗：唯讀exec使用經審核的升權，不修改安全設定；文件仍以apply_patch編輯。
- 初次猜測的engine／ADR／STE檔名不存在：用rg --files與實際模組定位，不將缺檔誤判為功能不存在。
- 部分CVF來源讀取失敗：使用同論文的arXiv或作者頁，不依二手摘要下結論。
- 使用者要求週額度73%left收尾，但可用工具未提供帳戶週額度。依OpenAI Docs核對介面status入口並明告限制，停止擴大研究範圍；未假稱能自動讀取或已到門檻。
- 其餘：無。

## 未解事項／風險

- 尚未完成系統性新穎性檢索，也沒有本地收益證據；「創新候選」不等於論文貢獻已成立。
- 第一輪R2-P／R2-T、Q/K硬體可重建契約、合法cache及區域KD細節尚需事前鎖定；第二輪保持proposed。
- 固定basis的理論FP等價不能直接推導binary／INT8等價或硬體latency不變。
- person-only暫緩、BBAT5 v1不變；無重切／抽樣資料View、無重標、無commit／push／刪除。
- 收尾盤點只新增必要Markdown，第二輪資料夾只有README／plan；沒有本次產生的cache、模型或暫存圖可提議清除。既有研究與未追蹤工作樹全部保留，不建立刪除清單或要求不必要的清理授權。

## 補充：Detect／Pose 已合併時，蒸餾是否仍有意義

- 使用者質疑合併訓練後蒸餾是否有用；本次只釐清適用條件，未變更實驗排程或實作。
- 唯讀核對fusion_model.py的DualHeadPredictionModule／GraphSharedDualHeadModel與joint_loss.py：共用trunk，任務head與criterion分開；訓練依所選task使用對應原生loss。這不是Detect head已在教Pose head。
- 原提案的teacher是另一份凍結FP-QK joint模型，student仍是雙head joint BinaryQK模型；旨在補二值化損失，不能自動修復FP joint本身的任務衝突。教師是否有較好的目標訊號、scope是否可更新，尚未以本地實驗證明。
- 同權重／算式／狀態／輸入下的完全相同teacher不提供初始差異；凍結後最多形成後續正則化訊號，不能當成新增能力的證據。兩個standalone teachers則是另一個遺忘／衝突研究，不混進本方案。
- 文獻依據：[Hinton等人的蒸餾原論文](https://arxiv.org/abs/1503.02531)；只支持teacher知識轉移的一般概念，不提供本地多任務收益證明。
- 驗證結果：程式文字路由查核完成；GPU、模型載入、AP／梯度測量均0。困難：无。未解風險：尚無合法FP joint teacher的配對優勢證據，因此方向1仍是條件式候選。

返回[工作紀錄索引](<README.md>)或[第二輪入口](<../../proposals/round2-innovation/README.md>)。
