# 2026-09-04 論文第 3.1 節對 YOLO26M 的適配更正

## 變更內容與原因

- 重新按指定論文第3.1.1–3.1.4逐項分析模型預算、LR短篩、多階段transfer與WST/HOG/Canny filter-assisted training。
- 更正先前過度聚焦 Detect/Pose gradient conflict 的定位：衝突投影仍有本地量測依據，但它是獨立方向，不是論文第3.1節最直接的移植。
- 新增完整研究報告 `docs/research/2026-09-04-paper31-to-yolo26m-training-adaptation.md`。
- 新增獨立方向 `optimizations/p3-hog-companion-training/`，包含：
  - `README.md`：方法、推導、範圍與最小消融；
  - `plan.md`：實作不變量、probe、F0/F1/F2、gate與停止條件；
  - `architecture-report.md`：論文原圖、目前Full35、建議訓練圖與部署圖。
- 同步修正根README、優化索引、研究索引、衝突安全方向與舊研究報告的定位。

這次選擇 `P3 Box-Aware HOG Companion Supervision`，原因是它保留論文「fixed filter輔助representation learning」的核心，但把HOG從永久輸入改成停止梯度的training-only target：不改3-channel RGB stem、不要求正式硬體每張圖重算HOG，且部署前可完全移除。

## 驗證方式與結果

- 核對指定PDF第3.1.1–3.1.4與原始MSFA論文／官方程式；確認官方filter路徑會灰階、串接filter channels並改backbone `in_channels`。
- 核對目前Full35 graph：shared layers 0–22，Detect/Pose皆消費`[16,19,22]`，P3為256 channels／stride 8，layer16輸出為post-MASF P3。
- 核對J0→J3 scope、AdamW role LR、persistent optimizer state、loss的batch-sum語義與full-resume／inference checkpoint差異。
- 文件檢查：確認新增方向三個檔案存在、Markdown code fences成對、無行尾空白，並檢查相對本地連結可解析。
- 結果：文件與方案驗證通過；未執行GPU訓練、AP validation、checkpoint修改或dataset修改，所以HOG增益仍是`proposed`假說，不是已驗證結果。

## 困難與解法

- 困難：原本報告把論文的「bridge」概念過度映射到本地gradient conflict，未直接回答filter-assisted training如何用在YOLO26M。
- 解法：保留舊報告作獨立衝突方向並加定位更正；另做逐節適配，只把最有機理且不改部署圖的HOG target列為本篇首案。
- 困難：論文PDF頁碼與檔案頁索引不同。
- 解法：依章節標題完整定位3.1.1–3.1.4，並用原始MSFA論文與官方程式交叉核對。
- 困難：一般 `apply_patch` 一度因 `bwrap: loopback: Failed RTM_NEWADDR` 無法更新既有檔案。
- 解法：改用相同工具的 escalated interactive stdin模式完成精確patch，未改用直接覆寫或破壞性操作。

## 未解事項與風險

- P3對tiny ball／細bat是否仍有足夠HOG有效cell尚未實測，是第一個go/no-go probe。
- `mu0`只能用事前固定的gradient-ratio安全校準，不能看validation反覆調整。
- F2若未勝F1，只能說generic companion supervision可能有用，不能宣稱HOG orientation有效。
- person head、MASF位置或RepConv若後續改變，必須以新final-lineage parent重跑本方向。
- 任何新BBAT5實驗仍只能使用Canonical BBAT5 v1；HOG target須on-the-fly產生，不得物化成新的資料版本。

返回[工作紀錄索引](<README.md>)。
