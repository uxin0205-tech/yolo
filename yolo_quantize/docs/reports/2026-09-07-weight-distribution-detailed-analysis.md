# SD4／三元權重選擇依據與既有實驗詳細分析

日期：2026-09-07。這是[全模型盤點與四天計畫](2026-09-07-full-model-audit-four-day-plan.md)的詳細補充，兩份合讀。此次只整理既有證據、查核定義並生成圖表，沒有新增 GPU 實驗、訓練或 live queue。

## 1. 先回答：為什麼比較 SD4 與三元？

你的核心方向是對的：不同層的權重分布不同，因此不應要求 backbone、neck、head 全部使用相同低位元格式。但「一個越接近零越好、一個越分散越好」不能直接當作選擇規則。要同時看**尺度正規化後的碼本匹配、量化造成的能量損失，以及模型對該層誤差的敏感程度**。

| 格式 | 本專案表示方式 | 值得測試的分布特徵 | 不能直接推出的結論 |
|---|---|---|---|
| Fixed-SD4 | α × {0, ±1/64, ±1/32, ±1/16, ±1/8, ±1/4, ±1/2, ±1}；15種值、4bit碼 | 靠近零仍需保留多種非零幅值，且幅值跨尺度；對數間隔可能比均勻間隔更合適 | 小權重多就一定勝過W4；任意分散分布都適合 |
| LS-SD4 | 相同SD4碼本，QAT更新scale | 初始尺度需要與權重、activation、任務共同調整 | 可學習scale等於可任意移動15個碼點；現有CPU表已測出LSQ效果 |
| 三元 | α × {−1, 0, +1} | 可捨棄的近零權重較多，剩餘非零幅值能由同一組內的一個α代表 | 零比例越高越好；非零幅值越分散越好 |
| W4–W8 | 帶scale的均勻整數格點 | 需要較均勻的絕對精度，或特殊格式造成任務誤差 | W4的位元數與SD4相同，所以所有層一定表現一樣 |

這裡 SD4 定義取自 `src/yolo_quantize/weight_formats.py` 的 `_sd4_candidate`，不是泛指其他論文的所有「SD4」。它在零附近格點較密，但大幅值附近格點較疏。**三元的非零幅值越分散，單一α通常越難代表它們**；若你指的是跨數個數量級的幅值分布，SD4是值得檢驗的候選，但離群值也可能拉壞scale。

「絕對數值小」本身也不足：將整層權重和scale一起乘相同比例，理想無約束縮放下的相對重建誤差不變。因此需要看相對於每個量化group尺度的形狀，而非只比較不同層的原始數值大小。

## 2. 三元的關鍵不是製造最多的零

TWN以帶scale的三元值近似全精度權重，核心是最小化重建距離；不是單純最大化零的比例。[原始論文：Ternary Weight Networks](https://arxiv.org/abs/1605.04711)

固定非零集合 S 時，可用 α = mean(|wᵢ|, i∈S) 表示非零幅值；總平方誤差可以拆成：

`E = Σ(i∉S) wᵢ² + Σ(i∈S) (|wᵢ|−α)²`

第一項是歸零損失，第二項是非零幅值被壓成單一尺度的損失。即使很多權重接近零，只要它們累積能量高或位於敏感層，歸零仍可能傷害mAP。即使非零數量很少，幅值彼此差距大也可能不適合三元。

例如 `{0,0,−0.8,+0.8}` 可由三元精確表示；`{0,+0.05,+0.4,+1.0}` 的非零幅值無法由單一α精確表示。這是解釋用例子，不是本模型實測分布。

Paper-TWN、filterwise TWN、exact ternary不能合併成同一個結果：它們的閾值／尺度求法及group粒度不同。「exact」只表示特定重建目標下的尺度最佳化，不代表端到端任務最佳。也不要把三元權重、三來源資料融合或其他三源訓練演算法視為同一件事。

## 3. 目前究竟量到了哪些分布資料？

來源：`artifacts/reports/v36-qsilu-v35-parent-148-layer-9format-cpu-v1.json`。CPU reference 是 V35 learned parent，148個deployment Conv/Linear paths，每路9種格式，共1332筆。

現有資料有原始權重的min/max、mean、mean absolute、population std、exact-zero ratio；每種格式另有重建NRMSE、cosine、量化後零比例及估算bytes。本次已全部整理成[逐層格式CSV](../../deliverables/full-model-audit-2026-09-07/weight-format-evidence.csv)，不需重跑训练。

尚缺原權重的固定定義近零比例、分位數、完整histogram、各group非零幅值變異及丟棄能量。因此**目前不能聲稱已證明「某區因為近零密度高所以SD4勝出」**。量化後零比例也不能冒充原權重近零比例。本次圖是重建誤差圖，不是偽造的原權重分布圖。

CPU比較的另一個限制：W4–W8與SD4使用per-output-channel；exact ternary為per-tensor、Paper-TWN為layerwise，TWN-v3為filterwise。粒度較細可以有更多scale，誤差與metadata都會變；比較時必須分開記錄，不能全歸因格式。

## 4. 已完成的CPU結果如何解讀？

完整十區×9格式表見[數據附錄](../../deliverables/full-model-audit-2026-09-07/weight-evidence-tables.md)；[比較圖PNG](../../deliverables/full-model-audit-2026-09-07/weight-format-nrmse.png)另有PDF/SVG可分享。

**148路中41路的靜態optimal SD4 NRMSE低於同粒度W4**。因此SD4有局部擬合價值，但沒有支持整個模型一律SD4；這41路也不是已驗證mAP勝出的41路。

例：`graph.model.0.conv`，CPU NRMSE如下：

| W8 | W4 | Fixed-SD4 | exact ternary | filterwise TWN-v3 | Paper-TWN |
|---:|---:|---:|---:|---:|---:|
| 0.004585 | 0.088339 | 0.156046 | 0.559921 | 0.431578 | 0.683394 |

該層SD4重建並不如W4，而三元误差更大。因此不應因為全模型某些SD4層有效，就把backbone入口也直接改SD4或三元。但這仍是weight-only證據，不是該層mAP下降量。

NRMSE = ||W−Q(W)||₂ / ||W||₂；例如0.156表示誤差RMS約為參考權重RMS的15.6%，**不是mAP下降15.6%**。圖表以區內path中位數彙整，避免大層完全掩蓋小層；代價是它不代表全區能量加權誤差，逐層CSV應一併檢視。

## 5. 已做PTQ能不能支持候選選擇？

已完成特殊格式12格、uniform12格、final1格，25格的完整total delta在附錄。特殊格式是backbone/neck/head各取格式排序top4路徑，而不是十區全量替換；不同格式可能改到不同路徑。

- special-backbone：三種三元候選都reject，最差指標接近全面崩潰；目前不應直接整組送短QAT。
- special-neck：Fixed-SD4、filterwise TWN為recover；其餘三元reject。
- special-head：Fixed-SD4、filterwise TWN為recover；其餘三元reject。
- 12格uniform與final為reject。但uniform是累積配置，不能當作獨立W6/W5等格式無效的證據。

以neck Fixed-SD4為例，total mAP50／mAP50–95變化為−1.657／−4.053百分點，並沒有通過−1.5／−4百分點門檻。先前較小的−0.637／−2.973百分點是incremental，不是包含activation的總損失。

重要限制：這批PTQ使用早期qSiLU recovery checkpoint，CPU排序用V35 learned parent。它可以保留為歷史風險訊號，但**不足以完成同parent的layer sensitivity因果比較**。這也是下一輪應先統一parent，而不是立刻多跑幾十組QAT的原因。

## 6. 已做QAT：目前確實有可用成果

V36已完成3epochs；148個deployment權重路徑全部fake quant：133 W8、13 LS-SD4、1 W6、1 W4；activation為qSiLU＋LSQ+ A8，共124個quantizers。目前這組沒有三元權重，因此不能聲稱三元已通過這次QAT。

| epoch index | 最差mAP50總下降（pp） | 最差mAP50–95總下降（pp） | 搜尋門檻 |
|---|---:|---:|---|
| 0 | 1.847 | 2.410 | 不通過 |
| 1 | 0.835 | 1.157 | 通過 |
| 2 | 0.976 | 1.145 | 通過 |

16項任務數字與accepted逐列對照見附錄，包含COCO、COCO Person、BBAT box/pose與ball/bat分項。epoch1的COCO mAP50／50–95為0.662372／0.486455，Person為0.834406／0.611734。這是搜尋集成果，尚不是formal final成績。

這證明「已有一個全deployment權重fake-quant混合配置，在短QAT後通過搜尋門檻」。**不證明SD4相對W4更好，也不證明LSQ本身帶來多少提升**，因為未做同parent、同routes的固定scale／學習scale配對。CPU fixed-sd4與QAT LS-SD4不可直接當成該消融實驗。

另外，全模型純整數仍需protected modules、activation邊界、add/concat/attention/decode、scale與packing及實際backend驗證。float checkpoint大小不是4bit部署大小。

## 7. 把你的分布假設變成可驗證的實驗

先鎖同一個驗證過的parent及既有canonical資料，CPU分布、probe、PTQ與QAT都記錄相同hash。每層允許不同格式，但不另訓練一個baseline、不改資料split。

| 層次 | 記錄／比較什麼 | 決策用途 |
|---|---|---|
| 原權重分布 | 每group尺度正規化；P(|w|≤0.1 RMS)、P(|w|≤0.25 RMS)、P50/P90/P99、max/RMS、正負不對稱 | 判定近零集中、尾部、跨尺度；這些閾值是診斷定義，不是準確率gate |
| 格式投影 | SD4與三元的零比例、各碼占比、歸零能量占比、非零幅值變異、NRMSE/cosine、含scale的bytes | 判斷錯誤來自歸零、非零幅值壓縮還是outlier |
| 單層probe | 固定activation輸入，逐路替換後的輸出NRMSE／cosine | 找出weight error小但輸出敏感的層；不叫mAP驗證 |
| 十區PTQ | 同parent一次只改一區；跨格式盡量使用相同paths，另外標明自選top-k實用路由組 | 分離格式差異與選層差異；全16項total delta |
| 少量短QAT | 同parent/routes比較固定SD4與LS-SD4；三元僅給可恢復區域 | 驗證scale學習與任務恢復，而非替每個CPU候選訓練 |
| 累積驗證 | backbone鎖定後再neck，再head；每次重算總gate | 排除獨立winner拼接失效，維持全權重覆蓋 |

三元保留至少一個與SD4相同group粒度的控制組；若使用Paper-TWN原始粒度，明記「演算法＋粒度聯合比較」。新增控制格從既有擴展格替換，不增加總時間上限。

候選選擇採精度／估算儲存的非支配集合：先保留通過總門檻者，再比較壓縮與實際可支援的部署成本；recover只作短QAT候選。無實測kernel時不以理論低bit宣稱GPU更快。特殊格式優先，W6/W5及必要W7/W4接續，不顛倒你要求的順序。

## 8. 四天計畫如何吸收這次分析？

沿用主報告96小時窗口與最多6組×3epochs，不額外擴大訓練量：Day1補原分布統計及parent一致性；Day2特殊格式十區PTQ與固定probe；Day3依敏感度做逐區累積、再uniform；Day4最多兩個finalists驗證與報告。

固定SD4／LS-SD4公平配對優先占用短QAT預算；三元只有在同parent的局部PTQ可恢復且有壓縮價值時才占其餘名額。CPU分布分析取代盲目多跑訓練，不保證在四天完成所有148路×所有格式的mAP或QAT。

本次**完成的是報告與證據整理**；新分布histogram、同parent完整敏感度矩陣、上述配對QAT、formal與整數部署仍未完成。四天DAG仍為planned_not_enqueued，沒有假稱queue已啟動。

## 9. 交付與驗證

- [完整數據附錄](../../deliverables/full-model-audit-2026-09-07/weight-evidence-tables.md)：十區×9格式、25個PTQ、epoch1全16指標。
- [1332筆原統計與格式誤差](../../deliverables/full-model-audit-2026-09-07/weight-format-evidence.csv)：可自行篩選區域與path。
- [比較圖PNG](../../deliverables/full-model-audit-2026-09-07/weight-format-nrmse.png)、[PDF](../../deliverables/full-model-audit-2026-09-07/weight-format-nrmse.pdf)、[SVG](../../deliverables/full-model-audit-2026-09-07/weight-format-nrmse.svg)。英文圖避免字型相依，中文定義在本文與附錄。
- [工作紀錄](../worklogs/2026-09-07-weight-evidence-supplement.md)：生成器驗證與限制。

這次沒有修改訓練程式、檔案清理、GPU程序或Git發行。詳細資料可分享，但須連同比較限制，不能只截取最好的數字作最終結論。
