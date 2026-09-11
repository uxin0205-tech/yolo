# 2026-09-03：BinaryQK 少量 scale／codebook 決策整理

> 2026-09-04 後續整理：本紀錄保留當日決策歷史；最新入口、計畫、架構圖與證據索引已集中到
> [OPT-BINARYQK-SCALE-CODEBOOK](<../../proposals/binaryqk-scale-codebook/README.md>)。

## 工作目的

回答「是否可準備8個或少量 scale給實際硬體使用，以補回 BinaryQK精度」；區分 codebook values與
runtime assignment，並在既有 fixed-PoT site isolation／QAT／KD主線下加入最小、條件式scale子矩陣。

## 變更內容與原因

1. 新增[少量 PoT scale／codebook硬體研究](<../research/2026-09-03-binaryqk-scale-codebook-hardware.md>)：
   - 比較 per-token K=4/8、fixed channel-group G=4/8與 per-image global K-codebook。
   - 依正式 `D=32,H=4,N=400`、兩 sites、兩 Hadamard bases推導 storage與 operation上界。
   - 明記 PoT只把乘法改成 shift的條件，不能直接宣稱整條 attention或 target latency變快。
2. 更新[方向 README](<../../proposals/binaryqk-accuracy-recovery/README.md>)：
   - 說明目前已有16個 fixed coefficient slots；單純改成8-entry固定 LUT只有壓縮效果。
   - 新增 `A8` per-token 3-bit PoT與 `B4` fixed 4-group PoT公式、成本與使用邊界。
3. 更新[最小計畫](<../../proposals/binaryqk-accuracy-recovery/plan.md>)與
   [架構圖](<../../proposals/binaryqk-accuracy-recovery/architecture-report.md>)：
   - 整體主線不變，仍先做2個 fixed-PoT site-isolation validations，再做 matched QAT／ranking KD。
   - 只有 magnitude殘差仍明顯才以 cached Q/K replay比較 C0、B4、A8；最多帶一個 winner進完整驗證。
4. 更新優化、研究與工作紀錄索引。本次未修改 production code、checkpoint或資料集，也未執行訓練。

## 主要推導與決策

- 現有 global dynamic相對 PoT只增加 `0.000505537` mAP，只關閉 FP–PoT `0.011045914`缺口約
  `4.58%`；因此 global 4/8-entry codebook不是大缺口解法。
- `A8` 每張圖仍需 `409,600` magnitude abs、`396,800` reduction adds、`12,800`個3-bit indices與
  `2,560,000`個 pair exponent／variable shifts；只作 fidelity ceiling。
- `B4` 不做 runtime reduction／selector，但64個固定 slots需要4-way partial popcount；naive上界由
  `2.56M`增至`10.24M` terms。它是 production-oriented首測，不是已證明加速的 winner。
- 若只建立8個固定 values而沒有 input-dependent或 group assignment，表示力不會增加；關鍵是
  assignment沿 image、token或 channel哪一軸變化，不是 codebook表長。

## 驗證方式與結果

本輪實際執行結果：

- 硬體算術 assertion：`PASS`。重新計得 current pairs `1,280,000`、basis terms `2,560,000`、
  A8 slots `12,800`、abs `409,600`、adds `396,800`、3-bit index `4,800 B`、B4 fixed slots `64`、
  B4/G8 partial terms `10,240,000`／`20,480,000`，缺口關閉率 `4.5767%`。
- Markdown local-link checker：`PASS`；方向、計畫、架構、研究與工作紀錄連結皆存在。
- Scale-codebook關鍵內容 assertions：`PASS`；A8／B4皆保持條件式範圍，沒有取代 site isolation主線，
  也沒有承諾補回`0.011`。
- trailing-whitespace `rg`無 matches；exit `1`是「沒有找到違規空白」的預期語義。
- accuracy regression audit如預期 exit `1`並保留4個紅燈；YOLO26 A-FINAL仍相對 B26-FP
  `-0.011641`。這是未解回歸，不是驗證腳本錯誤。
- 未執行 GPU training、完整 detection validation、資料集改動或 production model修改。

## 遇到的困難及解法

- 困難：使用「8個 scale」可能同時指8個固定 LUT values、8個 runtime choices或8個 channel groups，
  三者的成本與表示力不同。
- 解法：以「values是否固定」與「assignment何時決定」拆解；分成 A8動態 token index、B4固定 group
  assignment與最低優先的 global codebook。
- 困難：PoT常被直接等同免費 bit shift，但本地 `1/sqrt(32)=2^-2.5`，且 group或token方案仍有
  popcount segmentation、reduction、index、alignment及softmax型別轉換。
- 解法：只報 operation/storage上界，要求 custom-kernel bit-true parity與 target p50/p95 latency gate。
- 困難：sandbox偶發 `bwrap: loopback: Failed RTM_NEWADDR`，連唯讀程序也可能無法啟動。
- 解法：只對必要的本地唯讀稽核與 repository內文件 patch使用受控權限；未擴大到訓練或外部資料。

## 未解事項與風險

- 尚無 A8／B4完整 detection mAP、QAT或 target-kernel結果，不能宣稱能補回約`0.011`。
- B4的4-way partial popcount可能抵銷 binary QK收益；A8的variable shift/index traffic也可能成為瓶頸。
- Full35正式多任務metrics與先前單任務YOLO26 scale ablation是不同 lineage，不得混成直接差值。
- 未修改資料集；未來若以BBAT5驗證，仍只能使用不可變的`bbat5-v1`正式入口。
