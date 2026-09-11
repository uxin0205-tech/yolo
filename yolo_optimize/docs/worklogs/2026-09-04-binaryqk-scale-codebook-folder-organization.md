# 2026-09-04：BinaryQK scale／codebook 獨立資料夾整理

## 工作目的

依使用者要求，將「少量或8個 scale如何兼顧 BinaryQK精度與硬體」從一般 BinaryQK精度恢復文件
拆成獨立優化方向，讓後續每個優化假設都有自己的入口、計畫、架構圖、證據與停止條件。

## 變更內容與原因

1. 建立
   [`optimizations/binaryqk-scale-codebook/`](<../../optimizations/binaryqk-scale-codebook/README.md>)：
   - `README.md`：定義 P2條件式定位、C0／B4／A8決策與範圍。
   - `plan.md`：定義 cached replay、target-kernel、完整 validation與 matched QAT gate。
   - `architecture-report.md`：以 terminal圖呈現目前 C0、預計 B4與條件式 A8資料流。
   - `evidence.md`：集中本地 metrics、程式來源、算術上界與尚未證明事項。
2. 保留
   [2026-09-03 完整研究報告](<../research/2026-09-03-binaryqk-scale-codebook-hardware.md>)
   在 `docs/research/`，由新資料夾連結；沒有複製第二份研究全文，避免來源漂移。
3. 將一般
   [BinaryQK精度恢復 README](<../../optimizations/binaryqk-accuracy-recovery/README.md>)、
   [計畫](<../../optimizations/binaryqk-accuracy-recovery/plan.md>)與
   [架構圖](<../../optimizations/binaryqk-accuracy-recovery/architecture-report.md>)中的 B4／A8重複細節
   改為短摘要與新方向連結，使 P1主線維持 site isolation → QAT → ranking KD／STE-window。
4. 更新[優化方向索引](<../../optimizations/README.md>)、[子專案 README](<../../README.md>)與本工作紀錄
   索引。P1與 P2現在分列，啟動條件不再混在同一列。
5. 未移動、刪除或改寫原始 metrics、checkpoint、trace與資料集；未修改 production code，也未執行
   GPU訓練、完整 validation或 target-kernel benchmark。

## 驗證方式與結果

- 新資料夾完整性：`PASS`；README、計畫、架構圖、證據共4份文件存在。
- Markdown local-link checker：`PASS (12 files)`；本次涉及的方向文件、索引、README與工作紀錄
  所有本地連結皆存在。
- scale/codebook算術 assertions：`PASS`。以 `S=2,B=2,H=4,N=400,D=32` 重新得到：
  - token pairs `1,280,000`、C0 basis terms `2,560,000`、token slots `12,800`。
  - A8 `409,600` abs、`396,800` reduction adds、3-bit indices最小 `4,800 B`。
  - C0 `16` fixed slots、B4 `64` fixed slots及 `10,240,000` partial terms。
  - global dynamic關閉 FP–PoT缺口 `4.5767%`。
- 重複與格式檢查：`PASS`；P1不再保存完整 B4／A8架構，且相關文件沒有反引號誤跳脫或
  trailing whitespace。
- `python3 scripts/audit_accuracy_regressions.py`：依預期 exit `1`並保留4個既有紅燈；
  YOLO26 A-FINAL相對 B26-FP仍為 `-0.011641`。這是未解的既有精度回歸，不是文件整理錯誤。
- 未執行 GPU training、完整 detection validation、target-kernel benchmark、資料集或 production
  model修改。

## 遇到的困難及解法

- 困難：研究全文若直接搬入新方向，既有研究索引與歷史工作紀錄會失去穩定連結；若複製又會形成
  兩份可能漂移的權威內容。
- 解法：研究報告保留在 `docs/research/` 作唯一完整證據，新方向以 `evidence.md`建立可追溯入口。
- 困難：scale子矩陣原本散落在 P1 README、plan與 architecture-report，容易誤讀成 P1首輪工作。
- 解法：原文件只保留啟動條件與短摘要；C0／B4／A8細節集中到 P2資料夾。
- 困難：sandbox偶發 `bwrap: loopback: Failed RTM_NEWADDR`，一般本地命令無法啟動。
- 解法：只對必要的 repository讀取、patch與驗證使用受控權限，未擴大到外部資料、訓練或部署。

## 未解事項與風險

- 尚未實作 C0 early-return、B4/A8 reference、cached replay或 target kernel。
- 尚無 B4／A8完整 detection mAP與 matched QAT證據，方向狀態仍是 `proposed`。
- B4可能因4-way partial popcount失去速度；A8可能因 reduction、index與 variable shift超出硬體預算。
- 既有 global dynamic僅關閉 FP–PoT缺口 `4.5767%`，任何方案都不能先承諾補回約 `0.011`。
- 本次沒有清理候選：沒有生成 cache、checkpoint或大型 artifacts，因此沒有需要刪除的內容。
- 未修改資料集；未來若用 BBAT5驗證，仍只能使用不可變的 Canonical BBAT5 v1正式入口。
