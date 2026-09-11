# 2026-09-02：BinaryQK 精度恢復方向整理

## 工作目的

針對 YOLO26 BinaryQK 相對 B26-FP 下降約 `0.011641` mAP50-95 的問題，找出較可信的原因、可補回
精度的方法與最小必要實驗，並依 repository 的「一個方向一個資料夾」契約保存，不與 MASF、
RepConv、PTQ 或資料變更混在同一輪。

## 變更內容與原因

1. 新增 [BinaryQK 方向 README](<../../optimizations/binaryqk-accuracy-recovery/README.md>)：
   - 整理 B26-FP、W-DIR、V1-BR、A-FINAL 與舊 YOLO11 recovery 數據。
   - 以 residual decomposition 說明 global Q/K magnitude 為何會丟 token-pair ranking。
   - 將 hardware-first 候選排序為 fixed-PoT site isolation/hybrid、matched direct QAT、單一
     FP-teacher KD；per-token/head dynamic 降為條件式 accuracy ceiling。
   - 明記 W-DIR 已是 full-model direct recovery，且最佳 checkpoint 已清除，避免提出不可執行的
     resume 計畫。
2. 新增 [最小實驗計畫](<../../optimizations/binaryqk-accuracy-recovery/plan.md>)：
   - 重用既有 V1-DYN／V1-SHEAD／V1-P2 scale screen，只新增 `ISO-10-BIN`、`ISO-22-BIN`
     兩個 validation-only jobs。
   - 只讓唯一 winner 與 matched `FP-CTRL` 做 direct QAT；首 seed 失敗立即停止。
   - ranking/KL 仍有明確問題時才增加一個 KD arm，不做大矩陣 sweep。
3. 新增 [終端架構圖](<../../optimizations/binaryqk-accuracy-recovery/architecture-report.md>)：
   - 以 ASCII 圖表示目前 two-site/fixed-PoT、預計 single-site hybrid、teacher KD，以及條件式
     per-token／clipped-STE 修正。
   - 列出 per-token scale 共用一套 scale時的 `6,400` 個 token scales／`1,280,000` 個 pair-scale
     operations下限，以及 two-basis獨立時 `12,800`／`2,560,000` 的上界。
4. 新增 [第一手來源研究](<../research/2026-09-02-binaryqk-accuracy-recovery-primary-sources.md>)：
   - 外部只採原始論文、作者 repository/source與 PyTorch、TensorRT 官方文件。
   - 明確區分論文證據、本地證據與尚待驗證的本專案推論。
5. 更新根 README、優化方向索引、研究索引與本工作紀錄索引，讓後續可直接找到本方向。
6. 更正 2026-08-31 整體研究內已失效的 W-DIR resume／擴大 trainable-scope 建議，同步更新結論、
   recovery 優先順序、最小矩陣與立即決策；保留原始 metrics，並加上最新版方向與計畫連結。

## 本地診斷與驗證方式

### 既有 accuracy 回歸稽核

執行：

~~~bash
python3 scripts/audit_accuracy_regressions.py
~~~

初次診斷結果：腳本正常回傳 `1`，共 4 個既有 fail-closed 紅燈。BinaryQK 重要數值為：

- YOLO11 raw sign：`0.459297`，相對 E0 `-0.051554`。
- YOLO11 T6-F/A：相對 matched FP `-0.000830`。
- YOLO26 W-DIR：`0.507457`，相對 B26-FP `-0.010540`。
- YOLO26 A-FINAL：`0.506357`，相對 B26-FP `-0.011641`。

回傳 `1` 是預期的紅燈語義，不是腳本 crash；本次只整理方案，沒有產生新 AP，所以紅燈不應被
人為改成綠燈。

### CPU 機理 probes

父權重：`/home/uxin/yolo/yolo_attention/weights/yolo26m.pt`；兩張 COCO images、640×640、CPU。

per-token/head dynamic scale 相對 global dynamic scale（不是正式 fixed-PoT inference）：

| site | top-10 overlap | KL | 結論 |
|---|---|---|---|
| `model.10.m.0.attn` | `0.4206 → 0.5119` | `0.5431 → 0.4651` | 同時改善 |
| `model.22.m.0.1.attn` | `0.4962 → 0.5952` | `0.8810 → 0.6188` | 同時改善 |

每-head sign positive ratio 約 `0.44–0.57`，沒有嚴重失衡；`|Q/K|>1` 的 clipped-STE saturation
約 `0.35–0.54`，因此 threshold 暫不優先，positive pre-sign scale／learnable STE window保留為
條件式方法。這些 probes 只有兩張影像，沒有正式 AP 或 machine-readable artifact，不能當成功證明。

後續直接唯讀載入 retained V1-BR 與 N1-SHIFT/A-FINAL checkpoint，確認兩者皆為
`scale_mode=power_of_two`、`fixed_coefficients_ready=true`。實際 fixed coefficients：

- site 10：四個 heads 的 `[identity, Hadamard]` 都是 `[0.25, 0.25]`。
- site 22：head 0 是 `[0.125, 0.25]`，head 1–3 都是 `[0.25, 0.25]`。

因此正式 inference 是 fixed per-head/basis PoT，不是 per-image dynamic；原 CPU probe 只能支持
per-token granularity 候選。hardware-first 首輪不重跑 `DYN-GLOBAL`；只有前面 recovery 均失敗且
接受 dynamic-scale成本時，才以 `TOKEN-BOTH - DYN-GLOBAL` 的成對實驗做正確歸因。

### 既有 scale 與 per-token-like 消融重新稽核

YOLO26 同一 W-DIR parent 的 evaluation-only scale screen 已完成：

| Run | scale | mAP50-95 | 相對 V1-DYN |
|---|---|---:|---:|
| V1-DYN | per-sample/head global dynamic | `0.507457` | `0` |
| V1-SHEAD | calibrated fixed per-head | `0.506879` | `-0.000579` |
| V1-P2 | calibrated fixed per-head/basis PoT | `0.506952` | `-0.000506` |

PoT 差距在既定 `0.001` tie band內；使用者又明確要求避免每-image scale重算，因此不再把 global
dynamic列為首輪 production候選。舊 YOLO11 E1-S `0.45930` → global scaled-sign `0.48056`，以及
T1 sign QAT `0.50789` → T2 global scaled-sign QAT `0.50951`，證明 magnitude scale 有幫助；但
T3/T4/T5 同時改 residual basis、threshold與 QK products，N4 又是加法 side channel，均不是乾淨的
per-token multiplicative scale 單因子。YOLO26 T5-SCR `0.492907` 也未勝 H-SCR `0.495243`，同樣有
basis confound。因此相關歷史結果應重用，不再重跑；真正尚缺的是兩個 site 的 fixed-PoT isolation。

### fixed-mode eager 冗餘稽核

唯讀檢查 `yolo_attention/src/yolo_attention/binary_basis.py`：`_coefficient()` 目前先執行
`_dynamic_coefficient()`，之後 fixed mode才回傳 `_fixed()`。所以 PyTorch eager reference確實仍會
對每張圖做不影響輸出的 magnitude reduction；Hadamard兩個 bases合計每 site四次、兩 site八次。
這是控制流冗餘，不是 fixed-PoT硬體需求。
方向計畫已新增 early-return／bit-true parity gate，要求正式 profiling與 export前移除此冗餘。本次
沒有取得修改 production code的授權，因此只記錄與排入工程 gate，未修改 `yolo_attention`。

### 文件驗證

完成修改後執行：

- Markdown 相對連結存在性檢查。
- direction contract 檢查：`README.md`、`plan.md`、`architecture-report.md` 齊全。
- 關鍵敘述檢查：不得把 W-DIR 寫成 attention-only，不得要求 resume 已刪除 checkpoint，不得把
  per-token scale 說成 BinaryAttention 官方原樣方法。
- 重新執行 accuracy regression audit；預期仍回傳 `1` 並保留 BinaryQK 紅燈。

最終驗證結果：

- Markdown local-link checker：`PASS`，新增與同步文件的相對連結都存在。
- direction contract：`README.md`、`plan.md`、`architecture-report.md` 三檔齊全。
- hardware-first 關鍵內容 assertions：`PASS`；正式 fixed-PoT scale、兩站實際係數、既有 scale
  screen重用、首輪 2 個 site-validation jobs、conditional per-token、two-basis成本上界與 eager
  early-return限制一致。
- checkpoint scale assertion：V1-BR 與 N1-SHIFT/A-FINAL 都通過 `power_of_two`、ready flag及兩站
  fixed-coefficient 精確值檢查。
- trailing-whitespace `rg`：無 matches；命令回傳 `1` 是「沒有找到違規空白」的正常結果。
- accuracy regression audit：如預期回傳 `1` 並保留 4 個紅燈；其中 A-FINAL 仍是 `-0.011641`。
- 未執行 GPU training、完整 validation、資料集改動或 production model 修改。

## 遇到的困難及解法

- 困難：預設 sandbox 多次出現 `bwrap: loopback: Failed RTM_NEWADDR`，連唯讀命令也無法啟動。
- 解法：只對必要的唯讀稽核與 repository 內文件操作使用受控的 escalated command；未擴大到 GPU、
  網路下載、資料集或 production model 修改。
- 困難：W-DIR 報告仍留有最佳數字，但 cleanup 已永久移除 checkpoint，舊 next-step 文件仍建議
  從 W-DIR resume。
- 解法：交叉核對 training audit、cleanup report與 retained checkpoint；將 V1-BR 限定為當前診斷
  parent，正式 QAT 改由 final same-lineage FP winner 建立。
- 困難：初版方向文件將 `_magnitude()` 的 global dynamic 計算誤寫成正式 V1-BR/A-FINAL inference
  scale；實際 variant與 checkpoint 都是 `power_of_two` fixed coefficients。
- 解法：直接檢查 generated variants、`BinaryScore._coefficient()` 與兩個 retained checkpoints，
  更正 calibration／inference資料流、補實際係數；再稽核既有 scale screens後撤回首輪新增
  `DYN-GLOBAL` 的建議，只把它保留為 per-token條件式配對 control。
- 困難：初版把兩張 image 的 per-token probe 排為第一優先，沒有充分反映 production 不接受
  每-image reduction與 N² pair-scale epilogue 的限制，也重複了既有 global scale消融。
- 解法：依使用者提出的硬體約束重新排序；首輪只新增兩個 fixed-PoT site-isolation validation，
  per-token改為 hardware-friendly方法失敗後才開啟的 accuracy ceiling。
- 困難：第一次 checkpoint assertion 的 `python -c` 將 `\n` 當成 literal，造成 `SyntaxError`。
- 解法：改成無內嵌換行的單行 list-comprehension assertion；重跑 exit `0`，兩個 checkpoints數值一致。

## 未解事項與風險

- 尚未執行 2 個 fixed-PoT site-isolation validation 或任何 GPU QAT；per-token 改善目前只是機理訊號。
- V1-BR 已共同適應兩個 binary sites，還原單站 FP 的結果是 hybrid deployment診斷，不是乾淨因果證明。
- per-token 的 pair-scale epilogue 可能侵蝕 binary kernel latency，需 custom/tiled kernel與 target profile。
- FP teacher、learnable STE window與 threshold 的 YOLO26 收益都尚未由 matched multi-seed 證明。
- 現有 YOLO26 metrics 是固定 evaluator 下的 Ultralytics internal metric，不應寫成官方 COCO API AP。
- 無資料集修改；若未來使用 BBAT5，只能依 `bbat5-v1` canonical dataset規範執行。
