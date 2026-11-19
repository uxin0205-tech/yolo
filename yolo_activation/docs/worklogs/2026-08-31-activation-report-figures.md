# 2026-08-31：Activation 四項精度、中央誤差與運算量圖表

## 任務與範圍

依使用者要求，把 activation 階段的成果整理成可直接閱讀的圖表，主精度圖只呈現 `COCO Δ`、
`Person Δ`、`BBAT box Δ`、`BBAT pose Δ` 四項，保留 SiLU 作為零差異基準，並排除表現過差、
不再參與決策的 ReLU。另放大中央函數誤差，加入運算量解析估計，並同步更新權威報告與入口文件。

本次只處理既有結果的讀取、核對、繪圖與文件整理；沒有啟動訓練、GPU job、PTQ 或 QAT，也沒有
改動 COCO2017、Canonical BBAT5 v1 的影像、標註、正式 split、YAML 或 registry。

## 變更內容與原因

1. 新增 `scripts/generate_activation_report_figures.py`，讓精度、函數誤差及運算量圖表都能從既有
   machine-readable 證據重新產生，不需手抄數值。
2. 新增四項精度長條圖。圖分為 Bit-True zero-shot 與 10-epoch recovery，並各自提供全範圍及
   SiLU 基準附近的放大視窗；比較 SiLU、Hardswish、qSiLU-PQ、SIPA Quality、SIPA Shift。
3. SiLU 明確列為 accepted baseline，四項差異皆為 `0.000000`。ReLU 不出現在新圖、CSV 與最終
   比較表，但歷史數值仍保留於原始 JSON，避免破壞實驗可追溯性。
4. 新增中央函數與 signed error 放大圖。上半部比較函數形狀，下半部同時提供全部候選誤差與
   qSiLU／SIPA 的 ±0.03 局部視窗，解決原圖比例過大、細微差異看不清楚的問題。
5. 新增運算量解析估計圖與 CSV，分開列出每元素 sigmoid、變動乘法、常數乘法、平方、比較、
   加法、shift-add 代理量，並將變動乘法投影到 640×640 Detect／Pose 每張影像。
6. qSiLU 同時列出目前 eager reference 與預期 fused coefficient-select 資料路徑，避免把所有
   分支都執行的軟體參考成本誤當成硬體融合後成本。
7. 更新 `reports/completed-activation-integrated-report.md`、根層 README、`reports/README.md`、
   `reports/figures/README.md` 與 `pyproject.toml` 的繪圖依賴與入口。
8. 新增 `tests/test_activation_report_figures.py`，鎖定四項指標、排除 ReLU、保留 SiLU 以及 qSiLU
   operation projection／fused contract。

## 數值口徑與結果

- 四項精度資料來源：`reports/full35-activation-results.json`。
- SiLU 是 accepted reference，因此四項 Δ 固定為 0。
- qSiLU-PQ 在 10-epoch recovery 的四項差異為：COCO `-0.002749`、Person `-0.000876`、
  BBAT box `-0.004285`、BBAT pose `-0.001731`；正式八項 gate 的 worst delta 為 `-0.008635`，通過
  `-0.015` 門檻。
- 主圖依使用者要求只顯示四項摘要；正式 gate 仍使用八項，兩者不可混為相同判定口徑。
- 函數誤差在 `[-8,8]`、400,001 個 float64 點計算。qSiLU-PQ 對 SiLU 的 MAE 為
  `0.0051297`、RMSE 為 `0.0060978`、最大絕對誤差為 `0.0108319`。
- deployment activation elements 由 raw profiling artifact 核對：Detect 每張影像 71,577,600，
  Pose 每張影像 72,652,800。這只是解析 operation proxy，不是 latency、功耗或硬體資源實測。

## 產物

- `reports/figures/activation-accuracy-overview.png`／`.svg`
- `reports/figures/activation-accuracy-summary.csv`
- `reports/figures/activation-central-error-enlarged.png`／`.svg`
- `reports/figures/activation-function-metrics.csv`
- `reports/figures/activation-operation-estimate.png`／`.svg`
- `reports/figures/activation-operation-estimate.csv`

## 驗證方式與結果

- `/home/uxin/yolo/.venv/bin/python -m pytest`：`63 passed`；4 個既有 legacy ONNX exporter
  deprecation warnings，無測試失敗。
- `ruff check .`：通過。
- `ruff format --check .`：71 份 Python 檔案格式正確。
- `scripts/validate_phase0.py`：六種 activation 的 curve／hardware proxy 驗證通過。
- `scripts/activation_training.py toy-dry-run`：輸出有限值，原模型未被修改。
- 三張 PNG 均以原始解析度人工檢視；文字、圖例、零基準、局部視窗與刻度可辨識，沒有裁切重疊。
- 產生器連續重跑後，9 個 `activation-*` PNG／SVG／CSV 的 SHA-256 全部一致。SVG 使用固定日期
  metadata，避免 Matplotlib 自動時間戳造成無內容差異。
- 沒有因本次工作啟動訓練程序或使用 GPU。

## 困難與解法

1. `apply_patch` 受環境 `bwrap: loopback: Failed RTM_NEWADDR` 阻擋，包含工作區及 `/tmp` 副本皆
   可重現。保留錯誤證據後，僅對唯一命中的 SVG metadata 片段做精確暫存替換；其餘檔案以 patch
   形式套用，並在寫回前後檢查差異。
2. queue 使用連字號名稱，registry 使用底線名稱。產生器採明確 normalization 後再查詢，避免默默
   漏掉候選。
3. 初版圖曾有字型缺字與註解重疊。改寫標籤、調整圖例與畫布配置，再逐張以原始解析度檢視。
4. SVG 初次重跑只有 `<dc:date>` 不同。儲存 SVG 時固定 `Date` metadata 後，第二次連續重跑的
   SHA-256 已完全一致。

## 未解事項或風險

- 運算量目前是基於原始碼與 element count 的靜態代理估計；尚未完成目標 FPGA／ASIC／GPU 的
  synthesis、latency、功耗、面積或記憶體流量實測。
- qSiLU 的係數選擇與 signed-digit shift-add 是硬體候選資料路徑，仍需 compiler IR、HLS／RTL
  bit-exact 與綜合報告確認實際成本。
- qSiLU seed-1 20-epoch finalist 曾被人工中止，seed 2 未執行；不能把 10-epoch recovery 擴張為
  多 seed 最終結論。
- BCSP mixed-policy 搜尋尚未完成；目前沒有可宣稱的 BCSP placement 結果。
- activation 與量化強耦合；PTQ、QAT、Q8／Q12／Q16 全碼枚舉及 saturation／rounding 消融應留給
  下一個 quantization 工作列共同評估。
