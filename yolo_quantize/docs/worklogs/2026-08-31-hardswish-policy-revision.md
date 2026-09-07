# 2026-08-31：納入Hardswish、排除poly_quality並整合Q3區域證據

## 目標與授權邊界

依使用者要求，重新檢視 `yolo_activation/reports/full35-q3-cpu-final-report.md` 後，將Hardswish納入後續量化規畫，並把 `poly_quality` 排除於2026-08-31之後的active matrix。使用者寫的 `poly_quantity` 對應上游正式registry識別字 `poly_quality`。

本輪只做研究查核、CPU程式實作、dual-view parity、weight reconstruction分析與未執行矩陣準備。未使用GPU，未跑新PTQ calibration／probe、mAP validation、QAT、正式訓練、export或硬體benchmark。

## 變更內容與原因

### Q3證據查核

- 查核GitHub `main` commit `30bd50e2c9e16feb7adb989cb042e1a13091c067` 的指定報告；遠端Git blob `cf21ddd028f527f0d6a00bdd28531b3fc9014a7c`、13,794 bytes，與本機同名報告逐位元一致。
- Q3是22／22個完整CPU zero-shot full-validation cells，沒有訓練或量化。其Hardswish結果全部以qSiLU checkpoint `767918…6190e` 為權重根，每次只換一個activation region。
- 可直接取用的是placement順序：`neck_attention`、`masf`、`backbone_attention`；不能把Q3寫成uniform Hardswish勝出，也不能把Bit-True backend誤稱為本專案A8／W8結果。
- 完整來源與不可跨用邊界記錄於 `docs/reports/2026-08-31-q3-activation-parent-evidence.md`。

### Active／historical政策修訂

- Active uniform A8 parents改為 `qsilu_pq`、`hardswish`、`poly_shift`。
- `poly_quality` 從future matrix、active CLI、racing與QAT promotion移除，但既有V0–V2 JSON、manifest、報告、圖表及hash全部保留為historical evidence，不刪除、不覆寫。
- Uniform Hardswish使用本機 `best_joint.pt`，SHA-256 `79e0e4…7731`。該檔metadata指向zero-based epoch 8，八項最差delta `-0.014121`；同一run的final epoch 9最差delta `-0.016884`。兩個selector事實都凍結於checkpoint intake，並把uniform Hardswish標為experimental而非正式winner。
- Hardswish與poly_shift recovery使用FP32 CIoU修正，qSiLU recovery早於該修正；後續若進QAT，必須用新的matched sham凍結loss precision、optimizer與資料順序。

### Regional activation policy seam

- `Full35ActivationPolicy` 新增 `region_assignments`，可表達「預設qSiLU＋指定region Hardswish＋LSQ+ A8」。
- Policy ID會排序region assignments，確保可重建；重複、空白、不支援activation或未知region一律fail closed。
- adapter沿用上游190-site manifest建policy，並保存activation counts；124個deployment-eligible activation sites與66個training-only paths的既有邊界不變。
- CPU建立三個Q3單區view manifest：`neck_attention` 為1 Hardswish／189 qSiLU；`masf` 與 `backbone_attention` 各為3／187。
- 三個regional policy共用完全相同的qSiLU checkpoint權重，因此只做graph／dual-view parity，不重複產生沒有資訊增益的3×2,960筆weight-only分析。

### CLI、矩陣與SD4路由

- `prepare_v1_v3.py --parent` 現在只接受三個active uniform parents；若要重建歷史，必須明確使用 `--historical-parent poly_quality`。
- 新增可重複的 `--activation-region REGION=ACTIVATION` 與 `--view-only`，讓regional policy只建立必要的parity證據。
- V4 uniform仍為3 parents × W8／W7／W6／W5／W4 = 15個量化cells，另有3個FP32-weight matched controls。
- V4H獨立做3個Q3單區policy × `[matched control, W8]` = 6 jobs；只有 `neck_attention` 與 `masf` 單區都通過才新增2個組合jobs。
- V5 uniform為3 parents × 10 weight regions × 5 bits = 150個量化cells。Activation region與weight region不視為同一集合。
- 以新active parent重新計算Fixed SD4 routing：各parent deployment 37層、master 35層勝matched W4，保守交集仍為34層。候選集合雖與舊v1相同，來源hash不同，因此新增v2 manifest而不改舊檔。
- 所有V4+設定明記 `execution_authorized: false`；沒有自動啟動GPU或QAT。

## 實驗與分析結果

### Uniform Hardswish CPU證據

- Dual view完成179→0 BN fold、148條deployment path及inference contract parity。
- Forward NRMSE `5.759842e-6`，max error／reference peak `3.202221e-5`，通過既定parity gate。
- V2 main profile完成2,960筆：uniform 1,480、Fixed SD4 1,184、Paper-TWN 296；nonfinite為0、`gpu_used=false`。
- Hardswish deployment NRMSE：W8 `0.006240`、W7 `0.012459`、W6 `0.024462`、W5 `0.049417`、W4 `0.095724`、Fixed SD4 `0.160400`、Paper-TWN `0.677474`。
- 這些是weight reconstruction proxy，不是mAP，也不是硬體latency／power。

### Regional dual-view parity

| Policy | Forward NRMSE | Max error／reference peak | 結果 |
|---|---:|---:|---|
| qSiLU＋Hardswish `neck_attention` | `3.395364e-6` | `7.118448e-6` | pass |
| qSiLU＋Hardswish `masf` | `3.423228e-6` | `6.440059e-6` | pass |
| qSiLU＋Hardswish `backbone_attention` | `3.809068e-6` | `5.286480e-6` | pass |

以上只證明policy建構、BN fold與CPU forward契約可用；尚未證明A8 observer、W8 PTQ或完整mAP通過。

## 驗證方式與結果

### TDD與契約測試

- 先以舊active parent集合、缺少regional seam及CLI拒絕Hardswish重現5個失敗、5個通過的Red狀態。
- 完成active／historical邊界、regional policy與CLI後，目標測試先達10 passed，再加入artifact／矩陣稽核後達12 passed。
- 全量功能測試：`CUDA_VISIBLE_DEVICES=-1` 下 `62 passed in 57.37s`。
- `ruff check src tests scripts` 通過；第一次 `ruff format --check` 只指出新測試檔的排版差異，已用ruff格式化，最終重驗為34個檔案全部已格式化。
- 三份regional manifests、Hardswish view／analysis、SD4 v2的bytes與SHA-256均寫入 `v1-v3-cpu-delivery-v2.yaml`。
- 唯讀稽核成功解析17份YAML、16份JSON並核對9組bytes／SHA；Hardswish 2,960筆numeric值皆finite，V4 15格、V4H 6＋2條件格、V5 150格及 `execution_authorized=false` 全部通過。

## 遇到的困難與解法

### GitHub頁面擷取與來源一致性

- 困難：指定GitHub網頁的瀏覽擷取出現cache miss。
- 解法：改以GitHub Contents API、`git ls-remote` 與本機Git blob唯讀交叉驗證；遠端與本機Q3報告的blob、bytes一致，沒有用不明快取內容替代。

### Checkpoint selector不一致

- 困難：Uniform Hardswish的 `best_joint.pt` 是epoch 8且通過舊gate，但上游run-level結論用final epoch 9並失敗；只抄單一結果會把metric綁錯checkpoint。
- 解法：建立checkpoint-level intake，同時保存epoch 8與final epoch 9的值；量化parent明確綁epoch-8 SHA並標experimental。

### 設定排版錯誤

- 困難：新增v3 YAML時曾出現一處縮排錯誤。
- 解法：以YAML parser立即攔截並修正，再新增契約測試確認active parents、regional cells與 `execution_authorized=false`。

### CPU分析耗時

- 困難：Hardswish完整2,960筆MSE-grid／SD4分析約需20分鐘。
- 解法：只對新uniform checkpoint做一次完整分析；regional policies共用同一權重根，因此只建立view parity，避免重複計算與誤增證據筆數。

### Sandbox格式化

- 困難：第一次格式化命令因 `bwrap: loopback: Failed RTM_NEWADDR` 未啟動。
- 解法：依環境規範使用受核准的單檔ruff格式化，沒有用其他寫檔方式規避。

## 未解事項與風險

- 尚無Hardswish × A8 × W8–W4的量化後八項mAP；不能宣稱Hardswish或任何weight bit勝出。
- Q3不是A8／W8實驗；它只提供activation placement優先序。
- `neck_attention + masf` 多區組合尚未量測，單區delta不能相加。
- Uniform parent recovery的FP32 CIoU版本不完全一致，QAT必須重新做matched control／sham。
- Fixed SD4仍只有static routing proxy；LS-SD4、Channel-TWN、TTQ與A-SD4尚未訓練或驗證。
- 尚無target FPGA／ASIC bit-true kernel、latency、power或resource量測；`hardware-friendly` 仍是operator／編碼取向，不是實測加速。
- 根Git工作樹仍dirty且branch diverged；本輪未commit、push、reset或checkout。發布前需另做finish-work稽核。

## 主要產物

- `docs/reports/2026-08-31-hardswish-policy-revision.md`
- `docs/reports/2026-08-31-q3-activation-parent-evidence.md`
- `configs/experiments/full35-quantization-plan-v3.yaml`
- `configs/experiments/v4-plus-prepared-plan-v3.yaml`
- `artifacts/manifests/hardswish-uniform-parent-intake-v1.yaml`
- `artifacts/manifests/weight-views-hardswish-a8-v3.json`
- `artifacts/reports/weight-format-analysis-hardswish-a8-main-v3.json`
- `artifacts/manifests/fixed-sd4-routing-candidates-v2.yaml`
- `artifacts/manifests/v1-v3-cpu-delivery-v2.yaml`
