# 2026-08-29：實作PTQ並完成backbone_early bit sensitivity第一階段

## 任務與範圍

依使用者同意，先做 PTQ 與 bit sensitivity，不啟動 QAT 或正式訓練。固定 qSiLU＋A8、poly_quality＋A8、poly_shift＋A8三條完整activation parent，先完成 `backbone_early` 的W8／W4隔離敏感度；使用者之後要求GPU讓其他人優先使用，因此W6、完整validation與下一個region均暫停。

## 變更內容與原因

1. 新增`Full35WeightRegionCatalog`，fail-closed分類所有Conv／Linear weights為10個部署region、training-only或Binary Q/K protected。
2. 新增`UniformWeightSpec`與`WeightQuantizationAdapter`，支援2至16 bit signed uniform、per-output-channel max或MSE scale、nearest-even、數值scorecard與context結束後逐位元還原FP32 master weights。
3. 擴充`Full35ActivationAdapter.build`，允許載入明確activation parent checkpoint；explicit checkpoint必須提供並通過SHA-256，載入順序是在activation function replacement後、A8 observer wrapping前。
4. 新增`WeightSensitivityStudy`與CLI，從frozen plan展開cell，禁止新SiLU run，支援list-only、指定activation／region／bits、原子JSON、resume與既有輸出防覆寫。
5. runner使用matched activation A8＋FP32 weights作reference，每個cell只暫時量化一個region，輸出weight numeric、逐layer numeric、Detect／Pose raw、TopK、anchor Jaccard、timing與GPU memory。
6. 新增CLI entry point `yolo-quantize-weight-sensitivity`。
7. 先跑poly_quality W8 tracer，再完成三條parent乘W8／W4的6格`backbone_early`矩陣。
8. 產出完整JSON、CSV scorecard、result manifest與中文結果報告。
9. 同步README與現行實作計畫，將已完成的第一個region、W6 pending、完整validation未做及GPU暫停寫成明確停點。

## TDD執行紀錄

本次依既有計畫使用TDD skill，公開seam沿用使用者先前已同意的`WeightRegionCatalog.inspect`、`WeightQuantizationAdapter.quantized`與`WeightSensitivityStudy`／CLI。

1. 紅燈：套件沒有`Full35WeightRegionCatalog`；綠燈：代表性10區域與training-only／Binary Q/K分類通過。
2. 紅燈：catalog沒有真實Full35 summary；綠燈：251個Conv／Linear精確分為151部署、96 training-only、4 Binary Q/K protected，10區域modules／weights皆和frozen plan一致。
3. 紅燈：沒有W4 adapter；綠燈：已知數值nearest-even結果正確，只修改指定region，protected不變，離開context後FP32逐位元還原。
4. 紅燈：MSE scale尚未實作；綠燈：outlier-heavy已知分布的W4 MSE低於max scale且仍完整還原。
5. 紅燈：Full35 adapter不接受explicit parent；綠燈：poly_quality checkpoint path、SHA-256與EMA state source正確。
6. 紅燈：沒有reviewed matrix與CLI；綠燈：cell順序、parent SHA、禁止SiLU與list-only均通過。
7. 整合紅燈：CLI僅能list；綠燈：單格真實GPU tracer及6格矩陣成功完成。

## 驗證方式與結果

- 真實graph catalog：251 modules、26,451,392 weights；151部署候選、22,572,608 weights；96 training-only；4 Binary Q/K protected。
- explicit三個parent checkpoint的本機SHA-256均和frozen plan一致。
- tracer `poly_quality＋A8／backbone_early／W8`成功：weight NRMSE 0.011978，但Pose raw NRMSE 0.100267、TopK overlap 0.55，證明不能只看weight重建誤差。
- 6格矩陣：6 passed、0 failed、formal training false。
- W8：weight NRMSE約0.01198，worst raw NRMSE 0.10027至0.10661，minimum TopK 0.48至0.60。
- W4：weight NRMSE約0.16162，worst raw NRMSE 0.25789至0.26388，minimum TopK僅0.0167至0.03，判定`backbone_early`純PTQ collapse。
- BBAT5仍使用canonical `bbat5-v1` train／val exemplar，未改assignment、split、影像或標註。
- 完整CPU回歸：`43 passed in 32.39s`。
- Ruff靜態檢查：`All checks passed!`；格式檢查：`21 files already formatted`。
- list-only按frozen plan正確展開3條parent × W8／W4共6格，plan SHA-256不變。
- result YAML可safe-load；source plan與JSON／CSV／tracer hash全數符合manifest；JSON為6 passed、0 failed，CSV為6列。
- README、結果報告與工作紀錄索引的相對Markdown連結均存在。
- 無QAT、無正式訓練、無完整mAP validation。

## 困難與解法

- 困難：內建`apply_patch`多次受`bwrap: loopback: Failed RTM_NEWADDR`間歇性阻擋。
- 解法：能使用時仍優先`apply_patch`；失敗時只在可寫子專案內使用帶唯一marker／存在性檢查的精確替換，寫入後以測試、ruff與hash重新驗證。
- 困難：初次檢查時GPU由其他MambaPose工作占用約23 GiB。
- 解法：先完成CPU inventory與TDD；確認GPU完全空閒後才執行tracer與6格矩陣，未終止或干擾他人程序。
- 困難：W8的weight NRMSE很小，但forward proxy放大約到0.10。
- 解法：scorecard同時保留weight、raw output、TopK與anchor metrics，不使用單一weight MSE選擇policy。
- 困難：使用者在準備W6時要求GPU讓其他人先用。
- 解法：未啟動W6；立即確認本次程序已結束，目前GPU只剩其他人的PID 3110820，後續只做CPU與文件整理。

## 未解事項或風險

- W8尚未跑COCO box、BBAT box、BBAT pose完整mAP，不能宣稱通過0.01 incremental gate。
- W6 conditional rescue尚未執行；它是使用者允許非2冪bit width後最重要的下一個比較點。
- W4只在`backbone_early`純PTQ被淘汰；尚未測第一層留W8、其他early layers W4、layer-wise敏感度或QAT recovery。
- 尚未進`backbone_deep`、neck、MASF、attention safe subsets或Detect／Pose heads。
- Fixed SD4、LS-SD4與ternary仍等待uniform bit baseline與分布路由。
- GPU工作保持暫停，需使用者後續明確恢復才會繼續。
