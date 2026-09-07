# 2026-08-29：重新整合backbone到head的INT8／INT4 weight規劃

## 任務與範圍

依使用者要求回到規劃階段，停止新增SiLU實驗，整理A3至A8結果，並將weight量化改成從backbone逐步到Detect／Pose heads的INT8／INT4敏感度、完整指標與可選擇停點。本次只做唯讀graph盤點及文件／配置更新，沒有啟動weight實驗、validation、QAT或正式訓練。

## 變更內容與原因

1. 將future activation shortlist縮成qSiLU＋A8、poly_quality＋A8、poly_shift＋A8；SiLU只保留歷史基準，Hardswish因A3至A8 proxy皆被支配而不晉級。
2. 將A3／A4判為主線淘汰、A5不晉級、A6研究用、A7探索、A8作weight分析第一主線。
3. 唯讀載入accepted Full35，盤點251個Conv／Linear modules與26,451,392個weights，建立151-module／22,572,608-weight部署候選catalog。
4. 將training-only 96 modules與Binary Q/K 4個projection modules從一般weight quantization排除。
5. 將head拆為tower與final predictor；attention拆為安全子集與受保護Q／K，MASF維持拓樸但三個內部Conv獨立測試。
6. 建立W0分布、W1隔離probe、W2累積完整validation、W3 QAT、W4 SD4／ternary、W5 lower A-bit工作樹。
7. 定義每格固定scorecard、W8 0.01 incremental gate、0.04 total gate、0.06 recovery band與每region停下來讓使用者選的流程。
8. 以codebase-design的deep module原則，將未來interface集中為WeightSensitivityStudy，內含WeightRegionCatalog、WeightQuantizationAdapter與SensitivityEvaluator seams。
9. 新增machine-readable計畫及中文報告，並同步README、現行實作計畫與工作紀錄索引。

## 驗證方式與結果

- A3至A8 Pareto重算排除SiLU：Hardswish在每個bit均被至少一個候選支配。
- A8非SiLU結果：poly_quality worst raw NRMSE 0.081644、minimum pair overlap 0.6933；poly_shift 0.098934／0.6867；qSiLU 0.101851／0.6133。
- 真實Full35 module盤點：251 modules、26,451,392 weights。
- 部署候選與排除數可完全對回：151 candidate＋96 training-only＋4 Binary Q/K protected等於251；22,572,608＋3,747,712＋131,072等於26,451,392。
- backbone_deep＋neck占部署候選72.075%；MASF＋兩個predictor只占0.623%。
- 三個completed short-recovery checkpoint都存在、大小皆106,825,541 bytes，SHA-256已寫入machine-readable plan。
- weight plan YAML解析、151／96／4 module算術、22,572,608／3,747,712／131,072 weight算術、W1 60格契約及training hold全部通過。
- 三個父checkpoint檔案大小與SHA-256逐一比對通過。
- 完整pytest：34 passed in 9.05s。
- ruff check通過；ruff format --check回報16 files already formatted；所有本地Markdown連結通過。
- 沒有修改Full35、yolo_activation、dataset、checkpoint或Git外部狀態。

## 困難與解法

- 困難：若直接做3 activation乘10 regions乘多種weight format與所有累積組合，矩陣會快速爆炸。
- 解法：先做60格便宜隔離probe，再以accuracy／hardware／balanced三角色的cumulative beam逐region保留非劣父policy。
- 困難：activation與weight強耦合，但weight敏感度仍需能定位損失來源。
- 解法：W0至W2先固定A8並一次只改一個region；所有晉級結果仍保存完整activation＋weight policy並重跑三任務metrics。
- 困難：Binary Q/K、attention、MASF與training-only head不能用一般Conv規則一體處理。
- 解法：建立fail-closed region catalog；Q／K與PWL保護，V／projection／FFN另列安全子集，MASF只量化內部Conv且不改拓樸。
- 困難：內建apply_patch仍因bwrap loopback RTM_NEWADDR無法讀檔。
- 解法：先保留apply_patch失敗證據，再經核准使用受控精確區段替換，只修改yolo_quantize文件與配置。

## 未解事項或風險

- 尚未實作weight quantizer、region adapter、BN-folded analyzer或validator runner。
- A3至A8是小樣本proxy，不是完整mAP；A8只是第一主線，不是正式winner。
- W4是否需要QAT必須由W1／W2決定；目前只能依經驗預期，不能先宣稱。
- 固定30% diagnostic view尚未建立；開始QAT前仍需遵守BBAT5 immutable assignment與group-safe manifest。
- SD4與ternary只完成路由規則，尚無真實weight distribution結果。
- 沒有板上latency／power／resource，成本只能先報metadata-aware proxy。
- 正式訓練保持未啟動。
