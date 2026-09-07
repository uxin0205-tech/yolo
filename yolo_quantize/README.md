# yolo_quantize

本次GitHub發行範圍見[PUBLICATION_0907.yaml](PUBLICATION_0907.yaml)，不含weights、runs或資料副本。完整盤點重建需要本機原始checkpoint；[權重分析附錄](deliverables/full-model-audit-2026-09-07/weight-evidence-tables.md)可由已公開CPU profile與CSV重建。原PUBLICATION_MANIFEST.yaml是0829歷史紀錄。

詳細補充：[SD4／三元分布與已測結果](docs/reports/2026-09-07-weight-distribution-detailed-analysis.md)／[完整數據附錄](deliverables/full-model-audit-2026-09-07/weight-evidence-tables.md)。僅整理既有證據，未新增GPU實驗。

> 2026-09-07最新盤點：[全模型量化報告與四天計畫](docs/reports/2026-09-07-full-model-audit-four-day-plan.md)／[數據與圖表](deliverables/full-model-audit-2026-09-07/README.md)。V36已完成3 epochs，148部署權重路徑均fake-quant；epoch1/2搜尋雙門檻通過。完整整數部署及十區独立accuracy sensitivity仍待完成。後續四天DAG尚未排入live queue。下列2026-09-05及更早進度保留為歷史。

本子專案研究Full35 Detect＋Pose聯合模型的activation-aware weight quantization。現行硬門檻是：activation替換與weight量化合併後，八項`mAP50`相對accepted Full35的最差絕對下降不得超過`0.015`，且八項`mAP50–95`下降不得超過`0.04`；兩族都過才是green。V19 `poly_shift+A8+all-W8`已鎖定Epoch 5，V29逐區PTQ已完成且短QAT可續跑封存；現行GPU主線轉為qSiLU自己的all-W8 parent與逐區量化。formal與最終優化仍只給鎖定finalists。

## 文件入口

- [現行實作計畫](IMPLEMENTATION_PLAN.md)：模型、資料、activation、weight、batch、門檻與逐步gate。
- [新增整合roadmap](configs/experiments/full35-integrated-quantization-roadmap-v1.yaml)／[Paper-TWN逐區規劃](docs/reports/2026-09-04-integrated-quantization-and-paper-twn-plan.md)：不改舊結果，將目前V19、W8/mixed、SD4/LS-SD4、ternary、activation coupling與formal串成backbone→neck→head流程。
- [V29七組來源與qSiLU接手設計](docs/reports/2026-09-05-v29-selection-and-qsilu-handoff-plan.md)：說明七組短QAT的PTQ/re-gate依據、第一組負結果、qSiLU Q0–Q7及regional Hardswish條件旁支。
- [V19／V29逐區量化執行報告](docs/reports/2026-09-04-v19-v29-progressive-quantization-execution.md)：整合歷史activation與bit結果、V19鎖定parent、148×5 CPU profile、V29 PTQ、recovery gate修正、7-job QAT queue與後續停止線。
- [V19 sham checkpoint匯出修復紀錄](docs/worklogs/2026-09-04-v19-sham-checkpoint-export-recovery.md)：epoch 0 gate、fused-head contract根因、red→green測試與乾淨重啟。
- [poly_shift PTQ、特殊格式與QAT入口總報告](docs/reports/2026-09-03-poly-shift-ptq-special-format-and-qat-entry.md)：W8–W4、mixed-bit、SD4／ternary、整體雙指標與剩餘工作。
- [V5 mAP50與bit邊界報告](docs/reports/2026-09-03-v5-map50-bit-boundary-search.md)：`-0.015`總門檻、W8 re-gate、W7–W4完整結果與後續路由。
- [現行v5 machine-readable契約](configs/experiments/full35-quantization-plan-v5.yaml)／[bit search結果](artifacts/reports/v5-qsilu-backbone-early-bit-search-v1.json)／[W8 mAP50 re-gate](artifacts/reports/v5-qsilu-backbone-early-w8-map50-regate-v1.json)。
- [凍結V4 W8搜尋驗證](docs/reports/2026-09-03-v4-qsilu-backbone-early-w8-search-validation.md)：保留原mAP50–95契約、三角色raw evidence與hash。
- [V4 W8 machine-readable結果](artifacts/reports/v4-qsilu-backbone-early-w8-search-v1.json)／[reviewed search plan](configs/experiments/v4-qsilu-backbone-early-w8-search-v1.yaml)。
- [V4 W8 delivery manifest](artifacts/manifests/v4-qsilu-backbone-early-w8-search-delivery-v1.yaml)：釘住計畫、checkpoint、資料View、程式與三角色metric hashes。
- [CPU P0詳細報告](docs/reports/2026-09-01-cpu-p0-integer-exact-routing.md)：integer boundary、三parent exact矩陣、routing v3結果與GPU停止線。
- [CPU P0 delivery](artifacts/manifests/exact-w4-sd4-cpu-delivery-v1.yaml)／[integer boundary manifest](artifacts/manifests/full35-integer-boundary-contract-qsilu-pq-a8-v1.json)／[Fixed SD4 routing v3](artifacts/manifests/fixed-sd4-routing-candidates-v3.yaml)。
- [架構與實驗稽核報告](docs/reports/2026-09-01-architecture-experiment-audit.md)：已修問題、現行矩陣、GPU前blocker與研究方向。
- [量化文獻與研究新意稽核](docs/research/2026-09-01-quantization-literature-and-innovation-audit.md)：33組第一手來源、強baseline與可否證研究假說。
- [凍結v4 machine-readable契約](configs/experiments/full35-quantization-plan-v4.yaml)／[V4+歷史矩陣](configs/experiments/v4-plus-prepared-plan-v4.yaml)／[歷史universe v2](configs/experiments/weight-region-sensitivity-plan-v2.yaml)。
- [Hardswish policy修訂報告](docs/reports/2026-08-31-hardswish-policy-revision.md)：uniform／regional證據、active矩陣、CPU結果與停止線。
- [Q3 parent證據查核](docs/reports/2026-08-31-q3-activation-parent-evidence.md)：指定GitHub報告、checkpoint selector與不可跨用邊界。
- [V1–V3凍結交付manifest](artifacts/manifests/v1-v3-cpu-delivery-v2.yaml)：保存2026-08-31的active parent、來源hash、CPU產物與future boundary；不是v4執行manifest。
- [歷史Fixed SD4 routing v2](artifacts/manifests/fixed-sd4-routing-candidates-v2.yaml)：qSiLU／Hardswish／poly_shift在十點grid下的34層候選；保留作ablation，不覆寫成exact結果。
- [歷史V0–V3 CPU報告](docs/reports/2026-08-30-v1-v3-cpu-implementation.md)／[delivery v1](artifacts/manifests/v1-v3-cpu-delivery-v1.yaml)：保留含`poly_quality`的原始證據，不再是active matrix。
- [無訓練activation smoke報告](docs/reports/2026-08-29-activation-output-smoke-v2.md)：30格耦合矩陣、限制與後續候選。
- [Activation預選老師版報告](docs/reports/2026-08-29-activation-preselection-report.md)：指標白話解釋、A3至A8選擇與SD4納入方式。
- [老師版PNG](deliverables/activation-preselection-teacher-v1.png)／[PDF](deliverables/activation-preselection-teacher-v1.pdf)／[SVG](deliverables/activation-preselection-teacher-v1.svg)。
- [Weight region重新規劃](docs/reports/2026-08-29-weight-region-replan.md)：A3至A8判定、真實region盤點、INT8／INT4矩陣、指標與QAT方法。
- [Weight region歷史v1計畫](configs/experiments/weight-region-sensitivity-plan-v1.yaml)：保留舊結果血緣，不是現行執行入口。
- [backbone_early PTQ結果報告](docs/reports/2026-08-29-weight-ptq-backbone-early-v1.md)：W8／W4六格scorecard、容量、限制與下一步。
- [backbone_early結果manifest](configs/experiments/weight-ptq-backbone-early-result-v1.yaml)／[完整JSON](artifacts/reports/weight-ptq-backbone-early-v1.json)／[摘要CSV](artifacts/reports/weight-ptq-backbone-early-v1-summary.csv)。
- [工作紀錄](docs/worklogs/README.md)。
- [qSiLU＋LSQ+ provisional policy](configs/activation/qsilu-pq-lsq-plus.yaml)。
- [smoke實驗契約](configs/experiments/activation-smoke-v2.yaml)。
- [舊研究規格](quantize_spec.md)：LS-SD4／LSQ+、FREQ、O2M→O2O KD與ternary方法參考；和Full35真實契約衝突時不作執行基線。
- 三元權重變形器神經網路加速電路之設計與晶片實現.pdf：Paper-TWN、LSQ+與Progressive Quantization參考來源。

## 固定來源

- 模型：/home/uxin/yolo/yolo_combine/final/full35/，只透過adapter讀取，不直接修改final bundle。
- COCO Detect：/home/uxin/yolo/coco2017.yaml。
- BBAT5 Pose：/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml。
- BBAT5 screening：/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose-search.yaml；子專案runtime View只含指回canonical entry的symlink與local cache。
- Activation evidence：/home/uxin/yolo/yolo_activation/。本專案沒有也不依賴yolo_activation2。

所有新BBAT5工作必須沿用不可變bbat5-v1 assignment。本次calibration只讀canonical train exemplar，search validation使用既有固定search-val；沒有重切、改label或建立30% view，也沒有碰formal val。

## 目前狀態

2026-09-05：

- V29已在第二組matched-sham完成epoch 1 checkpoint後依使用者要求安全暫停；第一組paired QAT完成但mAP50-95未通過，所有原artifact與resume checkpoint由hash封存清單保留。
- V30 qSiLU完整Queue已由hash-pinned archive handoff啟動，自全十區W8 PTQ開始，之後依gate進paired QAT、148×8 CPU profile、逐區PTQ及matched短QAT；不套用poly_shift weight winner。
- Hardswish只登錄Q3支持的regional條件旁支；先測neck-attention一格，不採uniform Hardswish或未量測的多區組合。

2026-09-04：

- V19 paired QAT已完成：patience=5合法early stop，Epoch 5通過雙門檻並鎖定為`poly_shift+A8+all-W8` parent。
- 148 deployment paths × exact W4／Fixed-SD4／Paper-TWN v2／TWN-v3 filter-wise／exact ternary共740格CPU profile已完成；只作排序，不取代mAP。
- V29完成十階段backbone→neck→head PTQ；最後累積鎖定MASF exact ternary、neck-attention Fixed-SD4、Pose tower exact W4，worst total mAP50／mAP50–95為`-0.013842／-0.013098`。
- 發現YAML recovery負delta在parser被再次取負號；已用red→green測試修正。原V29 metrics、state與checkpoint未覆寫，GPU validation未重跑。
- corrected re-gate得到12個recover，依6 main＋2 sentinel上限選出6＋1共7個paired短QAT；第一個matched sham已啟動。
- 短QAT採AdamW、最多15 epochs、patience=5、3 FP＋6 ramp＋6 full、Detect logical128／micro16、Pose16、不加noise；每600秒只讀外部事件，錯誤最多重試一次並優先從`last.pt`續跑。
- formal、multi-seed、20／60 epoch延伸、MuSGD對照、export與最終硬體優化仍未啟動。

2026-09-03：

- `poly_shift+A8` W8–W4逐區敏感度、mixed-bit、Fixed-SD4、Paper-TWN與七格耦合PTQ均已完成；九區W8通過，`backbone_attention_safe`暫留FP32。
- accuracy PTQ為九區W8＋單一SD4 Detect predictor：最差mAP50／mAP50–95 `-0.011415／-0.017322`、總weight約`3.551×`；balanced SD4 Detect head約`3.611×`但mAP50–95只剩`0.000212`餘裕。
- 新判定同時保留八項mAP50與八項mAP50-95；前者總下降上限`0.015`，後者伴隨上限`0.04`，兩者皆過才列green。
- QAT的BN-only fold、weight fake quant、scale-only epoch、quantizer獨立LR、mAP雙門檻、deployment materialization與matched-sham前置gate已完成；GPU smoke以logical128／physical16通過，V19 matched sham執行中。
- mAP50門檻已固定為八項total delta都`>= -0.015`，且total包含activation替換；平均值不能蓋過最差任務。
- qSiLU＋A8／`backbone_early` W8重新套用mAP50 gate後仍為`green`：最差total是COCO box `-0.011408`，最差W8 incremental是COCO person `-0.000781`。
- W7–W4完整search validation已完成：W7 `recover`（`-0.020950`）；W6 `reject`（`-0.044686`）；W5 `reject`（`-0.286769`）；W4 grid `reject`（`-0.971333`）；W4 exact `reject`（`-0.854778`）。目前Pareto三角色均為W8。
- 其餘九區isolated W8也已完成：8格green，只有`backbone_attention_safe` recover（COCO box `-0.017922`）。isolated accuracy是MASF，balanced／hardware是neck；這些不能相加外推。
- accepted／matched／candidate皆包含COCO person與BBAT ball／bat box／pose共八項mAP50；使用完整5,000張COCO val與既有600張BBAT5 search-val，不是formal val。沒有訓練。

- V0已保留`5090 Profile 0829`公開activation證據與本機歷史PTQ；公開報告和publication commit逐位元相同。
- activation仍無正式winner；active uniform parents為qSiLU＋A8、Hardswish＋A8、poly_shift＋A8。`poly_quality`只保留歷史證據，不再進future execution。
- corrected catalog為251個Conv／Linear：148 deployment、99 training-only、4 Binary Q/K protected；舊151-layer結果只作歷史診斷。
- 三個active uniform parent皆完成179→0 BN fold與one-to-one CPU forward parity；另有三個qSiLU-root regional Hardswish policy通過相同parity。
- active三parent各2,960筆、合計8,880筆舊static measurements；另保留2,960筆歷史`poly_quality`。這些數值來自十點`mse_grid_v1`，可作歷史proxy，不能再稱全域MSE最佳或直接promotion。
- exact scaled-codebook scale solver已用獨立assignment搜尋驗證；三個active parent各完成1,184筆、共3,552筆uniform W4／Fixed SD4 grid-vs-exact雙View結果。exact在888個uniform與888個SD4 cells均不劣於grid。
- routing v3要求三parent × master/deployment六個比較全數嚴格勝出，得到36個static候選；比grid v2多2層、未移除舊34層。這不是mAP winner，`execution_authorized=false`。
- qSiLU BN-folded graph的integer boundary已建立CPU reference：124個activation quantizers、21個reviewed core Concats、20個reviewed core Adds，148/148 W8/A8 MAC bounds通過INT32。LSQ+ offset的bias／padding lowering與實際saturation仍待GPU calibration。
- SD4已補齊15個logical values／16個nibble codes、雙zero、canonical padding與pack round-trip；這是encoding正確性，不是硬體速度結果。
- Paper-TWN在任何parent／view都沒有層的靜態MSE勝W4；保留作static control與未來progressive QAT，不列PTQ主線。
- 固定diagnostic manifest包含每task 32 calibration／64 probe、COCO person及BBAT group-safe coverage；v2 plan另釘住manifest SHA-256，active runner再驗證manifest內容與全部影像／label SHA-256，最後使用固定第一張probe作結構診斷。BBAT assignment未改。
- `backbone_early`整區低於W8的PTQ已被實證排除；這不排除個別低敏感layer使用W7/W6/W5/W4或SD4，也不能外推到其他region。
- 現正依序執行matched sham、全W8 QAT、mixed-bit QAT與Fixed-vs-learned SD4對照。formal、multi-seed、export與最終硬體優化尚未啟動。

## 重現命令

Intake screening與finalization：

    PYTHONPATH=src /home/uxin/yolo/.venv/bin/python scripts/activation_intake.py --gate screening
    PYTHONPATH=src /home/uxin/yolo/.venv/bin/python scripts/activation_intake.py --gate finalization

CPU回歸：

    CUDA_VISIBLE_DEVICES=-1 /home/uxin/yolo/.venv/bin/python -m pytest -q
    /home/uxin/yolo/.venv/bin/ruff check src tests scripts
    /home/uxin/yolo/.venv/bin/ruff format --check src tests scripts

歷史V1／V2產物重建入口（保留原`mse_grid_v1`語意，不會產生新版exact routing）：

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src /home/uxin/yolo/.venv/bin/python scripts/prepare_v1_v3.py

重建單一parent的歷史V1／V2 main profile；完整CPU分析約需10–12分鐘：

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src /home/uxin/yolo/.venv/bin/python scripts/prepare_v1_v3.py --parent qsilu_pq --profile main --view-output artifacts/manifests/weight-views-qsilu-pq-a8-v2.json --analysis-output artifacts/reports/weight-format-analysis-qsilu-pq-a8-main-v2.json

重建Q3 regional policy的dual-view parity，不重複相同checkpoint的weight-only分析：

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src /home/uxin/yolo/.venv/bin/python scripts/prepare_v1_v3.py --parent qsilu_pq --activation-region neck_attention=hardswish --view-only --view-output artifacts/manifests/weight-views-qsilu-pq-hardswish-neck-attention-a8-v3.json

`poly_quality`已從`--parent` choices排除；若只為重建凍結歷史證據，必須明寫`--historical-parent poly_quality`。

重建本輪exact CPU evidence時必須使用新的output root，封存artifact會拒絕覆寫：

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src /home/uxin/yolo/.venv/bin/python scripts/prepare_exact_weight_cpu.py --parent qsilu_pq --output-root /tmp/yolo-quantize-exact-rebuild
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src /home/uxin/yolo/.venv/bin/python scripts/prepare_exact_weight_cpu.py --parent hardswish --output-root /tmp/yolo-quantize-exact-rebuild
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src /home/uxin/yolo/.venv/bin/python scripts/prepare_exact_weight_cpu.py --parent poly_shift --output-root /tmp/yolo-quantize-exact-rebuild
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src /home/uxin/yolo/.venv/bin/python scripts/prepare_exact_weight_cpu.py --routing-only --output-root /tmp/yolo-quantize-exact-rebuild
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src /home/uxin/yolo/.venv/bin/python scripts/prepare_exact_weight_cpu.py --delivery-only --output-root /tmp/yolo-quantize-exact-rebuild

現行v2 matrix只讀展開（不載入model、不碰GPU）：

    PYTHONPATH=src /home/uxin/yolo/.venv/bin/python -m yolo_quantize.weight_sensitivity --regions backbone_early --bits 8,4 --scale-method mse_grid_v1 --list-only

2026-09-03單格search契約只讀檢查（不碰GPU）：

    PYTHONPATH=src /home/uxin/yolo/.venv/bin/python -m yolo_quantize.search_validation --plan configs/experiments/v4-qsilu-backbone-early-w8-search-v1.yaml --list-only

已完成v5 bit race的只讀／續跑入口；完成狀態不會重跑validation：

    PYTHONPATH=src /home/uxin/yolo/.venv/bin/python -m yolo_quantize.search_race --plan configs/experiments/v5-qsilu-backbone-early-bit-search-v1.yaml --list-only
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/uxin/yolo/.venv/bin/python -m yolo_quantize.search_race --plan configs/experiments/v5-qsilu-backbone-early-bit-search-v1.yaml --output artifacts/reports/v5-qsilu-backbone-early-bit-search-v1.json --device 0 --resume --execute-reviewed-plan

已完成產物的契約／hash續跑檢查；因三角色都完成，`--resume`不會重跑validation：

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/uxin/yolo/.venv/bin/python -m yolo_quantize.search_validation --plan configs/experiments/v4-qsilu-backbone-early-w8-search-v1.yaml --output artifacts/reports/v4-qsilu-backbone-early-w8-search-v1.json --device 0 --resume --execute-reviewed-plan

本次完整回歸結果見現行[工作紀錄](docs/worklogs/README.md)與[CPU P0 delivery](artifacts/manifests/exact-w4-sd4-cpu-delivery-v1.yaml)。

Activation、weight與search runner都支援原子寫入、resume與既有輸出防覆寫。使用者已授權依v5逐階段完成，但每階段仍以明確cell清單與停止線執行；不會從PTQ自動跳進formal validation或正式訓練。
