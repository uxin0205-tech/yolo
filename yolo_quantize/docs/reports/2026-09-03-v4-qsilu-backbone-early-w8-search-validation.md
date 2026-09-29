# V4 qSiLU＋A8／backbone early W8 搜尋驗證報告

日期：2026-09-03。

## 結論

第一個新版 GPU bridge 已完成，判定為 **green**。本輪沒有訓練，也沒有使用 BBAT5 formal validation；它只在同一個 search-validation 契約下比較三個角色：

1. accepted：原始 Full35 SiLU、activation-output quantization 關閉、FP32 weight。
2. matched：qSiLU recovery checkpoint、LSQ+ A8、FP32 weight。
3. candidate：與 matched 完全相同，只把 `backbone_early` 的 21 個 Conv／Linear、1,218,240 個 weight 做 per-output-channel W8 PTQ。

candidate 相對 accepted 的八項最差總下降為 COCO box `-0.013504`，通過使用者同意的 `-0.04` 門檻；candidate 相對 matched 的 W8 最差額外下降為 BBAT ball box `-0.007147`，通過 `-0.01` incremental 門檻。因此這一格可進入後續 search pool，但 **green 不等於正式 winner、硬體速度證明或可直接展開所有低位元**。

完整 machine-readable 結果是 [`v4-qsilu-backbone-early-w8-search-v1.json`](../../artifacts/reports/v4-qsilu-backbone-early-w8-search-v1.json)，SHA-256 `738f5103656deebcb248ad6e7d0770caa662fa191c90882f45f8f78e47bc6ad2`；計畫、checkpoint、資料、程式與三角色metrics hash另由[`delivery manifest`](../../artifacts/manifests/v4-qsilu-backbone-early-w8-search-delivery-v1.yaml)釘住。

## 為什麼需要三個角色

activation 與 weight 量化是耦合的，不能把各自單獨最好的結果直接相加。這次刻意拆成兩種 delta：

- `matched - accepted`：完整 qSiLU parent policy 的變化。它同時包含 qSiLU function、qSiLU recovery checkpoint 與校正後 LSQ+ A8，不能再宣稱是「純 A8 誤差」。
- `candidate - matched`：兩者使用同一 qSiLU checkpoint、相同 activation placement、相同 A8 calibration 與相同 evaluator，唯一差異是 `backbone_early` W8；因此這才是本輪可歸因給 W8 的 incremental effect。

所有 delta 都是 mAP50–95 的絕對值差；例如 `-0.01` 是下降 1 個百分點，不是下降 1%。

## 不可變輸入與執行契約

### 模型

- accepted checkpoint：`/home/uxin/yolo/yolo_combine/final/full35/weights/combined/inference/best_joint.pt`，SHA-256 `d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c`。
- qSiLU parent checkpoint：`/home/uxin/yolo/yolo_activation/artifacts/runs/full35/short-recovery-v2-lr01-uniform-qsilu-pq-seed1/inference/best_joint.pt`，SHA-256 `7679186695317e431cd7deb17289f426f4b39b7a4993e4548e74f5ba2766190e`。
- reviewed W8 diagnostic：[`weight-ptq-qsilu-backbone-early-w8-diagnostic-v3.json`](../../artifacts/reports/weight-ptq-qsilu-backbone-early-w8-diagnostic-v3.json)，SHA-256 `0a61dc6eafead2f410fadf1373b4bf14c4e5d948a481ece81884e23821a8ab85`。
- 執行計畫：[`v4-qsilu-backbone-early-w8-search-v1.yaml`](../../configs/experiments/v4-qsilu-backbone-early-w8-search-v1.yaml)，執行時 SHA-256 `3da5d7370685cc5c5efb47f24d1c33d4d6f710d25fdb7d7493dbd7e9d97b3985`。

### 資料與 evaluator

- Detect：完整 COCO val，並輸出 `coco/person/box/map50_95`，沒有省略老師要求的 person 指標。
- Pose：canonical `bbat5-v1` 既有 `pose-search.yaml` 的固定 600 張 search-val；它來自 formal-train scope，不碰 683 張 formal val。
- BBAT5 runtime View：[`manifest.json`](../../artifacts/datasets/bbat5-v1-pose-search-v1/manifest.json)，5,364 train／600 val、source-group overlap 0、`assignment_changed=false`。View 只含直接指回 canonical entry 的 symlink，沒有複製或改動影像／label。
- calibration：固定 manifest 的 COCO train 32 張＋BBAT5 train 32 張，124/124 observer 有效，範圍 `[-0.285666, 268.327606]`。
- evaluator：Full35 官方 `JointValidator`、Bit-True backend、640 px、Detect batch 32／Pose batch 16、workers 4／8、rect validation、無 train augmentation。
- Ultralytics 對 COCO final validation 會自動開啟 `predictions.json` 與 faster-coco-eval，即使傳入的 `save_json` 是 false；三個角色都走相同行為，因此不造成比較不公平。八項 gate 使用 Full35 `extract_*_metrics` 的固定 key，和既有 accepted `0.498022` 契約一致，不混用 console 額外印出的 COCOeval AP。

### Graph

- 三個角色都是 BN-folded deployment View，BatchNorm 數量為 0。
- 148 個部署候選 weight modules、22,571,840 個 deployment weights。
- 4 個 Binary Q/K modules、131,072 個 weights 持續 protected，沒有套一般 W8 quantizer。
- qSiLU／A8 wrapper 在 official Detect 與 Pose task materialization 後仍存在；CPU 整合測試另證明 W8 candidate weights 會傳入 task model，離開 quantization context 後 shared model 權重逐位元還原。
- training-only O2M、pose flow 與 sigma 不計入部署量化收益。

## 八指標結果

| 指標 | accepted | matched qSiLU+A8 | candidate + W8 | parent delta | W8 incremental | total delta |
|---|---:|---:|---:|---:|---:|---:|
| COCO box | 0.498022 | 0.484891 | 0.484518 | -0.013131 | -0.000373 | -0.013504 |
| COCO person box | 0.620381 | 0.609584 | 0.609470 | -0.010797 | -0.000115 | -0.010912 |
| BBAT box | 0.836442 | 0.834941 | 0.829223 | -0.001502 | -0.005718 | -0.007219 |
| BBAT pose | 0.976784 | 0.981120 | 0.980946 | +0.004336 | -0.000174 | +0.004162 |
| BBAT ball box | 0.779040 | 0.777961 | 0.770814 | -0.001080 | **-0.007147** | -0.008227 |
| BBAT bat box | 0.893845 | 0.891921 | 0.887632 | -0.001924 | -0.004288 | -0.006212 |
| BBAT ball pose | 0.959489 | 0.970034 | 0.969455 | +0.010545 | -0.000579 | +0.009966 |
| BBAT bat pose | 0.994078 | 0.992205 | 0.992437 | -0.001873 | +0.000232 | -0.001641 |

parent delta 只是完整 parent policy 差異；本報告不把正 delta 解讀成量化必然提升，也不把不同 checkpoint 的差異單獨歸因給 qSiLU 或 A8。

## Gate 判定

| Gate | 門檻 | 最差結果 | 判定 |
|---|---:|---:|---|
| candidate 對 accepted，八項 total | 每項 `>= -0.04` | COCO box `-0.013504` | 通過 |
| candidate 對 matched，W8 incremental | 每項 `>= -0.01` | BBAT ball box `-0.007147` | 通過 |
| QAT matched sham | 本輪不適用 | `sham_gate_applicable=false` | 不判定 |

machine-readable decision 是 `green`。JSON 中 `maximum_sham_drift=0` 只是沒有提供 PTQ sham 時的序列化預設值，**不能解讀成已測得 sham 漂移為零**。

W8 incremental 的最小安全餘裕是 `0.01 - 0.007147 = 0.002853`；雖然通過，ball box 已是目前最敏感指標。因此低位元測試必須繼續保留 ball/bat 分類指標，不能只看 BBAT aggregate。

## W8 數值與容量證據

- quantized modules：21。
- weight elements：1,218,240，約占全部 deployment weights 的 5.397%。
- aggregate NRMSE：`0.0054508`。
- SQNR：`45.2708 dB`。
- cosine：`0.9999853`。
- clipping rate：`0.00037349`。
- occupied codes：256/256。
- W8 code：1,218,240 bytes；2,368 個 FP32 scales：9,472 bytes。
- 該區由 FP32 的 4,872,960 bytes 降到 1,227,712 bytes，約 `3.969×`，節省約 `3.476 MiB`。這只是 weight storage proxy，不包含整圖 activation buffer、boundary requant、packing 或硬體 kernel 成本。

## 執行時間與 GPU

- accepted official validation：29.53 秒。
- matched official validation：87.63 秒。
- candidate official validation：87.94 秒。
- qSiLU A8 calibration：3.70 秒，峰值 allocated memory 約 474.58 MiB。
- 裝置：NVIDIA GeForce RTX 5090，PyTorch `2.11.0+cu128`。
- 完成後 GPU 回到 440 MiB／0% utilization；runner 沒有留下訓練程序。

matched／candidate 比 accepted 慢，主要是 Python/PyTorch fake-quant wrapper 的研究實作成本；這不是 packed integer kernel 的硬體 latency，不能拿來宣稱 qSiLU 或 W8 的實際速度。

## 下一步建議與停止線

1. 將這個 W8 cell 標為 **search-eligible**，不是 formal winner。
2. 下一輪先在同一 `backbone_early` parent 做 W7／W6／W5／W4 與 exact uniform W4／exact Fixed SD4 的 output diagnostic；全做診斷，但只把未崩潰且數值合理者送八指標 search。
3. search validation 優先跑 W7、W6 與 W4 最佳強 baseline；W5 作相鄰 bit sentinel。若 ball box incremental 超過 `-0.01`，不硬推該格式，改進 QAT recovery pool 或淘汰。
4. `backbone_early` bit curve 清楚後，再以 W8 往 `backbone_deep → attention-safe → neck → MASF → heads` 做 isolated-region sensitivity；不能把本輪 green 外推為整個 backbone 或全模型 W8 green。
5. Fixed SD4、LS-SD4 與 ternary 只送入 CPU distribution／exact reconstruction 支持的層，並各自保留 matched qSiLU+A8 control。LS-SD4 與 ternary QAT 尚未執行。
6. 本輪到此停止；未自動啟動 V4 其餘格、V4H、V5、QAT、formal validation 或最終訓練。

## 限制

- 只有一個 activation parent × 一個 region × 一個 weight bit 完成 search validation。
- BBAT5 使用 600 張固定 search-val，不是 683 張 formal val；正式結論仍需 locked finalist confirm。
- W8 是 fake-quant/dequantized weight，尚未完成 packed export、bias／padding correction、實際 activation saturation、native integer graph與目標 FPGA／ASIC量測。
- accepted 與 matched checkpoint 不同，所以 total delta 是完整 policy delta；只有 candidate-minus-matched 可乾淨歸因給本輪 W8。
- 沒有 QAT、matched QAT sham、多 seed 或正式訓練結果。
