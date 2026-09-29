# 2026-09-03：poly_shift W8–W4、SD4／ternary 與 QAT 入口

## 變更內容與原因

- 完成 `poly_shift + LSQ+ A8` 累積 W8、isolated W7/W6/W5/W4 exact、mixed-bit、Fixed-SD4、Paper-TWN 與特殊 route 耦合 search validation。isolated region 不能相加外推，因此所有組合都重新跑完整 COCO val 5,000 張與 BBAT5 固定 search-val 600 張。
- 所有 active 判定改為 mAP50＋mAP50–95 雙門檻：八項 total mAP50 每項下降不得超過 `0.015`，八項 total mAP50–95 每項下降不得超過 `0.04`。checkpoint selector 同時使用兩族指標排序。
- `dual_metric_regate.py` 新增透明的八指標診斷平均與固定 `0.2/0.2/0.2/0.4` joint priority score；兩者明標不是 evaluator 官方單一 mAP。
- 新增 V23 mixed-bit、V24 Fixed-SD4／Paper-TWN matched-path control、V27 七格耦合 route 計畫與報告。舊 artifact 不覆寫，全部以 SHA-256 串接。
- QAT runtime 新增 BN-folded training graph、LSQ+ A8 calibration、weight fake quant、scale-only→joint schedule、quantizer獨立 optimizer group、dual metric selector、matched-sham absolute-drift gate，以及 sham 完成／`best_joint` checkpoint SHA-256 前置門檻。
- 依 RTX 5090 smoke 實測，Detect physical microbatch 從32降為16；logical batch保留128，Full35每個 macro 仍依既有 joint contract 使用兩個 logical Detect batch與一個 Pose batch。
- QAT Pose loader 使用既有 `pose-search.yaml` 與 symlink-only runtime View。新增只在QAT context生效的 validator，重新驗證 canonical registry、source YAML hash、5,364/600 split counts、`.rf.` group separation及assignment未改，離開context後恢復上游validator。
- 新增 PTQ／特殊格式／QAT 入口總報告：`docs/reports/2026-09-03-poly-shift-ptq-special-format-and-qat-entry.md`。

## 驗證方式與結果

- 九區 W8 為 dual green：最差 mAP50／mAP50–95 `-0.011855／-0.017322`，總 deployment weight 約 `3.549×`；`backbone_attention_safe` 加入後 mAP50 超限，暫留 FP32。
- isolated W7：8 green、2 recover；W6：7 green、1 recover；W5：5 green、1 recover、1 reject；W4：4 green、1 reject。W4 green 為 MASF、neck attention、Pose tower、Pose predictor；Detect predictor W4 reject。
- V23 mixed-bit 中只有 small-W4 frontier dual green；三個九區低位候選皆只進 recover，不能取代九區 W8 PTQ 主線。
- V24 matched-path 結果顯示 Fixed-SD4 在 Detect tower／predictor 勝過或明顯穩於 W4；Paper-TWN balanced 只進 recover、safe reject。
- V27 七格 mAP50 原判都 green；補 mAP50–95 後，單獨 SD4 tower 因 `-0.040012` 改為 recover，其餘六格 green。accuracy／balanced／hardware 分別為「SD4 predictor」、「SD4 Detect head」、「W4 neck attention＋SD4 Detect head」。
- V27 source SHA-256 `68fd8b3a31844390681e0b1692354c572cfc574d349fdae2bca7062db71dc28e`；dual report `db05395178302647249911773a6a33fc4828006651fac797831a320cc3a00b1b`。
- V19 CPU graph preflight v3 ready：148 weight quantizers、124 activation quantizers、396 qparams、6 qparam groups、4 Binary Q/K protected、final BN=0；artifact SHA-256 `de8783c8cf0200e3b0ec387578ee1d515f1e6e1f878b3880d3b1e7ff7de3d257`。
- GPU smoke 第一次 Detect physical batch32 OOM，當時約佔30.82 GiB；physical batch16重跑通過，峰值 `26,541 MiB`、396/396 qparam有有限梯度。通過 artifact SHA-256 `3ad03a9c282b47d3f20ec3ba1b35fa765b6bba99452a3eba27e9db6ee3c2f551`。
- targeted CPU 測試涵蓋 QAT plan/runtime/graph/weights/schedule/optimizer/calibration/validation、dual metric與mixed policy；最新局部修改均通過 pytest 與 Ruff。完整全專案回歸會在 training 不占用 GPU／檔案後補跑。

## 困難與解法

- 困難：只看 mAP50 會把 SD4 tower 判成 green，但它的最差 mAP50–95 為 `-0.040012`。
- 解法：所有候選同時重讀 hash-pinned 16項 raw metrics；兩族都過才green，並新增整體診斷摘要供排序。
- 困難：physical Detect batch32在完整QAT graph OOM。
- 解法：降為physical16、保留logical128，用梯度累積維持batch語意；先以GPU smoke驗證forward/backward、optimizer step與deployment materialization。
- 困難：上游 Full35 canonical validator只接受formal `pose.yaml`，會拒絕正確的固定search View。
- 解法：QAT context validator不偽造formal metadata，而是重新建立／驗證 `pose-search.yaml` runtime View；上游函式在context結束後恢復。
- 困難：第一次loader失敗留下run目錄，第二次實際使用search View但resolved config仍顯示formal source。
- 解法：兩個未完成run都在epoch完成前停止並完整封存，分別留下 `failure.json`；正式run重新開始且resolved config與runtime manifest一致指向search source。
- 困難：MuSGD的Muon matrix update不適用一維quantizer scale／offset。
- 解法：qparam group強制no-decay且不使用Muon；MuSGD仍須另立matched sham，不能載入AdamW optimizer state。

## 未解事項與風險

- V19 matched sham正在執行；若相對matched parent任一mAP50／mAP50–95絕對漂移超過`0.01`，或沒有gate-feasible `best_joint`，真正QAT會被程式阻擋。
- V19全十區W8、V25 nine-region-quality mixed-bit、Fixed-vs-learned SD4、qSiLU與regional Hardswish coupling尚待paired QAT／validation。
- Fixed-SD4 joint recovery若同時訓練activation與uniform quantizer scale，不能全部歸因為LS-SD4；需要另做固定SD4-scale對照。
- formal validation、多seed與export只給鎖定finalists；目前沒有正式winner。
- bias／padding lowering、實際Add／Concat requant、accumulator saturation、native integer graph、HLS／RTL／板上latency／power仍未完成。
- W5/W6/W7沒有native kernel前只報容量、traffic與BOP proxy，不宣稱GPU speedup。
