# 2026-08-27：Full35 Recovery 修正與 Sensitivity 優先執行

## 任務與範圍

使用者指出 Full35 權重是 COCO2017 Detect 與 BBAT5 Pose 的合併訓練結果，後續必須直接沿用
`YOLO_COMBINE/final/full35` 的 joint 設定，且不應跳過 sensitivity 就進入 recovery。本輪停止錯誤
佇列、診斷 SiLU 下降、修正資料／訓練契約，並改為先跑完整 validation 的 region sensitivity。

本輪未重新切分、抽樣或改動任何影像／標註；資料 fraction 固定 `1.0`、`resampling=false`。

## 診斷結論

### 權重與資料沒有接錯

- 初始權重確實是
  `/home/uxin/yolo/yolo_combine/final/full35/weights/combined/inference/best_joint.pt`，SHA-256
  `d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c`。
- accepted checkpoint 的完整 BitTrue 重驗仍逐位重現：COCO box `0.4980223365`、BBAT box
  `0.6300355218`、BBAT pose `0.9037171741`。原始 SiLU 權重沒有掉分。
- factory report 是單一 23-layer shared trunk，COCO80 Detect head 與 BBAT5 ball/bat Pose26 head；
  Detect `nc=80`，Pose `nc=2`、`kpt_shape=[2,3]`。
- 實際一個 joint epoch 使用完整 COCO 118,287 images；依 COMBINE macro 比例，Pose loader 循環為
  7,404 exposures（5,964 unique train images、1.24145 passes）。這不是另一個資料 split。

### 下降發生在錯誤 recovery，而不是 accepted baseline

錯誤 SiLU run 在第 1 epoch 後的 BitTrue 指標為 COCO `0.4903873430`、BBAT box
`0.6143490026`、BBAT pose `0.8954170224`；相對 accepted checkpoint 分別是 `-0.007635`、
`-0.015687`、`-0.008300`。BBAT box 因而未通過當時誤設的 0.01 gate。

根因是 activation recipe 把 E4 short-recovery 的 `learning_rate_scale` 寫成 `1.0`，直接使用原 J3
peak LR；但預註冊計畫要求探索 recovery 先用原 recipe 的 `0.1×`。同時，先前自行填入的 activation
gate `0.01` 也沒有經過 SiLU pilot 凍結。這兩項均已修正。

單純增加 epoch 不是根因解法：原 COMBINE J3 在 global epoch 58（J3 內第 6 個 epoch）取得最佳
joint checkpoint，繼續到 global epoch 63 後 COCO/BBAT pose 已回落，才由 patience 5 停止。

## 變更內容與原因

1. `activation-recipe.yaml` 現在直接引用
   `/home/uxin/yolo/yolo_combine/final/full35/configs/joint.yaml`，不再以子專案複本作正式入口。
2. 明確新增 `region_zero_shot_sensitivity`，排在任何 recovery 之前；使用完整 COCO val 5,000 與
   BBAT5 formal val 683，Float/BitTrue 都跑，不更新權重。
3. short recovery、region recovery 與 policy search 的 LR scale 改為 `0.1`：backbone `3.8e-7`、
   neck `1.9e-6`、MASF `3.8e-6`、attention `5e-8`、Detect/Pose heads `5e-6`。
4. COMBINE 原始 `0.08` 保留為相對 standalone fusion baseline 的 safety gate；activation 專用
   `maximum_map50_95_drop` 改回 `null`，等 SiLU pilot 與 sensitivity 後再凍結。
5. Finalist 維持 COMBINE J3 的 20-epoch 上限、patience 5、warm-up 3。若最佳 joint 出現在最後
   3 epochs 且仍改善，SiLU 與 finalists 一起追加 10 epochs，上限 30；不得只替候選延長。
6. 新增可續跑的 `scripts/full35_sensitivity.py`；每個 region 完成後立即寫 summary，重啟時復用已完成
   run，不重算也不覆蓋。
7. 新增 `scripts/full35_profile.py`，直接使用 COMBINE `build_task_loader`，分別在完整 COCO train 與
   BBAT5 formal train 收集 190-site streaming histogram/range/tail 統計，不保存 feature tensors。
8. COMBINE base physical microbatch 是 64；J3 recovery 才套用 release 實跑的 physical 32 override。
   兩者已在文件與 regression test 分開，避免把 4/8 forwards per macro 混為一談。

## 停止與保留狀態

- 後續 recovery queue 已以 exit 130 停止。
- 錯誤 SiLU run 在第 1 epoch Float/BitTrue validation 完成後停止，沒有
  `activation-experiment.json` completion marker，因此不能列入正式結果。
- 已產生的 logs 與 checkpoint 全部保留作診斷證據；未刪除任何權重或資料。

## Sensitivity 結果

11 個 regions 均已使用完整 COCO val 5,000 與 BBAT5 formal val 683 完成。每次只將單一 region
替換成 `poly_shift`，其餘維持 SiLU，且全程沒有 optimizer step：

| Region | Sites | COCO ΔmAP50-95 | BBAT box ΔmAP50-95 | BBAT pose ΔmAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| `backbone_attention` | 3 | -0.000228 | -0.000775 | +0.000571 |
| `backbone_deep` | 21 | -0.001191 | -0.001247 | -0.001273 |
| `backbone_early` | 21 | +0.000255 | +0.000465 | -0.002520 |
| `detect_one2many` | 18 | 0.000000 | 0.000000 | 0.000000 |
| `detect_one2one` | 18 | -0.001220 | 0.000000 | 0.000000 |
| `masf` | 3 | -0.000006 | +0.000202 | -0.000332 |
| `neck` | 33 | -0.000934 | -0.003098 | -0.007104 |
| `neck_attention` | 1 | -0.000075 | -0.000045 | -0.000001 |
| `pose_flow` | 24 | 0.000000 | 0.000000 | 0.000000 |
| `pose_one2many` | 24 | 0.000000 | 0.000000 | 0.000000 |
| `pose_one2one` | 24 | 0.000000 | -0.012352 | -0.000141 |

`neck` 是三任務共用且對 BBAT pose 最敏感的推論區域；`pose_one2one` 對 BBAT box 特別敏感。
`detect_one2many`、`pose_one2many` 與 `pose_flow` 的推論差值為零，是因為它們屬於訓練期輔助／
loss 分支，不可把零差值解讀成部署硬體收益。初步策略因此優先保留 `neck` 與 `pose_one2one` 的
SiLU 對照，先探索 `MASF`、attention、backbone 與 one2one detect；最終 policy 仍須結合完整
training split activation profile 才凍結。

## Profile 與區域配置結果

- COCO Detect train 118,287／118,287、BBAT5 formal Pose train 5,964／5,964 均已完成；沒有抽樣。
- Detect training graph 觀察 118 sites；Pose 透過正式 `NativeTaskLossRouter` 觀察 154 sites，包含 24 個
  loss-only `pose_flow` sites；兩者 `missing_sites=[]`，合併涵蓋完整 190-site manifest。
- 排除 one-to-many 與 flow 訓練期分支後，部署 activation elements 的主要佔比為：
  `backbone_early` Detect/Pose `50.36%/49.61%`、`neck` `19.96%/19.66%`、Detect one-to-one
  `14.23%`、Pose one-to-one `15.50%`。
- `poly_quality` 在所有部署 region 的分布加權 MAE 都低於 `poly_shift`，和 uniform zero-shot 較佳的
  accuracy 一致；但其硬體類別是 constant-multiply polynomial，因此保留為 accuracy extreme，
  `poly_shift` 仍是 shift/APoT hardware extreme。
- 新增 `policy-search-plan.yaml`：訓練期三區固定 SiLU；`neck`、`pose_one2one` 初始保護，只有同預算
  region recovery 通過才解鎖；預註冊 6 個 mixed trials、保留 2 個 adaptive slots，總上限 8。
- 新增 `scripts/full35_profile_analysis.py`，保存 deployment-normalized region cost、分布加權函數／
  導數誤差與 sensitivity risk tier 至 `decision-support.json`。

## 使用者授權的 qSiLU 延伸

依使用者要求新增 `qsilu_pq`，但不把 GELU 近似直接套到 SiLU。候選利用
`SiLU(x)-SiLU(-x)=x`，寫成 `x/2+h(|x|)`；在 `|x|` 的 `[0,1)`、`[1,2)`、`[2,4)`、
`[4,8)` 使用 C1 piecewise quadratics，`|x|>=8` 接 exact ReLU tail。全數係數是 dyadic，四段
共用一次 `u²`；global `(a,b,c)` 為：

1. `(57/256, 0, 0)`；
2. `(23/256, 17/64, -17/128)`；
3. `(-11/512, 91/128, -37/64)`；
4. `(-5/1024, 37/64, -5/16)`。

Phase-0 結果：MSE `2.5127e-5`、MAE `0.003665`、max error `0.010832`、導數範圍
`[-0.125,1.125]`；Q16.10 identity/tail 均 `0 LSB`、最大 float/emulator 差 `1.488 LSB`；ONNX
opset 18 無 `Exp/Sigmoid/Div`。Full35 CPU smoke 成功替換 190/190 sites，Detect/Pose 輸出皆有限。
primary-source 稽核確認 2022 年已有分段二次 SiLU FPGA 實作，且 QSiLU/QGELU 名稱有多重衝突；
因此 stable key 固定為 `qsilu_pq`，只稱「本專案係數候選」，不主張廣義方法新穎。此候選加入
uniform zero-shot 與 short recovery queue，但 region search 仍只使用預註冊的 `poly_shift`；必須等
目前 SiLU pilot 完成後才可執行 qSiLU zero-shot，不得中途混入。

## Corrected E4 啟動狀態

前置 baseline、uniform zero-shot、11-region sensitivity 與完整 profile 全部完成且 preflight
`ready=true` 後，已從 accepted inference checkpoint 啟動
`short-recovery-v2-lr01-uniform-silu-seed1`。實際 resolved config 為 10 epochs、seed 1、warm-up 1、
patience 0、J3 LR `0.1×`；physical Detect microbatch 32、每 macro 256 Detect + 16 Pose，AMP 目前無
溢位。此 run 是 stage-matched sham control，完成前不啟動 activation candidate。新增
`scripts/full35_monitor.py`，以一行 JSON 只讀最新 event 與最新 BitTrue 三指標，避免持續拉取 validator
完整輸出。

## SiLU sham control 指標下降診斷

使用同一套 Full35 BitTrue evaluator 做差分核對後，accepted `best_joint.pt` 與 activation 專案的
未更新 SiLU baseline 六項主指標逐位相同：COCO `map50/map50_95=0.670719/0.498022`、BBAT box
`0.892530/0.630036`、BBAT pose `0.908760/0.903717`。因此 checkpoint 載入、canonical data 與
identity SiLU policy 本身沒有造成下降；`epoch-0000` 是完成第一個 recovery epoch 後的驗證，不是
尚未訓練的 zero-shot。

機制確認為 recovery 從 inference `best_joint.pt` 載入 accepted EMA `state_dict`，再建立新的
AdamW、scheduler 與 EMA。安全的 `weights_only=True` 稽核顯示 inference 檔只有
contract/metadata/state_dict；release 的 `full-resume/best_joint.pt` 另保存 optimizer、scheduler、EMA、
AMP scaler、loader/RNG 與 `global_macro_step=26597`。後續稽核正式實驗規格確認：每個候選本來就必須
從相同 accepted checkpoint 乾淨載入並重建 optimizer/scheduler/EMA，full-resume 只作 lineage 與緊急
exact-resume 參照。因此這是預註冊的 fresh-optimizer sham control，不是 checkpoint loader bug；下降要
用來扣除額外訓練漂移，不能當成 SiLU zero-shot 下降。

epoch 5 相對 accepted baseline 的 `map50_95` 差值為 COCO `-0.001728`、BBAT box `-0.010118`、
BBAT pose `-0.001896`；BBAT box 退化主要集中在 ball（`-0.018704`），bat 僅 `-0.001533`。
原 J3 紀錄也顯示 global epoch 58 是 selected best joint，59 起 joint score 已不再提高，故從最佳點
繼續更新本來就不保證上升；fresh optimizer 會再增加漂移。使用者記憶中的 COCO `0.5+` 有兩個
對應值：accepted validation 的 Ultralytics `detect_raw mAP50-95=0.503682`，以及 joint 前 standalone
baseline `0.506401`；本專案 hard gate 使用明確匯出的 `coco/box/map50_95=0.498022`，另有
`map50=0.670719`，不可混用。

本輪當時只做診斷，沒有修改或停止進行中的 control。2026-08-28 完成規格再稽核後，保留 fresh
control 語意；修正項目改為讓 runtime activation gate 使用 delivered accepted baseline、凍結專用門檻，
並明確記錄 full-resume 的 lineage-only 角色。不得把本次 sham run 的下降誤認為 SiLU zero-shot 下降。

## 驗證方式與結果

- 新 regression tests 先呈紅：缺少 pre-recovery sensitivity phase，且 LR scale 實際為 `1.0`。
- 修正後 targeted tests 通過；新增 loss-aware profiler、profile analysis 與 bounded search plan 後，完整
  suite `45 passed, 4 warnings`。兩個 warning 均為既有 ONNX legacy exporter deprecation。
- 改為直接讀 COMBINE joint config 後，完整 hash/data/environment preflight 為 `ready=true`。
- Ruff 全專案 lint 與 format check 通過（52 files）；未殘留 `[DEBUG-` instrumentation。

## 困難與解法

1. 原始 2026-08-25 完整實驗漏斗把 recovery 排在 region sensitivity 前，但較早任務規格把
   sensitivity matrix 排在 fine-tuning 前；使用者本輪明確要求 sensitivity 優先，因此新增
   `region_zero_shot_sensitivity` 作為 pre-training screen，同時保留後續同預算 region recovery。
2. COMBINE base 與 J3 runtime 的 physical microbatch 不同；以 base=64、J3 override=32 分層記錄。
3. `apply_patch` 更新既有檔案仍受 bwrap loopback 故障阻擋；新增檔使用 `apply_patch`，既有檔只用
   精確字串替換，並立即以測試、Ruff 與唯讀內容檢查驗證。

## 未解事項與風險

- 11-region zero-shot sensitivity 與完整 training split profile 均已完成；corrected SiLU E4 control 正在執行。
- SiLU control 完成前 accuracy budget 仍維持 pending；候選不得先跑，也不能依候選 recovery 結果
  事後調整門檻。
- Activation accuracy budget 尚未凍結；目前只能使用原 COMBINE 0.08 fusion safety gate 防止災難性
  退化，不能事後根據候選結果任意調門檻。
- Target hardware profile 尚未提供，不能宣稱 latency、energy 或 FPGA resource 改善。
