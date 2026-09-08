# 2026-09-04：V19 parent、V29逐區PTQ與corrected短QAT執行報告

## 結論

本輪已把先前activation、W8–W4、SD4、三元權重與QAT工作串成可執行流程，且沒有改動歷史結果：

1. `poly_shift + LSQ+ A8 + all-W8`的V19 QAT已完成並鎖定Epoch 5。
2. 已對148個deployment weight paths完成五種格式、共740格CPU重建分析。
3. V29已完成十階段、backbone→neck→head的完整COCO val 5,000與固定BBAT5 search-val 600 PTQ驗證。
4. V29最後保留三個低位元單path，全部16項mAP指標仍通過總門檻。
5. 執行後查出recovery floor的符號錯誤；已red→green修正，原始metrics與queue均未覆寫，也未重跑GPU validation。
6. corrected re-gate得到12個recover candidates；依預註冊配額選6個main＋1個sentinel，現正依序執行matched sham→短QAT。

這是search階段結果，不是formal或最終硬體結果。20／60 epoch延伸、multi-seed、formal validation、export與硬體最佳化仍未授權執行。

## 一、前置成果如何接到本輪

### Activation

- active parents保留`qSiLU`、`Hardswish`、`poly_shift`；`poly_quality`只保留歷史證據，不再執行。
- 本輪鎖定parent是`poly_shift + LSQ+ A8`。A7／A6及activation-SD4留到weight Pareto形成後才測。
- Q3 Hardswish證據只支持指定regional policy，不外推為uniform Hardswish或多區winner。

### Graph與硬體邊界

- 251個Conv／Linear中，148個是deployment weight modules，99個training-only，4個Binary Q/K受保護。
- Binary Q/K、MASF拓樸與attention PWL原實作保持不變。
- one-to-many、pose-flow、pose-sigma仍供訓練supervision，但不計部署壓縮收益。
- `topk`、`gather`、box/keypoint decode不套一般weight quantization。

### Weight格式

- uniform：W8、W7、W6、W5、W4；先前整區結果顯示W8最穩，低bit必須細到單path。
- SD4：Fixed-SD4是固定codebook PTQ；LS-SD4是同codebook learned-scale QAT，兩者分開歸因。
- ternary：
  - Paper-TWN v2：`0.70 × mean(abs(W))`的layer-wise static proxy。
  - TWN v3：`0.75 × filter mean(abs(W))`、filter-wise。
  - exact-scaled ternary：相同`{-s,0,+s}` codebook，但scale採全域MSE最佳解。
  - QAT保留FP32 shadow weight；one-shot PTQ不稱為原論文訓練結果。

## 二、V19鎖定parent

V19為`poly_shift + A8 + 148層W8` paired QAT；patience已依使用者要求改為5，Epoch 10驗證後合法early stop，selector仍指向Epoch 5。

| 項目 | 結果 |
|---|---|
| parent id | `v19-poly-shift-a8-all-w8-qat-epoch5` |
| selected epoch | `5` |
| worst total mAP50 delta | `-0.014133355` |
| worst total mAP50–95 delta | `-0.012950486` |
| full-resume SHA-256 | `61f70406e5d5412530298ad26d585dd856942e68830be127891fd61e0877936e` |
| inference SHA-256 | `51a62d5ce60e104e0bf593fbd34769d280ab39cb5b5cbcff9b3f675709b5435b` |
| locked manifest SHA-256 | `6e2034bc6a84ee5f1dac2a22d38afbcdec9763b0fc22e9333b67bb0c1dbd48a7` |

V19通過使用者門檻：八項mAP50各自不得低於accepted `-0.015`，八項mAP50–95各自不得低於`-0.04`；平均值不能掩蓋最差任務。

## 三、148×5 CPU profile

CPU profile涵蓋148 paths×5 formats＝740 measurements，耗時約40.09秒；它只排序候選，不作mAP promotion。

| 格式 | weighted NRMSE | weighted cosine | weighted zero ratio | packed bytes |
|---|---:|---:|---:|---:|
| exact W4 per-output-channel | 0.094415 | 0.983235 | 0.252243 | 11,402,016 |
| Fixed-SD4 per-output-channel | 0.156697 | 0.982515 | 0.043185 | 11,402,016 |
| TWN-v3 filter-wise | 0.438680 | 0.842939 | 0.490919 | 5,759,056 |
| exact-scaled ternary per-tensor | 0.560222 | 0.833535 | 0.606096 | 5,643,552 |
| Paper-TWN v2 layer-wise | 0.677904 | 0.822758 | 0.485737 | 5,643,552 |

CPU delivery manifest為`artifacts/manifests/v28-progressive-weight-cpu-delivery-v1.yaml`；沒有使用GPU、訓練或formal validation。

## 四、V29逐區PTQ結果

V29每階段只在上一個locked policy上加入一個path。W4與Fixed-SD4都reject時，2-bit格式直接prune；green候選才可鎖定，recover留給短QAT。

| 階段 | PTQ結果 | 鎖定結果 |
|---|---|---|
| backbone early | W4 `-0.02843`、SD4 `-0.03992` mAP50，皆未過green | 保留W8 |
| backbone deep | W4 `-0.02025`、SD4 `-0.01852` | 保留W8 |
| backbone attention sentinel | W4／SD4均造成約`-0.88`以上最差下降 | 保留W8；Binary Q/K不動 |
| neck | W4 `-0.01836`、SD4 `-0.02425` | 保留W8 |
| MASF weight | W4、SD4、TWN-v3、exact ternary全green | 鎖exact ternary，容量較小 |
| neck attention FFN | W4／SD4 green；兩種ternary未過green | 鎖Fixed-SD4，精度優於W4 |
| Detect tower | W4／SD4未過green | 保留W8 |
| Detect predictor | W4明顯失敗；SD4只差mAP50門檻`0.000784` | PTQ保留W8；SD4進QAT recovery |
| Pose tower | W4／SD4 green；兩種ternary的mAP50–95失敗 | 鎖exact W4 |
| Pose predictor diagnostic | W4／SD4／exact ternary green，但禁止promotion | 保留W8；TWN-v3作QAT sentinel |

最後三個累積替換：

- `graph.model.16.p3_masf.context.project.conv` → exact-scaled ternary。
- `graph.model.22.m.0.1.ffn.1.conv` → Fixed-SD4。
- `graph.model.23.pose_head.one2one_cv2.1.0.conv` → exact W4。

容量估計：V19 all-W8為22,687,936 bytes；V29 mixed policy為22,424,772 bytes，少263,164 bytes（1.160%）。以22,571,840個deployment weights的FP32 payload作分母，估計由3.980×提升到4.026×。目前收益有限，但符合accuracy-first規則。

### 最終16項指標

| 指標 | candidate值 | 相對accepted delta |
|---|---:|---:|
| `bbat/ball/box/map50` | `0.968600573` | `+0.010218006` |
| `bbat/ball/box/map50_95` | `0.773280180` | `-0.005760002` |
| `bbat/ball/pose/map50` | `0.972037942` | `+0.012549088` |
| `bbat/ball/pose/map50_95` | `0.971970638` | `+0.012481783` |
| `bbat/bat/box/map50` | `0.993158281` | `-0.000920059` |
| `bbat/bat/box/map50_95` | `0.885706628` | `-0.008138061` |
| `bbat/bat/pose/map50` | `0.993609262` | `-0.000469078` |
| `bbat/bat/pose/map50_95` | `0.993051275` | `-0.001027066` |
| `bbat/box/map50` | `0.980879427` | `+0.004648974` |
| `bbat/box/map50_95` | `0.829493404` | `-0.006949031` |
| `bbat/pose/map50` | `0.982823602` | `+0.006040005` |
| `bbat/pose/map50_95` | `0.982510956` | `+0.005727359` |
| `coco/box/map50` | `0.656877247` | `-0.013841760` |
| `coco/box/map50_95` | `0.484924396` | `-0.013097941` |
| `coco/person/box/map50` | `0.830338251` | `-0.009071337` |
| `coco/person/box/map50_95` | `0.610374019` | `-0.010007470` |

最差mAP50為COCO box `-0.013841760`，剩餘門檻餘裕`0.001158240`；最差mAP50–95為COCO box `-0.013097941`，餘裕`0.026902059`。

## 五、recovery gate錯誤與修正

### 症狀

原V29完成後錯誤產生0個short-QAT jobs。以Detect predictor Fixed-SD4為最小反例：worst total mAP50 `-0.015784`雖未過green `-0.015`，但明顯位於recover `-0.04`內，卻被標成reject。

### 根因

YAML已將recovery floors表示成delta：`-0.04／-0.08`；parser直接保留負值，而gate又比較`>= -floor`，實際變成要求`>= +0.04／+0.08`，使recover分支不可能成立。

### 修正與證據

- parser現在把YAML負delta轉為內部正的drop magnitude `0.04／0.08`，並精確驗證四個門檻。
- 新增可重現負號錯誤的regression test，修正前確實red、修正後green。
- 新增immutable re-gate模組；只讀已完成candidate metrics，重新計算gate，不覆寫原state／short queue。
- source V29 state SHA-256仍為`756e2c6fc6432bc9b5f1ce8d2a69a82599cbe9b4f92ea8c42bdf380bf49a1990`。
- corrected report SHA-256為`2c96c5598741bd220a51aed807c3b64dd714b2679c9c227a79bd4d17292e460f`。
- corrected queue SHA-256為`e3f8cef374c06c7855be193561b87e6ef70c1281939dd5461e2759fdcfa73c06`。
- GPU validation重跑次數：0。

原有green/reject的PTQ promotion沒有改變；只修正「未過green但仍值得QAT recovery」的分類。

## 六、corrected短QAT queue

corrected re-gate得到12個eligible recover candidates。依預註冊上限，不把main與sentinel混用配額：最多6 main＋2 sentinel；實際選6 main＋1 sentinel。

| Queue role | Candidate | PTQ worst mAP50 | PTQ worst mAP50–95 | 相對V19節省bytes |
|---|---|---:|---:|---:|
| main | Pose tower exact ternary | -0.013842 | -0.071933 | 337,144 |
| main | Detect predictor Fixed-SD4 | -0.015784 | -0.016058 | 125,948 |
| main | Neck attention TWN-v3 | -0.016555 | -0.017826 | 148,476 |
| main | Neck attention exact ternary | -0.016849 | -0.017625 | 149,496 |
| main | Neck exact W4 | -0.018357 | -0.015020 | 196,608 |
| main | Backbone deep Fixed-SD4 | -0.018521 | -0.016347 | 196,608 |
| sentinel | Pose predictor TWN-v3 | -0.028368 | -0.045725 | 263,548 |

每個candidate順序固定為matched sham→QAT；QAT arm只有在相同計畫的sham完成證據存在時才會開始。

### 訓練超參數

| 項目 | 設定 |
|---|---|
| optimizer | AdamW，fresh state；不把V19 AdamW state錯接到新policy |
| epochs | 最多15 |
| patience | 5 |
| schedule | Epoch 0–2 FP32 blend；3–8六輪ramp；9–14六輪full quant |
| warmup | 1 epoch |
| activation | 沿用V19 `poly_shift + learned A8` state |
| weight warm start | V19 Epoch-5 full-resume的`ema_state` FP32 shadows |
| special path qparams | 由warm-started FP32 weight重新初始化 |
| Detect batch | logical 128、physical microbatch 16 |
| Pose batch | 16 |
| loss weights | Detect 1.0、Pose 0.25 |
| augmentation | accepted Full35原設定；不新增noise |
| retry | 每arm最多重試一次；有`last.pt`則精確resume |
| monitoring | 外部狀態檢查改為600秒event-only；完整console／JSON／CSV／checkpoint仍落盤。已啟動程序內原300秒只用於arm間GPU等待，為保留provenance不重啟。 |

第一個`Pose tower exact ternary` plan已通過真實CPU graph/data/hash preflight：124個activation quantizers、148個weight quantizers、4個Binary Q/K protected、179個BN fold、warm-start載入1,432 tensors、三個override paths重置6個qparam tensors，blockers為空。目前第一個matched sham正在GPU執行；尚無QAT結果可宣稱。

## 七、後續決策順序

```text
7個matched sham→短QAT
  → 每個candidate重新看16項total delta與candidate-vs-sham drift
  → 保留真正恢復且有容量意義的policy
  → 接回backbone→neck→head累積路線重新確認
  → qSiLU依V30另跑完整all-W8 parent、148×5、逐區PTQ與matched QAT平行線
  → 公平比較poly_shift／qSiLU各自形成的Pareto
  → 最多3個locked policies再做uniform Hardswish／Q3 regional Hardswish
  → 必要時才測A7／A6；activation-SD4維持獨立研究branch
  → 使用者確認finalists後，才做20／60 epoch、multi-seed、formal、packing與硬體量測
```

MuSGD若要測，必須另跑MuSGD matched sham，不能與目前AdamW直接歸因。green PTQ也不為了形式強制重訓；長訓練不會拿來盲救明顯reject候選。

## 八、狀態與證據入口

- V19 locked parent：`artifacts/manifests/v19-epoch5-locked-parent-v1.yaml`
- CPU profile：`artifacts/reports/v19-epoch5-progressive-weight-formats-cpu-v1.json`
- CPU delivery：`artifacts/manifests/v28-progressive-weight-cpu-delivery-v1.yaml`
- V29 plan：`configs/experiments/v29-v19-progressive-ptq-queue-v1.yaml`
- V29 frozen state：`artifacts/queues/v29-v19-progressive-ptq-to-short-qat-v1/queue-state.json`
- Recovery correction：`artifacts/reports/v29-v19-progressive-recovery-regate-v2.json`
- Corrected QAT source：`artifacts/queues/v29-v19-progressive-ptq-to-short-qat-v1/short-qat-queue-regated-v2.json`
- Live QAT event：`artifacts/queues/v29-v19-progressive-ptq-to-short-qat-v1/short-qat-regated-v2/execution-status.json`
- Live QAT state：`artifacts/queues/v29-v19-progressive-ptq-to-short-qat-v1/short-qat-regated-v2/execution-state.json`
- 完整console：`artifacts/queues/v29-v19-progressive-ptq-to-short-qat-v1/short-qat-regated-v2/supervisor-console.log`
- 完整qSiLU後續線：`configs/experiments/v30-qsilu-complete-quantization-lane-v1.yaml`

## 限制

- V29是search validation，不是formal validation。
- 靜態packed-byte是格式容量估計，不等於native latency、功耗或板上吞吐。
- corrected QAT仍在執行；任何candidate是否恢復必須等其paired結果，不能由PTQ或CPU NRMSE先宣稱。
- BBAT5仍使用不可變bbat5-v1；沒有重新切分、抽樣、改影像或標註。
