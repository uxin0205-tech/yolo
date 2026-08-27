# YOLO26m Full35 Detect + Pose JOINT 設計

日期：2026-08-24  
狀態：v2 程式與正式前置完成；Full35 等待／執行新版 GPU run

## 1. 模型邊界

正式模型是一個 Ultralytics DetectionModel graph。layers 0–22 只有一份；layer 23 是
`DualHeadPredictionModule`，保存 `f=[16,19,22]`、`i`、`type`、`np` 等 graph metadata，
並把相同 P3/P4/P5 list 分別送入 Detect 與 Pose26。每個 head 都取得自己的 list copy，
不共用 mutable list。

Detect 維持 COCO80 names/nc；目前 application 仍只消費 person。Pose 是
`{0: ball, 1: bat}`、`kpt_shape=(2,3)`、`flip_idx=(0,1)`。Pose26 的 detection、
one-to-many／one-to-one、progressive loss、uncertainty/RLE 與 visibility loss完整保留。

初始化：shared layers 與 Detect head 來自 Full35-A2 Detect Float checkpoint；Pose head
來自 architecture-matched canonical P3 checkpoint。所有 state 依名稱與 shape 載入，
不依 state_dict insertion order；任何差異都 fail-fast。

## 2. Loss 與 macro-step

每個 task criterion 透過不註冊 shared parameters 第二份副本的 adapter，看見自己的
原生 head。raw loss 是 fixed Ultralytics 版本中已乘 physical batch 的 loss。

設 Detect batches 的實際總影像數為 `N_d`，Pose 為 `N_p`：

```text
detect_mean = sum(detect_raw) / N_d
pose_mean   = sum(pose_raw) / N_p
joint_mean  = (w_d * detect_mean + w_p * pose_mean) / (w_d + w_p)
backward    = reference_batch_size * joint_mean
```

正式 joint stage 預設 `w_d=1.0`、`w_p=0.25`、`reference_batch_size=64`。這是根據
舊 run 的 Pose／Detect shared-gradient norm 平均約19.69倍所做的保守降權；仍逐 epoch
記錄兩任務 gradient norm/cosine，不在首版加入GradNorm或PCGrad。最後一個不完整 batch或只有一個 Detect
batch 的 tail macro 都以 actual `batch["img"].shape[0]` 重算，絕不固定乘0.5。

J0 是真正的 Pose-only macro：Detect 不 forward、不更新 Detect head/criterion，active
task weight會重新正規化，因此 `w_p=0.25` 不會把 J0 的 Pose gradient 額外縮成四分之一。

執行順序是 Detect1 forward/backward → Detect2 forward/backward → Pose forward/backward；
之後才 unscale、finite check、clip norm10、optimizer step、scaler update、zero grad、EMA。
因此三張 computation graph 不會同時保留。

E2ELoss progressive counter 每 epoch只更新實際啟用的 task，且寫入 exact-resume
snapshot；J0只推進Pose，J1/J2/J3才同時推進Detect與Pose。

## 3. 資料與排程

Detect 與 Pose 使用兩個 YOLODataset、dataloader、collate、augmentation。不可把 COCO
Detect labels 與 BBAT Pose labels合成同一 dataset。BBAT runtime View 必須能回溯
`bbat5-v1` registry，並保留5964/683 assignment、11-column labels與 visibility 0/2。

Detect loader 定義 joint epoch。每兩個 Detect batches 取一個 Pose batch，Pose 用完即
cycle。COCO 118,287 與 Pose 5,964 的 train count ratio 是19.83:1；physical128/16 下
每 macro 的 image exposure 是16:1。每 epoch記錄兩任務 images、dataset passes與
Pose wrap count。

首版兩任務都 `mosaic=0`。Detect `fliplr=.5`；Pose `fliplr=0`，因 bat endpoint 0/1
尚未有足夠語意證據證明水平翻轉是否應交換。

## 4. Stage 與 optimizer

| stage | epochs | patience | trainable scope | LR |
| --- | ---: | ---: | --- | --- |
| J0 | 8 | 0（固定跑滿） | 只訓練Pose head；shared trunk、Detect head全部凍結 | Pose head 2e-4 |
| J1 | max20 | 8 | shared neck與兩 heads；backbone、MASF、attention frozen | neck7.5e-5、heads2e-4 |
| J2 | max80 | 17 | backbone layers9+、neck、MASF、兩 heads；attention frozen | backbone1.5e-5、neck7.5e-5、MASF1.5e-4、heads2e-4 |
| J3 | max20 | 5 | full low-LR refinement，包含attention可微部分 | backbone3.8e-6、neck1.9e-5、MASF3.8e-5、attention5e-7、heads5e-5；manual enable |

Q/K projection 與 `score.gamma` 保持不可訓練；J2開始調MASF，J3才開放V、PE、
projection與relative bias。PoT/PWL calibration/constants/endpoint tables由guard逐值保護。

J 主線選 AdamW，因 joint fine-tuning 每階段相對短、兩任務梯度尺度不同，且本機固定
YOLO26文件也指出短 fine-tune 通常選 AdamW；MuSGD 更適合長 run，保留為鎖定 recipe
後的 challenger。兩者不可在同一 run 中途更換。weight decay明確0.00027，不套 stock
`nbs` 隱式重縮放。

J0/J1/J2各1 epoch warmup；J3保留3 epoch warmup，之後 cosine 到 base LR 的0.5。四個 semantic ownership 是
backbone、neck（另外標記MASF/attention）、Detect head、Pose head；任何 Parameter
object 出現在兩組會立即失敗。

獨立 Pose baseline 的 P1/P2/P3 patience 分別為10/12/20；P1/P2仍沿用原生 MuSGD
cosine，不另外疊 plateau scheduler。

J2在每個 epoch 的 Bit-True validation後監控 unconditional joint score。只有超過
歷史最佳至少 `min_delta=1e-4` 才算改善；連續8個 epoch 未改善時，從下一個 epoch起
把所有 semantic parameter group 的當期 cosine LR 乘0.5，整個 J2只允許一次。stale
計數到17即早停。optimizer種類、AdamW beta或 MuSGD momentum絕不在 run中途改變；
recovery state會先寫入完整 checkpoint，再執行停止判斷。

## 5. BatchNorm

每次 `model.train()` 後重套 policy。shared BN running mean/var與counter固定；只有
目前可訓練的head BN維持train，因此J0 Detect-head BN也保持eval。shared BN affine預設按 stage scope訓練，也可在 YAML 設為 false；關閉
不影響 head BN affine。

## 6. XNOR 與 Bit-True

保留既有 boolean sign/XNOR與其 Q/K gradient semantics。舊 broadcast 在
`B=32,H=4,N=441,D=32` 的 validation batch會產生約5.935GiB int64 reduction
workspace。physical train128若沿用 auto val256，單一 untiled validation workspace約
47.481GiB，必然超過 RTX5090。

新 backend在 query/key token各以32分塊，channel reduction不分塊，所以 forward逐值
相同、既有 Q/K gradient semantics相同。估算 train B128 workspace由19.531GiB降到
0.125GiB；val B256由47.481GiB降到0.25GiB。這只移除主導單一 allocation，完整
physical128可行性仍需空卡實測。

Float model是訓練 owner；每 epoch EMA state materialize成官方 Float及Bit-True Detect/
Pose graphs分別驗證。Float knots/values轉 Bit-True時不覆蓋 deterministic endpoint
tables；其餘 state必須完整同名同 shape。

## 7. Validation、gate、seed

八項 mAP50-95 hard gate：

1. COCO overall box
2. COCO person box
3. BBAT overall box
4. BBAT overall pose
5. ball box
6. bat box
7. ball pose
8. bat pose

每項相對 matching standalone baseline 的 delta 必須 `>= -0.08`；任何一項失敗都不能
由平均分數抵銷。joint score只在全部 gate可行後選模：0.2 COCO overall +0.2 person
+0.2 Pose box +0.4 Pose keypoint。

`best_detect`、`best_pose`、`best_joint`各自選；`last`每 epoch覆寫。正式 selection使用
Bit-True EMA。seed0先淘汰不合格方案；鎖定唯一 finalist後才跑seed1。只有一個 seed
時結論標 provisional。

## 8. Checkpoint、logging、限制

完整 snapshot含 live FP32、EMA、optimizer、semantic manifest、scheduler、GradScaler、
criterion counters、epoch/macro、兩 loader seed/pass、selector、RNG、config、source/data
provenance。只允許 optimizer/epoch boundary保存；inference-only weights另存。

每 macro/epoch記 JSONL與long-form CSV；輸出 loss、actual batch、LR、gradient norm/
cosine、task image count。每 backend validation與gate也記錄。PNG必有；TensorBoard在
`auto` 模式，缺套件時明確寫 capability而不安裝新 dependency。

v1只支援single GPU AMP、`compile=false`。DDP、PCGrad、GradNorm、Q/K STE、Partial75
正式 run與FPGA皆不在首版執行範圍。
