# 2026-08-24 Full35 JOINT 完整實作工作紀錄

## 需求與原因

依使用者確認的 spec，實作 YOLO26m Full35 shared backbone/neck、COCO80 Detect head與
ball/bat Pose26 head的正式多任務訓練。首版以軟體／單RTX5090為主，但必須保留 XNOR
與 Bit-True；Partial75只建立隔離備案，不執行。

## 變更內容

1. 新增 graph-preserving dual-head模型與完整 compatibility audit；拒絕 `nn.Sequential`
   截 trunk，也不在 forward跑兩個完整模型。
2. 新增 current Ultralytics 8.4.90 native Detect/Pose26 criterion adapter，完整保留 E2E、
   progressive、RLE、visibility。
3. 新增兩 loader與Detect-primary 2:1 scheduler；tail macro與loss均用 actual batch。
4. 新增 J1/J2/J3 semantic parameter policy、AdamW主線、MuSGD challenger、warmup+cosine、
   grad clip、shared BN policy與hardware-contract guard。
5. 新增 exact tiled XNOR；保留原 bundle SHA與原始程式，透過verified runtime adapter
   安裝。
6. 新增 Float/Bit-True official validators、8-metric hard gate、三種best＋last selector。
7. 新增 EMA、JSONL/CSV/PNG/TensorBoard capability、完整 exact-resume與獨立 inference
   weights。
8. 新增 single-pass `SharedDualPredictor`與train/validate/infer CLI。
9. 新增 standalone baseline validator；獨立 Pose P3保留自己的 trunk，再轉成 Bit-True，
   不把joint初始化模型誤稱獨立 baseline。
10. P1/P2/P3改用 canonical bbat5-v1與token_tile32；超參對齊 Full35來源。P2/P3可 tuning
    MASF與attention的V/PE/projection/bias；Q/K、gamma與Bit-True constants仍鎖定。
11. 建立 Full35 `enabled:true` 與 Partial75 `enabled:false` 的完全隔離 joint YAML／CLI。

主要新增模組：

- `fusion_model.py`, `factory.py`, `xnor.py`
- `joint_data.py`, `joint_loss.py`, `joint_trainer.py`, `formal_training.py`
- `stage_policy.py`, `hardware_contract.py`
- `validation.py`, `standalone_baseline.py`, `metrics.py`
- `resume.py`, `experiment_log.py`, `inference.py`, `joint_cli.py`

## 設定結果

- Full35 shared graph：26,529,701 parameters；獨立pair 45,580,762；減41.796%。
- data：COCO80 + canonical BBAT5 Pose 5964/683。
- train physical目標：Detect128、Pose16；validation Detect32、Pose16。
- macro：2 Detect +1 Pose；exposure16:1；reference loss batch64；weights1:1。
- J1 5、J2 40、J3 manual max20。
- optimizer：J主線AdamW、wd2.7e-4；MuSGD只作後續challenger。
- XNOR：exact bool tiled，token tile32；Q/K STE關閉。
- augmentation：兩任務mosaic0；Detect flip.5、Pose flip0。
- BN：shared running stats freeze；affine YAML可調，預設true；head BN train。
- gate：八項matching mAP50-95，每項最大drop0.08 inclusive。
- seed：seed0先篩選，唯一合格finalist才跑seed1。

## 驗證方式與結果

全部驗證均設定 `CUDA_VISIBLE_DEVICES=-1`，未配置GPU。

1. joint核心 targeted suite：42 passed／63.10s。
2. 真實 Full35 Float→Bit-True Pose state mapping：complete，兩個endpoint tables保留。
3. 真實 COCO/BBAT5、imgsz64、physical1的兩個完整2:1 macro-step：成功、無NaN/Inf、
   shared/Detect/Pose三組gradient都存在。
4. EMA updates=2；Detect/Pose E2E criterion updates都等於1，證明不是每macro更新。
5. shared BN：95 modules、0 changed；head BN：84 modules有更新。
6. XNOR report：backend bool_tiled、token_tile32；factory loading shared587、Detect240、
   Pose411 tensors，missing/unexpected/shape mismatch全為0。
7. CPU smoke loss（僅contract，不是精度）：macro1 Detect7.7884、Pose20.4662、joint14.1273；
   macro2 Detect3.8578、Pose14.0737、joint8.9657。
8. `joint.py preflight` 正確 exit1，且只列出正式 Pose26 checkpoint與八項baseline兩個
   blocker。

可重現證據：

```text
variants/full35/artifacts/formal-joint-cpu-smoke.json
variants/full35/artifacts/formal-joint-cpu-smoke/datasets/
```

## 遇到的困難及解法

1. 舊 XNOR OOM：確認實際發生於epoch2 validation第13 batch，不是train B16。以實際
   B32/rect672推導5.935GiB int64 workspace，改為逐值相同token tiled reduction。
2. sandbox的 `apply_patch` 更新既有檔時發生 `bwrap: loopback: Failed RTM_NEWADDR`。
   解法是先保留既有檔為內部實作或 `/tmp` 備份，再用 `apply_patch` 建立穩定公開
   seam；沒有刪除使用者資料。
3. TensorBoard套件目前不存在。依禁止自行升級dependency的要求未安裝；`auto` 模式
   會清楚記錄disabled原因，JSONL/CSV/PNG照常輸出。
4. 正式GPU不可使用，且當時已有外部GPU工作。所有動作均為CPU／唯讀或workspace
   artifact；沒有啟動訓練、validation或VRAM profile。

## 未解事項與風險

1. canonical Full35 Pose26 P1/P2/P3尚未跑，因此 joint正式初始化權重仍缺。
2. 八項same-evaluator standalone baseline尚未跑，不能計算正式delta/gate。
3. token tiled移除主導allocation，但physical Detect128完整AMP backward仍需空卡
   memory gate；不得預先承諾可放入5090。
4. 正式640精度、latency、能耗、FPGA、INT8/Bit-True硬體datapath均未執行。
5. bat endpoint 0/1語意尚未正式命名，因此Pose fliplr維持0。
6. DDP與compile不是v1支援範圍。

困難：如上。原始資料、標註、權威bundle與既有GPU程序均未修改。

