# architecture_2 統一實驗規格

規格版本：`2.3.0`
狀態：唯一正式規格
生效日期：2026-08-27

本專案承接 `yolo_combine` 選出的 YOLO26m Detect–Pose 融合 winner，評估 C0、C1、C2、C3 的單因子 C3k2 簡化。它不重新選擇融合方法，也不在結果出現前自動決定 C_best。

## 1. 單一規格與可追溯性

規範優先順序如下：

1. `EXPERIMENT_SPEC.md`：唯一正式定義。
2. `configs/**/*.yaml`：由本規格衍生的機器可讀設定。
3. `TRAINING_GUIDE.md`：理由與操作原則。
4. `RUNBOOK.md`：已驗證命令。
5. `plan.md`：只指向本文件。
6. `plan.pdf`、`YOLO26_Codex_Research_Protocol.md`：歷史參考。

每個正式 YAML 必須保存 `spec_version` 與 `spec_sha256`。run manifest、checkpoint lineage 與結果必須保存：

- spec、architecture YAML、effective training YAML、各 dataset YAML 的 SHA256；
- parent checkpoint 與 handoff manifest 的 SHA256；
- Handoff Revision、resolved candidate、seed、環境及執行裝置；
- 狀態標記：`measured`、`simulated`、`estimated`、`not_run`、`blocked` 或 `pending`。

規格變更必須升版並新增修訂歷史。舊結果保留原雜湊，禁止回溯改寫。

## 2. 專案邊界與正式任務

`yolo_combine` 擁有融合候選、雙權重／架構切換與 winner 選擇。architecture_2 只接受選定 winner，建立簡化候選並比較精度與成本。ViTPose 保持外部參考。

正式模型介面為：

```python
model(images, task="detect" | "pose" | "both")
```

實際 Full35 contract 使用單數 `task`，並以回傳 dict 的 `detect`／`pose` keys 證明路由；不得再以舊版
`tasks=` 假設呼叫。正式語意固定為：

- Detect：COCO80，應用層主要消費 person，但不得把 head 改成 2 classes。
- Pose：BBAT5，`nc=2`、`names={0: ball, 1: bat}`、`kpt_shape=[2,3]`。
- BBAT5 2-class Detect 是獨立診斷 view，不取代 COCO80，也不得單獨決定 C_best。

Pose 執行仍由使用者逐次決定。指定 Pose 資料位置不等於 opt-in；但使用者已於 2026-08-27 對本輪
`full35-j3-float20-seed0` 明確指定 `pose=true`，因此本輪正式 run YAML 的 authorization 與
training gate 都必須固定為 true。未來其他 run 仍須另行取得選擇。完整融合模型排名必須同時具備
Detect 與 Pose 指標。

## 3. 上游 Handoff Revision

正式 handoff 必須 fail closed，至少包含：

- producer=`yolo_combine`、唯一 revision ID、winner ID 與 `fusion_kind`；
- state-dict-only checkpoint、builder/config、architecture、training recipe、selection、fresh-process report；
- COCO Detect、BBAT5 Pose、BBAT5 Detect 診斷資料設定；
- 每個檔案的路徑與 SHA256；
- PyTorch、Ultralytics 及上游 Git revision；
- `detect_nc=80`、COCO80 names、`pose_nc=2`、ball/bat names、`kpt_shape=[2,3]`；
- `model(images, task=...)` 契約與 detect/pose/both forward 證據；
- Candidate Regions、protected paths、head paths、inherited MASF/attention/PWL frozen paths；
- 完整 resolved training recipe、task ratio、loss、optimizer、LR、augmentation、batch、nbs、seed 與 freeze policy。

允許的 `fusion_kind`：

- `shared_dual_head`：一個 shared region。
- `routed_dual`：detect-specific 與 pose-specific regions。
- `partial_shared`：至少一個 shared region，另可有 task-specific regions。

不同 Handoff Revision 的 checkpoint、配方或結果禁止混用。沒有通過 handoff 驗收時，Phase B/C 保持 blocked。

### 3.1 本輪鎖定的 Full35 final

本輪只接受 `/home/uxin/yolo/yolo_combine/final/full35/` 已升格的 accepted J3 immutable snapshot。
J2 僅保留於 `weights/rollback/j2/`，不得與 J3 parent 混用。parent 為
`weights/combined/inference/best_joint.pt`，SHA256 是
`d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c`，release state 為
`accepted_j3_with_j2_rollback`，環境固定 PyTorch
`2.11.0+cu128` 與 Ultralytics `8.4.90`。

實際 graph 是 `graph_shared_dual_head`：Detect 為 COCO80，Pose 為 ball/bat 2 classes、
`kpt_shape=[2,3]`，P3/P4/P5 inputs 為 layers 16/19/22，strides 為 8/16/32。候選區域固定為：

- `graph.model.6`、`graph.model.8`、`graph.model.13`、`graph.model.19`；
- heads `graph.model.23.detect_head`、`graph.model.23.pose_head` 受保護；
- `graph.model.10.m.0.attn`、`graph.model.16.p3_masf`、
  `graph.model.22.m.0.1.attn` 永久凍結。

architecture_2 必須從 final package 的 factory 重建 graph、嚴格載入全部 1,238 tensors，再各自 graft
C0～C3。驗證用的官方 Detect/Pose templates 也必須 graft 同一候選後才 materialize，禁止以 baseline
shape template 驗證已縮窄的候選。

## 4. C0 與候選矩陣

### 4.1 控制組

- **C0-Handoff**：上游 winner 的精確重建；state_dict 與輸出必須逐張量等價，不訓練。
- **C0-Control**：從 C0-Handoff 開始，使用與候選相同的 architecture_2 恢復訓練預算。C1–C3 只與它比較。

### 4.2 候選因子

| ID | 唯一變更 | 預期成本效果 | 主要精度風險 |
|---|---|---|---|
| C0 | 無 | 不變 | 無新增風險 |
| C1 | `e: 0.5 → 0.375` | 隱藏通道約縮窄 25%，Params/MAC/VRAM 下降 | feature capacity、小物件 |
| C2 | `inner_n: 2 → 1` | 減少內部 bottleneck 與深度 | context、表徵深度 |
| C3 | `3x3_3x3 → 1x1_3x3` | 第一個 spatial conv 成本明顯下降 | receptive field、ball/bat keypoints |

第一輪只允許 C1、C2、C3，禁止互相組合。C3-P5、R1、`e=0.25` 與其他組合只是未來議題；必須先看完 Float 結果、由使用者決定並升版 spec 才能加入。

### 4.3 候選區域解析

實際 module paths 禁止寫死，必須由 Handoff Revision 宣告並經 graph audit 驗證：

- shared winner：C1/C2/C3 套用 shared region。
- routed winner：分別解析為 D-C1..3 與 P-C1..3。
- partial-shared winner：依實際 regions 解析為 S-C*、D-C*、P-C*。

每個 resolved candidate 只能修改一個 region 與一個 factor。heads、融合 topology、MASF、BinaryQK、PWL attention、未選定 task branch 及 handoff protected paths必須保持不變。每個候選獨立從同一 C0-Handoff 建立，使用相同 deterministic seed，不得從另一候選初始化。

Transfer report 必須列出 matched、missing、unexpected、shape-mismatch tensors，並證明 parent 未被修改。graph snapshot 只供閱讀與報告，必須標示：

```yaml
standalone_loadable: false
builder: achitechure_2
```

## 5. 資料規格

原始資料保持唯讀：

- Detect：`/home/uxin/yolo/original/pose/detect_dataset/`
- Pose：`/home/uxin/yolo/original/pose/dataset/`

若需要修補或重分割，只能建立 immutable 衍生版本：

```text
/home/uxin/yolo/original/pose/derived/bbat5-v1/
```

該目錄必須有中文 README，並清楚列出來源、分割、修補、授權、重建命令與限制。規則如下：

- Pose labels 是 ball/bat 唯一權威標註；Detect labels 由每列前五欄衍生。
- 原始 Detect labels 只作一對一一致性 audit。
- 依檔名 `.rf.` 前 prefix 分組，以 seed 0 做 grouped 90/10 train/val。
- 同源影像不得跨 train/val；正式 val 不參與搜尋。
- hyperparameter search 只能在正式 train 內再做 grouped search split。
- 四個已知極小負座標只在衍生 Pose labels clamp 為 0，逐項寫入 patch manifest。
- Detect/Pose 共用同一 source assignment；不建立不存在的 test split。
- 本機 canonical 衍生版的影像使用 symlink，labels 寫入衍生版本；原始來源不得改動。
- BBAT5 formal val 與 COCO train view 的重疊必須列入 exclusion manifest；未能證明時標示 blocked，不能假裝通過。
- v1 建立後不可覆寫；任何規則或資料內容變更建立 v2。

`prepare-pose-data` 必須能從原始資料重建本機 canonical 版本。依使用者 2026-08-22 的明確授權，GitHub 另保存一份不可覆寫、可攜且不含 symlink 的完整 snapshot：

- 固定位置為 `artifacts/datasets/bbat5-v1/github-dataset/`，不得散落到專案其他位置。
- 允許提交 Pose／Detect 兩個 view 的影像、labels、portable dataset YAML、split lists、README 與 publication manifest。
- snapshot 只改變儲存形式，不改變 bbat5-v1 的 2.0.0 資料 lineage、group assignment 或四筆 patch。
- Pose／Detect 影像內容必須逐檔相同；匯出時不得留下指向 `/home/uxin/...` 的 symlink 或絕對 split path。
- 禁止提交 checkpoint、weight、cache、run、`.pt`、`.pth`、`.onnx` 或 deployment engine。
- `export-github-dataset` 必須預設只規劃，只有 `--execute` 才能建立 snapshot，且目的地存在時 fail closed。

依使用者 2026-08-23 的明確授權，GitHub 另發布根 repository 的 `original/` 可追溯來源樹：

- 包含 raw Pose `dataset/`、歷史 Detect `detect_dataset/` 與 canonical `derived/bbat5-v1/` 的
  影像、labels、YAML、README、split lists 與 lineage manifests。
- raw/basic split 只供重建與歷史稽核；上傳不改變所有新 run 必須使用 `bbat5-v1` registry 的規則。
- canonical image links 在 Git tree 使用 repository-relative symlink，且必須解析到同一份已發布
  raw image；不得新增資料副本、改寫 labels、改變 split 或修改 2.0.0 data lineage。
- 永久排除 weights、checkpoint、cache、runs、`.pt`、`.pth`、`.onnx`、engine 與 deployment artifact；
  Git 已追蹤的 original 權重必須從 0823 tree 移除，但本機檔案保留。
- `detect_dataset.zip` 為已解壓目錄的重複封裝且超過 GitHub 100 MB 單檔限制；完整解壓目錄必須
  發布，zip 不提交。portable canonical snapshot 仍保留於既有固定位置。

## 6. 正式 YAML 契約

`configs/catalog.yaml` 是正式檔案索引。候選、handoff、dataset、training、quantization 均使用標準 schema，未知欄位或 spec hash 漂移一律失敗。

Training YAML 必須把常用參數放在單一檔案中並說明來源，包括：

- `batch`（Ultralytics 鍵名，不是 `batch_size`）、`fraction`、augmentation `scale`、`cache`；
- `imgsz`、task ratio、nbs、optimizer、LR、weight decay、warmup、seed；
- loss、augmentation、workers、device、AMP、deterministic、freeze 與 validation；
- `model_scale=m` 與 augmentation `scale` 必須分開，不得混用。

Ultralytics 8.4.90 的 `fraction` 只截取排序後 train 清單的前段，不是隨機、分層或 grouped
sampling；不得用它取代 BBAT5 grouped search split。`cache=True` 代表 RAM 且可能破壞完全
deterministic，`cache=disk` 會在影像旁寫檔，只能指向獲授權的可寫衍生資料。

### 6.1 固定20%架構篩選 View

依使用者 2026-08-26 決定，正式長訓練前先對 C0-Control、C1、C2、C3 執行 Float 固定20%初篩。這是
run-specific screening View，不建立新的 canonical dataset，也不修改或複製影像／標註：

- COCO 從 train2017 以 seed 0 random-without-replacement 抽取約20%作訓練，另取互斥的 5,000 張
  train-only search-val；官方 val2017 不參與初篩。
- BBAT5 只從既有 search-train 依 `.rf.` 前 prefix 整組抽取約20%，Pose／Detect 共用 assignment，
  沿用既有 search-val；formal val 不參與。
- 固定清單、來源 hash、輸出 hash、類別統計與 leakage 證據保存於
  `artifacts/datasets/architecture-screen-20-v1/`；所有候選共用同一份 manifest。
- training YAML 的 `fraction` 保持 1.0，禁止再次截取 manifest。
- 20%結果只標示 `screening`／`estimated`，不得自動淘汰候選、選 C_best 或取代完整資料確認。


在 winner handoff 前，必須由上游決定的值以 `source: handoff`、`value: null` 明示，不得捏造。handoff 通過後產生一份完整 effective YAML；所有候選共用同一份，除 candidate/name/project/device/workers/cache 外不得不同。learning field 的 CLI override 一律拒絕。

`config-check` 必須：

- 驗證本機 Ultralytics 8.4.90 與全部 schema/spec hashes；
- 確認 catalog 無漏檔、無廢棄融合範本、候選只有 C0–C3；
- 確認 candidate factor 唯一且 target paths 由 handoff 解析；
- 列出 accepted-but-inactive、deprecated、overridden 與 blocked fields；
- 驗證 dataset 語意、Pose opt-in、量化 eligibility 與 CPU-only policy；
- 未知欄位、不支援值、hash 不符或 learning override 時 fail closed。

## 7. 訓練與公平性

本專案不宣稱重製官方內部 pretraining。C0-Control 與所有 resolved candidates 繼承 winner 的完整訓練配方，不進行 per-candidate tuning。

- COCO batch 只計 Detect loss。
- BBAT5 Pose batch 只計 Pose loss；缺少 person label 不得當成背景。
- BBAT5 診斷 paired batch 可計 Detect+Pose，但不得取代主資料。
- 每個候選從相同 C0-Handoff 建立新 optimizer。
- 永久凍結 inherited MASF/attention/PWL 的 parameters、buffers 與 eval state。
- 只解凍 changed region、相關 heads 及 handoff 允許的恢復區域；未修改 task-specific branch 保持凍結。
- 公平性以 optimizer steps、各 task effective samples、task ratio、accumulation、physical batch 與 validation events 為準；epoch 只作估算。
- seed 0 完成全部第一輪；需要正式確認的候選與 C0 再跑 seed 1，報告 mean 與 dispersion。
- extension 只在 handoff recipe 定義的 late-improvement gate 通過時執行。真正 resume 必須從候選自己的
  未 strip continuation checkpoint 恢復 optimizer、scheduler、scaler、EMA 與 epoch。Ultralytics 8.4.90
  正常完訓後的 stock `last.pt`／`best.pt` 已被 strip，禁止把它們宣稱為真正 resume；若上游未保存
  完整 continuation state，extension 保持 blocked，或另行升版定義「載入 weights、新 optimizer」的延長
  fine-tune。

執行順序固定為：S0 無資料 graph／Params／MACs 驗證；S1 C0–C3 Float 固定20%初篩；S1b 只對
結果接近者增加共同 seed；S2 由 C0 與使用者選定 finalist 使用完整資料確認。S1 以相對 C0 delta、
相同 optimizer steps／validation events 與學習曲線判讀；短訓練截止時仍明顯改善者標為 uncertain，不自動淘汰。

### 7.1 本輪 Float20 執行契約

- C0～C3 各自從同一 Full35 J3 EMA parent 建立新 MuSGD optimizer，固定 seed 0、640、AMP、
  deterministic、最多20 epochs、每 epoch Float search-val，不 early-stop。
- Detect logical batch 為128；依 Full35 J3 已驗證經驗，先用 physical microbatch 32 × 4 accumulation。
  佇列必須在任何候選正式 step 前先以 C0 跑完整一個 macro memory gate；若 OOM，四個候選一律共同改成
  physical 16 × 8，不得只對單一候選偷改。
- 每個 macro 使用兩個 logical Detect batches，因此 physical 32 時為8個 Detect microbatches；
  validation Detect batch 固定16。
- 若使用者 opt-in Pose，Pose train batch 與 validation batch 都固定16，所有候選使用相同 task weight、
  losses、資料與 validation events；若不啟用，結果只能標示 Detect-only screening。
- inherited MASF／attention LR 固定0；backbone `3.8e-6`、neck `1.9e-5`、Detect/Pose heads
  `5e-5`，momentum `0.948`、weight decay `0.00027`、warmup 1 epoch。
- 每 epoch 保存未 strip 的 model、EMA、optimizer、scheduler、scaler、criteria、RNG、loader state 與
  lineage；中斷只可從候選自己的 `last.pt` exact resume。
- C0-Control 的 search-val 各自決定 Detect box、Pose box、Pose keypoint confidence threshold；C1～C3
  固定沿用。AP50、AP50-95、per-class、Macro F1 與 Micro F1 分開保存。

使用者已於 2026-08-27 授權本輪 GPU 時段與 Pose=true。queue 仍須以「無 compute process、
free VRAM ≥30,000 MiB、utilization ≤10%，連續3次」才啟動，不能因目前空閒略過穩定性 gate。
Float20 完成後停止於使用者 review gate，不自動執行
PTQ、QAT-lite、正式 QAT、C_best 或完整資料長訓練。

## 8. 指標、報告與選擇

必須分開保存：

- mAP50 與 mAP50-95；
- COCO80 box overall、COCO person、BBAT5 Pose box、BBAT5 keypoint；
- ball/bat 各類別 AP50 與 AP50-95；
- Precision、Recall、per-class F1、Macro F1（主要）與 Micro F1（參考）；
- F1 所用 confidence threshold。

AP50 使用單一 IoU/OKS=0.50；AP50-95 平均 0.50:0.95。Pose box 使用 IoU，keypoint 使用 OKS。F1 threshold 只能用 C0-Control 的 search-val 決定一次，之後對所有候選與 formal val 固定。

Ultralytics 8.4.90 的 stock Pose `best.pt` 依 Pose mAP50-95 與 Box mAP50-95 的 combined
fitness 選擇；本案若以 Pose mAP50-95 為研究主指標，必須另外保存 pose-only best，並同時報告官方
combined fitness。BBAT5 的非 COCO `[2,3]` keypoint schema 使用套件的均勻 OKS sigma，報告不得
宣稱該 sigma 已經過棒球任務校準。Micro F1 是本專案依固定 confidence／OKS 規則聚合 TP/FP/FN 的
衍生指標，不是 Ultralytics 內建 property。

第一輪不使用固定精度 drop 自動淘汰。舊的 0.005/0.008 band 只能列為描述性敏感度，不得自動觸發候選或選出 winner。報告完整 Float 結果、成本與 Pareto 後：

- `c_best: null`
- `selection_status: pending_user_decision`
- C1/C2/C3 的 `quantization_eligible: pending`

由使用者決定可接受的精度—成本權衡、C_best、量化候選及是否新增 C3-P5/R1/組合。

## 9. 量化

C0 固定可進量化；其他候選只有使用者看完 Float 結果後明確核准才可執行。每個 eligible candidate 可依序執行：

- Q0：fused FP32 reference 與等價性。
- Q1：固定 calibration set 的 W8A8 PTQ simulation。
- Q2L：W8A8 QAT-lite simulation；從同候選 Q1 calibrated lineage 建立新 optimizer，固定200 optimizer steps，
  前50 steps 更新 observers、之後凍結，每50 steps validation；只判斷短 QAT 能否恢復部分 PTQ gap。
- Q2：完整 W8A8 QAT simulation；需另外取得 GPU 長訓練授權。

Conv weights 採 per-channel symmetric INT8，activations 採 per-tensor affine INT8。custom BinaryQK/PWL arithmetic 排除並列明。
Q2L 使用固定20% View、`lr0=parent_lr0×0.1`、AMP 關閉、最多3 epochs且以200 steps先到者為準，
不 early-stop、不自動接受量化。必須分開報告 `Q0-Q1`、`Q0-Q2L` 與 `Q2L-Q1`；正式 Q2 仍只在
完整 Float 決策後由使用者另行授權。所有結果標示 `simulation_only=true`；CPU 現階段只驗證 plumbing
與 regression，不宣稱正式部署 latency。

Fake quant 仍在浮點 tensor 上模擬 quantize-dequantize；未完成 backend convert、operator coverage、
accumulator／requantization／saturation 與目標硬體驗證前，不得稱為真實 INT8 deployment 或 Bit-True。

## 10. Phase Gates 與交付

### Phase A：現在執行

- 規格、中文 README、正式 YAML/schema、資料工具與 manifests；
- handoff validator、候選 scope resolver、state-dict checkpoint contract；
- CPU-only config/data/graph/forward/loss/backward/freeze/reload/dry-run tests；
- 64/128 fixture forward 與一次 640 geometry test；
- 狀態完成後標示 `ready_for_upstream_handoff`。

### Phase B：Full35 final 驗收

- 驗收 immutable Full35 J3 handoff、建立 C0-Handoff、執行實際 graph audit；
- 解析 C0～C3 真實區域並完成 CPU 結構、forward、reload 與 materialize 驗證；
- 完成後標示 `ready_for_float20_queue`。

### Phase C：本輪已授權範圍

- Pose gate 已由使用者明確設為 true；C0-Control、C1、C2、C3 依同一 batch plan 跑固定20% Float seed 0；
- 使用者檢視初篩與必要的共同追加 seeds，再決定 C0／finalists 是否執行完整 Float；
- 完整指標、成本、Pareto 與 C_best 都保持 `pending_user_decision`；
- eligible candidates 的 Q0/Q1/Q2L/Q2 均不在本次自動佇列，需 Float 結果與後續授權。

本輪只排程 Float20 screening，不執行完整資料長訓練或任何量化訓練。成果必須輸出
`REPORT.md`、CSV/JSON、`selection.json`、profiles、figures、lineage、effective configs 與中文工作紀錄。

## 修訂歷史

| 版本 | 日期 | 內容 |
|---|---|---|
| 1.0.0 | 2026-08-18 | 建立舊 Detect/Pose/量化規格。 |
| 1.1.0 | 2026-08-18 | 中文化與 Pose opt-in。 |
| 1.2.0 | 2026-08-19 | 單檔 training YAML 與設定檢查。 |
| 2.0.0 | 2026-08-22 | 改為承接 `yolo_combine` winner；動態解析 shared/routed/partial regions；C0-Handoff/C0-Control 分離；BBAT5 v1 衍生資料；measurement-first 選擇與逐候選量化資格；CPU-only Phase A。 |
| 2.0.1 | 2026-08-22 | 依 Ultralytics 8.4.90 官方原始碼修正 extension resume：只接受未 strip continuation state；補充 fraction/cache、Pose-only best／combined fitness、uniform OKS sigma、Micro F1 與 fake-quant 邊界。BBAT5 v1 保留建立時的 2.0.0 lineage。 |
| 2.0.2 | 2026-08-22 | 依使用者明確授權增加完整 GitHub dataset snapshot；影像與 labels 物化於獨立 `github-dataset/`，保留 bbat5-v1 既有 split／patch lineage，並持續禁止權重、checkpoint、cache 與 runs。 |
| 2.0.3 | 2026-08-23 | 依使用者明確授權發布 `original/` 的 raw Pose、歷史 Detect 與 canonical lineage；Git tree 排除權重、cache、run 與重複超限 zip，canonical images 使用相對 symlink，正式訓練入口仍固定為 bbat5-v1 registry。 |
| 2.1.0 | 2026-08-26 | 核准 C0–C3 固定20% Float 初篩；新增不可變 train-only manifests、Float20 template 與 Q2L 200-step QAT-lite 相容性檢查；正式 val、C_best、Pose opt-in 與 GPU 執行 gate 保持不變。 |
| 2.2.0 | 2026-08-27 | 鎖定 Full35 accepted J2 final handoff 與真實 `task=`／graph paths；核准排在現行 GPU 工作後的 Float20 queue、Detect logical128/physical32→16 公平 OOM gate、validation16、exact resume 與 Float-only validation；Pose 仍須使用者對本次 run 明確選擇，量化不自動啟動。 |
| 2.3.0 | 2026-08-27 | 上游正式將 J3 升格為 accepted、J2 移至 rollback；本輪 parent 改鎖 J3 checkpoint `d67fb45c…`，補建獨立 CPU fresh-process 證據，並依使用者明確指示固定 Pose=true、續行完整 C0～C3 Float20 queue。 |
