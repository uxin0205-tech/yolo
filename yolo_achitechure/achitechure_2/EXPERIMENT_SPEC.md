# architecture_2 統一實驗規格

規格版本：`2.0.3`
狀態：唯一正式規格
生效日期：2026-08-22

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
model(images, tasks={"detect", "pose"})
```

允許 `detect`、`pose` 或 `both`。正式語意固定為：

- Detect：COCO80，應用層主要消費 person，但不得把 head 改成 2 classes。
- Pose：BBAT5，`nc=2`、`names={0: ball, 1: bat}`、`kpt_shape=[2,3]`。
- BBAT5 2-class Detect 是獨立診斷 view，不取代 COCO80，也不得單獨決定 C_best。

Pose 執行由使用者決定。未提供明確 opt-in 時，可做設定、資料、graph 與 fixture 驗證，但不得啟動 Pose 長訓練或正式 Pose validation。正式主線排名若要宣稱完整融合模型結果，必須同時具備 Detect 與 Pose 指標。

## 3. 上游 Handoff Revision

正式 handoff 必須 fail closed，至少包含：

- producer=`yolo_combine`、唯一 revision ID、winner ID 與 `fusion_kind`；
- state-dict-only checkpoint、builder/config、architecture、training recipe、selection、fresh-process report；
- COCO Detect、BBAT5 Pose、BBAT5 Detect 診斷資料設定；
- 每個檔案的路徑與 SHA256；
- PyTorch、Ultralytics 及上游 Git revision；
- `detect_nc=80`、COCO80 names、`pose_nc=2`、ball/bat names、`kpt_shape=[2,3]`；
- `model(images, tasks=...)` 契約與 detect/pose/both forward 證據；
- Candidate Regions、protected paths、head paths、inherited MASF/attention/PWL frozen paths；
- 完整 resolved training recipe、task ratio、loss、optimizer、LR、augmentation、batch、nbs、seed 與 freeze policy。

允許的 `fusion_kind`：

- `shared_dual_head`：一個 shared region。
- `routed_dual`：detect-specific 與 pose-specific regions。
- `partial_shared`：至少一個 shared region，另可有 task-specific regions。

不同 Handoff Revision 的 checkpoint、配方或結果禁止混用。沒有通過 handoff 驗收時，Phase B/C 保持 blocked。

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

`prepare-pose-data` 必須能從原始資料重建本機 canonical 版本。

依使用者 2026-08-23 的明確授權，資料發布與位置契約固定如下：

- `/home/uxin/yolo/original/pose/` 是全 repository 唯一 BBAT5 資料資產庫。
- Git 發布 raw Pose `dataset/`、歷史 Detect `detect_dataset/` 與 canonical `derived/bbat5-v1/` 的
  影像、labels、YAML、README、split lists 與 lineage manifests。
- raw/basic split 只供重建與歷史稽核；所有新 run 只能使用 `bbat5-v1` registry／Task View YAML。
- canonical image links 在 Git tree 使用 repository-relative symlink，且解析到同一份 raw image；
  不得新增資料副本、改寫 labels、改變 split 或修改 2.0.0 data lineage。
- 各專案 `artifacts/datasets/` 不得提交或保存資料、dataset YAML、split、labels 或 manifests 副本。
- 2.0.2 的 architecture_2 portable snapshot 只保留於 Git 歷史；目前 tree 移除該目錄，
  `export-data-metadata`、`export-github-dataset` 與 `validate-github-dataset` 不再是正式 CLI。
- 永久排除 weights、checkpoint、cache、runs、`.pt`、`.pth`、`.onnx`、engine 與 deployment artifact；
  Git 已追蹤的 original 權重從 0823 tree 移除，但本機檔案保留。
- `detect_dataset.zip` 為解壓目錄的重複封裝且超過 GitHub 100 MB 單檔限制；
  發布完整解壓目錄但不提交 zip。

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

GPU 正被其他工作使用。本 revision 只允許 CPU Phase A 驗證；任何正式訓練、CUDA smoke、GPU latency/VRAM、QAT 或 CUDA compile 都等待使用者明確授權。

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
- Q2：W8A8 QAT simulation；需另外取得 GPU 長訓練授權。

Conv weights 採 per-channel symmetric INT8，activations 採 per-tensor affine INT8。custom BinaryQK/PWL arithmetic 排除並列明。所有結果標示 `simulation_only=true`；CPU 現階段只驗證 plumbing 與 regression，不宣稱正式部署 latency。

Fake quant 仍在浮點 tensor 上模擬 quantize-dequantize；未完成 backend convert、operator coverage、
accumulator／requantization／saturation 與目標硬體驗證前，不得稱為真實 INT8 deployment 或 Bit-True。

## 10. Phase Gates 與交付

### Phase A：現在執行

- 規格、中文 README、正式 YAML/schema、資料工具與 manifests；
- handoff validator、候選 scope resolver、state-dict checkpoint contract；
- CPU-only config/data/graph/forward/loss/backward/freeze/reload/dry-run tests；
- 64/128 fixture forward 與一次 640 geometry test；
- 狀態完成後標示 `ready_for_upstream_handoff`。

### Phase B：等待 yolo_combine winner

- 驗收 handoff、建立 C0-Handoff、實際 graph audit；
- 解析真正候選矩陣並在 CPU 驗證；GPU smoke 仍需使用者授權；
- 完成後標示 `ready_for_formal_training`。

### Phase C：使用者授權正式訓練後

- C0-Control、C1、C2、C3 Float seeds；
- 完整指標、成本、Pareto 與使用者選擇；
- 再執行核准候選的 Q0/Q1/Q2。

本輪不執行正式長訓練。成果必須輸出 `REPORT.md`、CSV/JSON、`selection.json`、profiles、figures、lineage、effective configs 與中文工作紀錄。

## 修訂歷史

| 版本 | 日期 | 內容 |
|---|---|---|
| 1.0.0 | 2026-08-18 | 建立舊 Detect/Pose/量化規格。 |
| 1.1.0 | 2026-08-18 | 中文化與 Pose opt-in。 |
| 1.2.0 | 2026-08-19 | 單檔 training YAML 與設定檢查。 |
| 2.0.0 | 2026-08-22 | 改為承接 `yolo_combine` winner；動態解析 shared/routed/partial regions；C0-Handoff/C0-Control 分離；BBAT5 v1 衍生資料；measurement-first 選擇與逐候選量化資格；CPU-only Phase A。 |
| 2.0.1 | 2026-08-22 | 依 Ultralytics 8.4.90 官方原始碼修正 extension resume：只接受未 strip continuation state；補充 fraction/cache、Pose-only best／combined fitness、uniform OKS sigma、Micro F1 與 fake-quant 邊界。BBAT5 v1 保留建立時的 2.0.0 lineage。 |
| 2.0.2 | 2026-08-22 | 依使用者明確授權增加完整 GitHub dataset snapshot；影像與 labels 物化於獨立 `github-dataset/`，保留 bbat5-v1 既有 split／patch lineage，並持續禁止權重、checkpoint、cache 與 runs。 |
| 2.0.3 | 2026-08-23 | 依使用者明確授權發布 `original/` 的 raw Pose、歷史 Detect 與 canonical lineage，並將 `/home/uxin/yolo/original/pose/` 定為唯一 BBAT5 資料資產庫；移除 2.0.2 architecture_2 重複 snapshot 與正式匯出 CLI，禁止 artifacts 保存資料副本；Git tree 排除權重、cache、run 與重複超限 zip，canonical images 使用相對 symlink，正式訓練入口固定為 bbat5-v1 registry。 |
