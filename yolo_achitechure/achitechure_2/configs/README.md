# configs：正式 YAML 導覽

[catalog.yaml](catalog.yaml) 是所有正式 YAML 的唯一機器可讀索引。任何未列入 catalog 的檔案都不是
正式實驗設定；任何未知欄位、spec hash 漂移或候選矩陣漂移都會被 config-check 拒絕。

## 目錄

    configs/
    ├── catalog.yaml
    ├── candidates/
    │   ├── c0.yaml
    │   ├── c1-e0375.yaml
    │   ├── c2-n1.yaml
    │   └── c3-mixed.yaml
    ├── runs/
    │   ├── full35-float-screen-20.yaml
    │   └── full35-c2-c3-auto-continuation.yaml
    ├── training/
    │   ├── cpu-smoke.yaml
    │   ├── float-screen-20.yaml
    │   ├── float-main.yaml
    │   ├── float-extension.yaml
    │   ├── quant-qat-lite.yaml
    │   └── quant-qat.yaml
    ├── data/
    │   ├── coco2017.yaml
    │   ├── coco2017-screen-20.yaml
    │   ├── bbat5-pose.yaml
    │   ├── bbat5-pose-screen-20.yaml
    │   ├── bbat5-detect-screen-20.yaml
    │   ├── bbat5-detect.yaml
    │   ├── bbat5-pose-search.yaml
    │   └── bbat5-detect-search.yaml
    ├── quant/w8a8-simulation.yaml
    └── schema/

本 revision 沒有 fusion template：融合實驗與 winner 選擇屬於 yolo_combine。本 revision 也沒有
C3-P5 或 R1 YAML；它們必須等使用者看完第一輪結果後才可能加入下一版 spec。

## Candidate YAML

| ID | 檔案 | 唯一因素 | 量化資格 |
|---|---|---|---|
| C0 | [c0.yaml](candidates/c0.yaml) | 不修改 | true |
| C1 | [c1-e0375.yaml](candidates/c1-e0375.yaml) | e 0.5 → 0.375 | pending_user_decision |
| C2 | [c2-n1.yaml](candidates/c2-n1.yaml) | inner_n 2 → 1 | pending_user_decision |
| C3 | [c3-mixed.yaml](candidates/c3-mixed.yaml) | 3x3_3x3 → 1x1_3x3 | pending_user_decision |

候選檔描述「改什麼」；實際 module path 必須由 handoff 驗證。本輪 Full35 J3 已固定解析為
`graph.model.6`、`graph.model.8`、`graph.model.13`、`graph.model.19`，四個位置都在 shared trunk。
共同欄位：

| 欄位 | 意義 |
|---|---|
| task_contract | Detect 80 classes、Pose 2 classes 與 keypoint shape |
| parent_policy | 全部由同一 immutable Full35 J3 EMA parent 獨立建立 |
| target_policy | module paths 由 accepted handoff Candidate Region 提供並驗證 |
| factors / baseline_factors | 唯一架構因素與 C0 基準 |
| protected_policy | Detect/Pose heads、shared topology、MASF 與 attention 不得改動 |
| architecture_delta | 中文 changed/unchanged/expected effects |
| standalone_loadable | 永遠是 false；不可直接 YOLO(yaml) |

## 正式 Run 與通用 templates

| config_id | 檔案 | 狀態 |
|---|---|---|
| full35-j3-float20-seed0 | [full35-float-screen-20.yaml](runs/full35-float-screen-20.yaml) | Pose=true；C0～C3訓練、profile與匯出已完成 |
| full35-j3-c2-c3-auto-seed0 | [full35-c2-c3-auto-continuation.yaml](runs/full35-c2-c3-auto-continuation.yaml) | `closed_rejected_by_user`；封存且不得執行full、PTQ或QAT-lite |
| cpu-smoke | [cpu-smoke.yaml](training/cpu-smoke.yaml) | 通用 CPU-only 契約；Full35 實測已通過 |
| float-screen-20 | [float-screen-20.yaml](training/float-screen-20.yaml) | 通用 Float20 template；已由上列 run 具體化 |
| float-main | [float-main.yaml](training/float-main.yaml) | 等使用者決策 |
| float-extension | [float-extension.yaml](training/float-extension.yaml) | 等 late gate 與未 strip continuation checkpoint |
| quant-qat-lite | [quant-qat-lite.yaml](training/quant-qat-lite.yaml) | 已建置但暫停，等候選重新核准／GPU |
| quant-qat | [quant-qat.yaml](training/quant-qat.yaml) | 等 Float 決策、量化資格、GPU 授權 |

Float20只讀取 `configs/runs/full35-float-screen-20.yaml`；封存狀態只讀取
`configs/runs/full35-c2-c3-auto-continuation.yaml`。兩者不需手動合併。`training/` 下的六份是
通用模板與後續階段契約，刻意分成：

- recipe：optimizer、LR、loss、augmentation、freeze、seed、task ratio 與 optimizer steps。
- adjustable：batch、fraction、scale、cache、imgsz、device、workers、epochs、patience。
- execution：CPU/GPU、Pose opt-in、是否為正式訓練。
- routes：COCO Detect 與 BBAT5 Pose 各自在什麼資料上計 loss。
- validation：必存的 AP/mAP/F1 與固定 threshold 來源。
- transition：fresh 或真正 resume；extension 明確拒絕已 strip 的 stock last.pt／best.pt。
- lineage：effective YAML 與所有 hashes 的要求。

### 為什麼 templates 仍有 source/value

`configs/training/` 是可供未來 handoff revision 重用的通用契約，因此必須保留值的來源：

    optimizer:
      source: handoff
      value: null

這表示解析 template 時 handoff 一定要提供該值，缺值就停止。本輪 Full35 handoff 已交付並驗收，
所以可執行數字已直接寫進 `configs/runs/full35-float-screen-20.yaml`；實際排隊時不需要理解或修改
`source/value`。`effective-config` 仍保留給未來完整 Float／extension template 使用。

## 常用參數如何調

Float20參數請改 [full35-float-screen-20.yaml](runs/full35-float-screen-20.yaml)；完整訓練、資格門檻、
PTQ與QAT-lite參數請改
[full35-c2-c3-auto-continuation.yaml](runs/full35-c2-c3-auto-continuation.yaml)。下面表格是Float20欄位；
兩份closed schema都會拒絕拼錯、未知或與本輪公平性契約衝突的欄位。

| 目的 | 正式 YAML 位置 | 目前值／提醒 |
|---|---|---|
| Detect logical batch | `training.batch.detect_logical` | 128；不是一次放進 GPU 的 physical batch |
| Detect physical batch | `training.batch.detect_physical_microbatch` | 32；C0 probe OOM 時全矩陣改用 fallback 16 |
| Pose train batch | `training.batch.pose_physical` | 16；只在 Pose=true 時使用 |
| validation batch | `training.batch.validation_detect/pose` | 兩者固定16，沿用 Full35 OOM 經驗 |
| 使用資料比例 | `training.fraction` | 固定1.0；20%已由 immutable manifest完成 |
| 幾何縮放 | `training.scale` 與 `training.augmentation.*.scale` | 0.5；不是 model scale |
| image cache | `training.cache` | false、true、ram；禁止 disk |
| input size | `training.imgsz` | 本輪固定640 |
| 裝置 | `training.device` | GPU 0；只由持鎖 queue 解鎖 |
| dataloader | `training.workers.detect/pose` | 4／8，可依主機 I/O 同步調整 |
| 訓練預算 | `training.epochs` | 20；四候選完全相同 |
| optimizer／nominal batch | `training.optimizer`／`training.nbs` | MuSGD／64 |
| LR groups | `training.learning_rates` | backbone、neck、heads 分開；MASF/attention 為0 |
| loss／augmentation | `training.loss_overrides`／`training.augmentation` | Detect 與 Pose 分開列全 |

Ultralytics 的正式鍵名是 `batch`，不是 `batch_size`。logical 128 由 microbatch 與 gradient
accumulation 實現；報告會同時保存 requested／effective physical batch。

`fraction` 不是隨機、分層或 grouped split。本輪20%樣本已鎖在 manifest，因此必須保持1.0；
BBAT5 其他搜尋只能使用既有 grouped search YAML，不可再次抽樣。

`cache` 控制 image cache；label index 無論如何都由 adapter 寫到子專案 `runtime-cache/`，不會寫進
COCO 或 immutable bbat5-v1。`disk` 不在正式 schema 中，避免 source-adjacent cache。

所有 learning fields 都必須寫入正式 YAML、保存新 hash，並同時套用 C0～C3。queue CLI 不接受
臨時覆寫 batch、fraction、scale、LR、optimizer、loss 或 augmentation；修改方法級契約時還必須先
升級 [EXPERIMENT_SPEC.md](../EXPERIMENT_SPEC.md)。

## Dataset YAML

| 檔案 | 角色 | 實際資料 |
|---|---|---|
| [coco2017.yaml](data/coco2017.yaml) | COCO80 Detect 主線 | /home/uxin/yolo/coco2017 |
| [coco2017-screen-20.yaml](data/coco2017-screen-20.yaml) | COCO80 train-only 20%初篩 | 固定 manifests |
| [bbat5-pose.yaml](data/bbat5-pose.yaml) | BBAT5 ball/bat Pose 主線 | derived/bbat5-v1/pose |
| [bbat5-detect.yaml](data/bbat5-detect.yaml) | 2-class paired 診斷 | derived/bbat5-v1/detect |
| [bbat5-pose-screen-20.yaml](data/bbat5-pose-screen-20.yaml) | grouped20% Pose 初篩 | 固定 manifest + search-val |
| [bbat5-detect-screen-20.yaml](data/bbat5-detect-screen-20.yaml) | grouped20% paired 診斷 | 與 Pose 同 assignment |
| [bbat5-pose-search.yaml](data/bbat5-pose-search.yaml) | formal-train-only Pose search | pose search lists |
| [bbat5-detect-search.yaml](data/bbat5-detect-search.yaml) | formal-train-only paired diagnostic | detect search lists |

BBAT5 Pose 與 Detect 共用同一 grouped assignment，test 明確為 null。衍生目錄另有四份乾淨、
可直接交給 Ultralytics 的 formal/search dataset YAML；Git 副本在
[artifacts/datasets/bbat5-v1/configs](../artifacts/datasets/bbat5-v1/configs)。

BBAT5 formal-train-only search YAML 繼續供 F1 threshold 與獲准的小規模研究。COCO 則新增專供
architecture-screen-20-v1 的 train-only 20%／5,000 search-val contract；它只作架構初篩，不升格為
一般超參數搜尋集，也不使用官方 val2017。

## Quantization YAML

[w8a8-simulation.yaml](quant/w8a8-simulation.yaml) 固定：

- Q0 fused FP32 reference。
- Q1 W8A8 PTQ simulation。
- Q2L W8A8 QAT-lite simulation：固定200 steps、observer前50 steps更新，只作相容性檢查。
- Q2 完整 W8A8 QAT simulation，必須另有 GPU 授權。
- Conv weight per-channel symmetric INT8。
- activation per-tensor affine INT8。
- custom HardwareFriendlyAttention/BinaryQK/PWL roots 排除。
- simulation_only=true。

C0維持全域預設eligible；本次run的C2/C3未通過四項精度下降≤0.008 gate，且已封存不採用，因此不會執行Q0/Q1/Q2L；C1與正式Q2亦未執行。

## Schemas

schema 皆採 JSON Schema Draft 2020-12，並保存 x-spec-version 與 x-spec-sha256：

- [catalog.schema.yaml](schema/catalog.schema.yaml)
- [candidate.schema.yaml](schema/candidate.schema.yaml)
- [training.schema.yaml](schema/training.schema.yaml)
- [dataset.schema.yaml](schema/dataset.schema.yaml)
- [quantization.schema.yaml](schema/quantization.schema.yaml)
- [screen-run.schema.yaml](schema/screen-run.schema.yaml)
- [continuation-run.schema.yaml](schema/continuation-run.schema.yaml)
- [handoff.schema.yaml](schema/handoff.schema.yaml)

本地 config-check 另外執行跨檔語意檢查，例如 C0–C3 唯一因素、COCO80/Pose2 類別、Pose opt-in、
量化資格、Ultralytics 8.4.90 與 forbidden files。

## 修改正式設定

1. 先修改 [EXPERIMENT_SPEC.md](../EXPERIMENT_SPEC.md)，升版並加入 revision history。
2. 更新相關 YAML 的 spec_version/spec_sha256。
3. 若新增正式檔，更新 catalog 與 schema。
4. 對所有候選同步公平性欄位。
5. 執行：

       $PY -m achitechure_2 config-check
       env CUDA_VISIBLE_DEVICES='' $PY -m pytest -q

不要複製舊訓練命令，也不要把 candidate YAML 當成可獨立 YOLO architecture YAML。
