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
    ├── training/
    │   ├── cpu-smoke.yaml
    │   ├── float-main.yaml
    │   ├── float-extension.yaml
    │   └── quant-qat.yaml
    ├── data/
    │   ├── coco2017.yaml
    │   ├── bbat5-pose.yaml
    │   └── bbat5-detect.yaml
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

候選檔描述「改什麼」，不描述「實際第幾層」。共同欄位：

| 欄位 | 意義 |
|---|---|
| task_contract | Detect 80 classes、Pose 2 classes 與 keypoint shape |
| parent_policy | 全部由同一 accepted yolo_combine winner 獨立建立 |
| target_policy | module paths 必須由 handoff Candidate Regions 提供 |
| scope_resolution | shared／routed／partial-shared 如何展開候選 |
| factors / baseline_factors | 唯一架構因素與 C0 基準 |
| protected_policy | heads、fusion topology 與 inherited modules 不得改動 |
| architecture_delta | 中文 changed/unchanged/expected effects |
| standalone_loadable | 永遠是 false；不可直接 YOLO(yaml) |

## Training templates

| config_id | 檔案 | 狀態 |
|---|---|---|
| cpu-smoke | [cpu-smoke.yaml](training/cpu-smoke.yaml) | Phase A，CPU-only |
| float-main | [float-main.yaml](training/float-main.yaml) | 等 handoff、Pose opt-in、GPU 授權 |
| float-extension | [float-extension.yaml](training/float-extension.yaml) | 等 late gate 與未 strip continuation checkpoint |
| quant-qat | [quant-qat.yaml](training/quant-qat.yaml) | 等 Float 決策、量化資格、GPU 授權 |

這四份都是完整 templates，不需要自行合併 canonical、overlay 或 stage fragment。它們刻意分成：

- recipe：optimizer、LR、loss、augmentation、freeze、seed、task ratio 與 optimizer steps。
- adjustable：batch、fraction、scale、cache、imgsz、device、workers、epochs、patience。
- execution：CPU/GPU、Pose opt-in、是否為正式訓練。
- routes：COCO Detect 與 BBAT5 Pose 各自在什麼資料上計 loss。
- validation：必存的 AP/mAP/F1 與固定 threshold 來源。
- transition：fresh 或真正 resume；extension 明確拒絕已 strip 的 stock last.pt／best.pt。
- lineage：effective YAML 與所有 hashes 的要求。

### source/value 是什麼

Winner 尚未交付前，本專案不能猜 upstream training recipe。每個值都保留來源：

    optimizer:
      source: handoff
      value: null

代表「handoff 一定要提供；現在 null 是刻意阻擋」。本地已確定的值會寫成：

    fraction:
      source: local
      value: 1.0

收到 handoff 後使用 effective-config 解析；若任何 null 沒有對應上游值就停止。輸出的 effective YAML
會是具體、可保存 hash 的單一設定，不再需要理解 source/value。

## 常用參數如何調

| 目的 | 正式鍵名 | Template 位置 | 允許值／提醒 |
|---|---|---|---|
| batch size | batch | adjustable.batch.value | 不要寫 batch_size |
| 使用資料比例 | fraction | adjustable.fraction.value | 0 < fraction ≤ 1；只截取排序後 train 前段 |
| 幾何縮放 augmentation | scale | adjustable.scale.value | 與 model_scale=m 無關 |
| dataset cache | cache | adjustable.cache.value | false、ram、disk |
| input size | imgsz | adjustable.imgsz.value | 會影響精度與記憶體 |
| 裝置 | device | adjustable.device.value | 現在 CPU smoke 固定 cpu |
| dataloader | workers | adjustable.workers.value | CPU smoke 是 0 |
| 訓練預算 | epochs / patience | adjustable 對應欄位 | 正式值由 winner recipe 繼承 |
| optimizer | optimizer | recipe.optimizer.value | 不得對單一候選不同 |
| 初始／最終 LR | lr0 / lrf | recipe 對應欄位 | 不得用 CLI 偷改 |
| nominal batch | nbs | recipe.nbs.value | 要和 physical batch 分開記錄 |
| task sampling | task_ratio | recipe.task_ratio.value | 要以 optimizer steps/effective samples 比較 |

name、project、device、workers、cache 是 runtime allowlist。batch、fraction、scale、LR、optimizer、
augmentation 或 loss 若要改，必須修改正式 YAML、更新 spec revision/hash，並讓所有候選共用；這是
為了防止單一候選得到不同訓練條件，不是限制 YAML 的可調性。

fraction 不是 grouped split；BBAT5 搜尋必須使用獨立的 search dataset YAML。cache=true／ram 可能
不是完全 deterministic，cache=disk 只允許寫入衍生資料，不能寫原始唯讀目錄。

解析範例：

    $PY -m achitechure_2 effective-config --manifest /path/handoff.json --template configs/training/float-main.yaml --output artifacts/effective-configs/float-main.yaml

這個命令只產生設定，不會開始訓練。

## Dataset YAML

| 檔案 | 角色 | 實際資料 |
|---|---|---|
| [coco2017.yaml](data/coco2017.yaml) | COCO80 Detect 主線 | /home/uxin/yolo/coco2017 |
| [bbat5-pose.yaml](data/bbat5-pose.yaml) | BBAT5 ball/bat Pose 主線 | derived/bbat5-v1/pose |
| [bbat5-detect.yaml](data/bbat5-detect.yaml) | 2-class paired 診斷 | derived/bbat5-v1/detect |
| [bbat5-pose-search.yaml](data/bbat5-pose-search.yaml) | formal-train-only Pose search | pose search lists |
| [bbat5-detect-search.yaml](data/bbat5-detect-search.yaml) | formal-train-only paired diagnostic | detect search lists |

BBAT5 Pose 與 Detect 共用同一 grouped assignment，test 明確為 null。衍生目錄另有四份乾淨、
可直接交給 Ultralytics 的 formal/search dataset YAML；Git 副本在
[artifacts/datasets/bbat5-v1/configs](../artifacts/datasets/bbat5-v1/configs)。

目前只有 BBAT5 有正式 train-only search config；COCO train-only search contract 尚未建立。因此
這兩份 YAML 可用於 F1 threshold 與獲准的 BBAT5 小規模研究，不能宣稱已完成雙任務融合 recipe 搜尋。

## Quantization YAML

[w8a8-simulation.yaml](quant/w8a8-simulation.yaml) 固定：

- Q0 fused FP32 reference。
- Q1 W8A8 PTQ simulation。
- Q2 W8A8 QAT simulation，必須有 GPU 授權。
- Conv weight per-channel symmetric INT8。
- activation per-tensor affine INT8。
- custom HardwareFriendlyAttention/BinaryQK/PWL roots 排除。
- simulation_only=true。

C0 預設 eligible；其他候選保持 pending_user_decision。

## Schemas

schema 皆採 JSON Schema Draft 2020-12，並保存 x-spec-version 與 x-spec-sha256：

- [catalog.schema.yaml](schema/catalog.schema.yaml)
- [candidate.schema.yaml](schema/candidate.schema.yaml)
- [training.schema.yaml](schema/training.schema.yaml)
- [dataset.schema.yaml](schema/dataset.schema.yaml)
- [quantization.schema.yaml](schema/quantization.schema.yaml)
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
