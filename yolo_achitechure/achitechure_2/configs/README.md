# configs：正式設定導覽

這個目錄保存 architecture_2 的所有正式設定。若不知道要選哪個檔案，先看
[catalog.yaml](catalog.yaml)，再依本文件的表格選一份 architecture YAML 與一份 training YAML。

## 最重要的規則

- 一個訓練階段只需要一份 `configs/training/...yaml`。
- 不需要，也不允許使用者自行合併 canonical、overlay 或 stage fragment。
- `train --config ...` 是推薦入口。
- training YAML 是 project launcher 設定，不可直接傳給 `YOLO(yaml)`。
- dataset YAML 可以作為 Ultralytics 的 `data=...`。
- architecture candidate YAML 必須交給 achitechure_2 builder graft，不能獨立載入。
- 所有正式檔案都由 [catalog.yaml](catalog.yaml) 登錄，`config-check` 會拒絕未登錄或缺少的檔案。

## 目錄

```text
configs/
├── README.md
├── catalog.yaml
├── candidates/
├── training/
│   ├── detect/
│   └── pose/
├── data/
├── quant/
├── fusion/
└── schema/
```

## catalog.yaml

[catalog.yaml](catalog.yaml) 是機器可讀的唯一目錄，包含：

- `architectures`：候選 ID 到 YAML 路徑。
- `training`：config ID 到完整 training YAML。
- `datasets`：dataset 名稱到 YAML。
- `quantization`：量化設定。
- `fusion_template`：未來融合範本。
- `schemas`：各類 JSON Schema。

新增或改名正式 YAML 時必須更新 catalog。

## 架構候選

| ID | 檔案 | 目標 layers | factors | 狀態 |
|---|---|---|---|---|
| C0 | [c0.yaml](candidates/c0.yaml) | 無 | `e=.5, n=2, 3x3_3x3, rep=false` | required |
| C1 | [c1-e0375.yaml](candidates/c1-e0375.yaml) | 6/8/13/19 | `e=.375` | required |
| C2 | [c2-n1.yaml](candidates/c2-n1.yaml) | 6/8/13/19 | `inner_n=1` | required |
| C3 | [c3-mixed.yaml](candidates/c3-mixed.yaml) | 6/8/13/19 | `1x1_3x3` | required |
| C3-P5 | [c3-p5-only.yaml](candidates/c3-p5-only.yaml) | 8 | `1x1_3x3` | conditional |
| R1 | [r1-rep.yaml](candidates/r1-rep.yaml) | 6/8/13/19 | `inner_n=1, rep=true` | conditional |

每份候選 YAML 的主要欄位：

- `title_zh`／`summary_zh`：人類可讀用途。
- `status`：required 或 conditional。
- `target_layers`：實際改動位置。
- `protected_layers`：禁止修改的位置。
- `factors`：本候選唯一架構變因。
- `inherited`：MASF 與 attention 的凍結契約。
- `heads`：Detect/Pose26 inputs 與 strides。
- `conditions`：條件候選的 base 與 trigger。
- `architecture_delta`：改變、不變與預期效果。
- `standalone_loadable=false`：不可直接 `YOLO(yaml)`。

## 完整 training YAML

### Detect

| config ID | 檔案 | stage |
|---|---|---|
| detect-d0-smoke | [d0-smoke.yaml](training/detect/d0-smoke.yaml) | D0 |
| detect-d1-main | [d1-main.yaml](training/detect/d1-main.yaml) | D1 |
| detect-d2-extension | [d2-extension.yaml](training/detect/d2-extension.yaml) | D2 |
| detect-q2-qat | [q2-qat.yaml](training/detect/q2-qat.yaml) | Q2 |

### Pose

| config ID | 檔案 | stage |
|---|---|---|
| pose-p0-smoke | [p0-smoke.yaml](training/pose/p0-smoke.yaml) | P0 |
| pose-p1-head-only | [p1-head-only.yaml](training/pose/p1-head-only.yaml) | P1 |
| pose-p2-neck-head | [p2-neck-head.yaml](training/pose/p2-neck-head.yaml) | P2 |
| pose-p3-full | [p3-full.yaml](training/pose/p3-full.yaml) | P3 |
| pose-p4-extension | [p4-extension.yaml](training/pose/p4-extension.yaml) | P4 |

每份正式 training YAML 的 top-level interface：

| 欄位 | 意義 |
|---|---|
| `config_id` | catalog 使用的穩定 ID |
| `title_zh`／`summary_zh` | 中文用途 |
| `task`／`stage` | Detect/Pose 與 D0–P4/Q2 |
| `status` | required、optional 或 conditional |
| `candidate_policy` | 可用候選與 trigger |
| `execution` | launcher、`--execute`、Pose opt-in 與是否可直接交給 Ultralytics |
| `dataset` | dataset YAML 路徑 |
| `transition` | fresh/resume、checkpoint role 與 gate |
| `ranking` | 是否參與架構排名、研究指標與 backend |
| `frozen_modules` | 全階段永久凍結 roots |
| `derived_training` | 必須由 parent 推導的欄位；包含 Q2 lr0 與 D2/P4 resume checkpoint path |
| `training` | 完整送入 Ultralytics 的 training arguments |

`training:` 是完整設定，不依賴其他 YAML。Launcher 只會加入 run-specific `project/name`、解析 dataset
絕對路徑，並在 resume 階段加入 `resume`。Q2 `lr0=0.000038` 已明確寫入 YAML；launcher 只驗證它等於
parent architecture LR 的 0.1 倍，不再偷偷改值。

## 常用可調參數

所有參數都直接放在每份 YAML 的 `training:` 區塊。以下是最常調整的欄位：

| YAML 鍵名 | 用途 | 目前正式值 | 注意事項 |
|---|---|---|---|
| `batch` | physical batch size | 16 | Ultralytics 正式鍵名是 `batch`，不是 `batch_size` |
| `imgsz` | 輸入影像尺寸 | 640 | 改動會影響 VRAM、速度與精度 |
| `fraction` | 使用 train dataset 的比例 | 1.0 | 範圍 0–1；例如 0.1 只用 10% 資料 |
| `scale` | 隨機縮放 augmentation 幅度 | Detect 0.95／Pose 0.5 | 不是模型 scale；會影響 augmentation |
| `cache` | dataset cache | false | 可設 false、ram 或 disk；需確認記憶體／磁碟 |
| `device` | 訓練裝置 | "0" | 可用 GPU index、CPU 或多 GPU 格式 |
| `workers` | dataloader workers | 8 | Windows 或低 RAM 環境可能需要降低 |
| `epochs` | 最大 epoch | 依 stage | D2/P4 是總 epoch，不是額外 epoch |
| `patience` | early stopping | 依 stage | 0 代表停用需先確認本機語意 |
| `lr0` | 初始 learning rate | 依 task | Q2 明確為 parent LR 的 0.1 倍 |
| `lrf` | 最終 LR factor | Detect 0.882／Pose 0.01 | 與 scheduler 一起作用 |
| `optimizer` | optimizer | MuSGD | 正式比較禁止單一候選改 optimizer |
| `weight_decay` | weight decay | 依 task | 會受 `nbs` 與 accumulation 影響 |
| `nbs` | nominal batch size | 64 | physical batch 仍由 `batch` 決定 |
| `amp` | mixed precision | true | 關閉會增加 VRAM 與時間 |
| `deterministic` | deterministic mode | true | 正式比較必須保持一致 |
| `resume` | 是否 resume | D2/P4 true | launcher 會把 true 解析成指定的 last.pt path |
| `freeze` | freeze layers | 依 Pose stage | P1 為 0–22，P2 為 11 |
| `multi_scale` | multi-scale training range | 0.0 | 0.0 代表停用 |
| `compile` | torch compile | false | 開啟前必須確認 custom graph 相容 |
| `profile` | 訓練時 profile | false | 正式 latency 仍使用獨立 profile 流程 |
| `rect` | rectangular batches | false | 改動可能影響 batch geometry |
| `single_cls` | 合併為單類別 | false | 本案正式資料不可任意開啟 |
| `mosaic`／`mixup`／`copy_paste` | augmentation | 依 task | 同一 ablation 必須完全一致 |
| `translate`／`scale`／`fliplr` | 幾何 augmentation | 依 task | Pose `fliplr=0` 是語意安全要求 |
| `val`／`split` | 是否驗證與 split | true／val | 目前不建立 Pose test split |
| `iou`／`max_det`／`conf` | 驗證／推論門檻 | 0.7／300／null | 正式比較必須一致 |
| `save`／`save_period` | checkpoint 保存 | true／-1 | best/last 仍會保存 |

其他已顯式列出的欄位包括 `time`、`pretrained`、`verbose`、`cos_lr`、`close_mosaic`、
`dropout`、`save_json`、`dnn`、`classes` 與全部 loss/augmentation 參數。

### 怎麼調整

1. 打開要使用的完整 YAML，例如 `training/detect/d1-main.yaml`。
2. 在 `training:` 下修改正式 Ultralytics 鍵名。
3. 執行 `config-check`，不支援的鍵或型別會直接失敗。
4. 正式 ablation 若修改共用 learning/augmentation 欄位，必須同步同 task 的相關 YAML 並升版 spec。
5. `device/workers/cache/model/name/project` 可依機器或輸出位置調整，不列為架構學習變因。

`config-check` 會阻止 D0/D1/D2 或 P0–P4 在共同參數上意外漂移。這代表參數確實可在 YAML 中調整，
但正式比較不能只替單一候選偷偷改設定。

## Dataset

- [coco2017.yaml](data/coco2017.yaml)：Detect D0–D2/Q2。
- [pose-grouped.yaml](data/pose-grouped.yaml)：Pose P0–P4。

`project_metadata` 提供中文用途與 task；其餘 `path/train/val/names/kpt_shape` 是 Ultralytics dataset
欄位。Pose dataset 的 `test: null` 是刻意的，因為來源沒有 test split。

## Quantization

[w8a8-simulation.yaml](quant/w8a8-simulation.yaml) 定義：

- targets：C0、C_best
- weight：per-channel symmetric INT8
- activation：per-tensor affine INT8
- observers：前三個 QAT epochs 更新
- `simulation_only=true`

## Fusion

[source-pair.template.yaml](fusion/source-pair.template.yaml) 目前只是 future-disabled 範本。它保存來源 A/B
checkpoint、architecture、head contract 與 hashes。`fusion_mode`、`switch_policy` 必須保持 null，直到
使用者選定實驗語意並升版 spec。

## Schema

所有 schema 都採 JSON Schema Draft 2020-12：

- [catalog.schema.yaml](schema/catalog.schema.yaml)
- [candidate.schema.yaml](schema/candidate.schema.yaml)
- [training.schema.yaml](schema/training.schema.yaml)
- [dataset.schema.yaml](schema/dataset.schema.yaml)
- [quantization.schema.yaml](schema/quantization.schema.yaml)
- [fusion.schema.yaml](schema/fusion.schema.yaml)

本機沒有額外安裝 `jsonschema` 套件；專案的 `config-check` 仍會以 fail-closed 程式驗證必要欄位、
精確矩陣、Ultralytics 支援與跨檔公平性。這些標準 schema 可供 IDE 或外部 validator 使用。

## 推薦命令

```bash
# 檢查所有設定
$PY -m achitechure_2 config-check

# 預覽 Detect D1
$PY -m achitechure_2 train \
  --config configs/training/detect/d1-main.yaml \
  --candidate C0 \
  --checkpoint artifacts/candidates/c0/float-parent.pt \
  --run-id c0-d1

# 預覽 Pose P1（不執行）
$PY -m achitechure_2 train \
  --config configs/training/pose/p1-head-only.yaml \
  --candidate C0 \
  --checkpoint /path/to/pose-graft.pt \
  --run-id pose-c0-p1
```

## 修改設定

1. 先改 `EXPERIMENT_SPEC.md` 並升版。
2. 修改對應完整 YAML。
3. 共用學習欄位要同步更新同 task 的所有階段。
4. 更新 spec hash。
5. 更新 catalog。
6. 執行 `config-check` 與 pytest。

如果只修改一份 YAML 的 batch、optimizer、augmentation 或一般 LR，跨檔公平性檢查會中止。
