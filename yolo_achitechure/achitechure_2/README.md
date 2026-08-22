# architecture_2：融合 YOLO26m 的 C0–C3 架構簡化評估

這個資料夾不是重新做一次融合，也不是單獨訓練一個普通 Detect/Pose 模型。它的工作是：

1. 等待本機 `/home/uxin/yolo/yolo_combine/` 完成「COCO80 Detect + BBAT5 Pose」融合實驗並選出 winner。
2. 以不可變 handoff 接收該 winner。
3. 在 winner 的可修改 C3k2 區域上，公平比較 C0、C1、C2、C3。
4. 完整呈現精度與成本，最後由你決定 C_best、是否跑 Pose、哪些候選進量化，以及是否新增組合實驗。

融合方式尚未由 yolo_combine 決定，可能是 shared、routed 或 partial-shared。因此本專案不固定 layer
編號；真正修改位置由 handoff 的 candidate_regions 決定。

## 現在完成到哪裡

目前是 Phase A，狀態為 ready_for_upstream_handoff：

- [EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md) 已是唯一正式規格，版本 2.0.2。
- C0/C1/C2/C3 YAML、training templates、dataset YAML、quantization YAML 與 schemas 已建立。
- handoff、動態候選、權重轉移、state-dict checkpoint lineage 已實作。
- detect／pose／both、loss、gradient、freeze、BN buffer、640 geometry 與 reload 已在 CPU fixture 驗證。
- BBAT5 v1 已實際重建並驗證：6,647 張、零 group leakage、4 個負座標修補。
- GPU 仍未獲授權；沒有執行正式訓練、CUDA smoke、GPU latency/VRAM 或 QAT。
- 尚未收到 yolo_combine winner，所以目前沒有正式 C0–C3 Float 結果，也沒有 C_best。
- Pose 正式訓練與 validation 預設不執行，仍由你決定。

快速查看狀態：

    export PYTHONPATH="$PWD/src"
    PY=/home/uxin/yolo/.venv/bin/python
    $PY -m achitechure_2 status
    $PY -m achitechure_2 config-check
    env CUDA_VISIBLE_DEVICES='' $PY -m pytest -q

## 規格優先順序

1. [EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md)：唯一 authoritative spec。
2. [configs/catalog.yaml](configs/catalog.yaml)：正式 YAML 的機器可讀索引。
3. [TRAINING_GUIDE.md](TRAINING_GUIDE.md)：訓練與公平性理由。
4. [RUNBOOK.md](RUNBOOK.md)：已驗證的實際命令。
5. plan.pdf、[YOLO26_Codex_Research_Protocol.md](YOLO26_Codex_Research_Protocol.md)：歷史參考。

[plan.md](plan.md) 只指向正式規格，不複製第二套規則。規格變更要先升版並更新 revision history；
舊實驗永遠保留當時的 spec hash。

## 任務語意

| 路線 | 類別／輸出 | 角色 |
|---|---|---|
| Detect 主線 | COCO80，class 0 是 person | 正式 Detect 訓練與評估；應用主要關注人 |
| Pose 主線 | ball=0、bat=1，kpt_shape=[2,3] | 正式 Pose 任務，但是否執行由你決定 |
| BBAT5 Detect | ball=0、bat=1 | 配對診斷 view，不取代 COCO80，也不能單獨決定 C_best |

所以「COCO80 Detect head 可以保留」和「BBAT5 有 2-class Detect」並不衝突：它們是兩個不同資料
與用途。融合 winner 的公開介面必須支援 model(images, tasks=detect|pose|both)。

## 專案邊界

    yolo_combine
      └─ 選出融合 winner（shared / routed / partial-shared 尚未確定）
           └─ 不可變 handoff revision
                └─ architecture_2
                     ├─ C0-Handoff：winner 原樣
                     ├─ C0-Control：與候選相同的恢復訓練預算
                     ├─ C1/C2/C3：每次只改一個因素
                     ├─ CPU 契約驗證
                     └─ 未來 Float 結果 → 你決定 C_best／Pose／量化／新候選

yolo_combine 負責雙權重、雙架構、路由或 shared fusion 的實驗與 winner 選擇。本目錄不重做融合，
也不提前假設最後是哪一種。

## C0、C1、C2、C3 在做什麼

| ID | 唯一變更 | 直覺 | 可能代價 |
|---|---|---|---|
| C0 | 不修改 C3k2 | winner reference | Params／MAC 不下降 |
| C1 | e: 0.5 → 0.375 | 縮窄 hidden channels，通常直接減少參數與運算 | feature capacity 可能下降 |
| C2 | inner_n: 2 → 1 | 每個 C3k2 少一個 inner bottleneck | 深度與表徵能力可能下降 |
| C3 | 3x3_3x3 → 1x1_3x3 | 第一個 3×3 改成 1×1，降低空間卷積成本 | receptive field／小物件特徵可能受影響 |

共同規則：

- C1/C2/C3 每次只改一個因素，第一輪禁止組合。
- Detect/Pose heads、融合 topology、inherited MASF／BinaryQK／PWL attention 都受保護。
- shared winner 解析成 C0/C1/C2/C3。
- routed winner 解析成 C0/D-C1…D-C3/P-C1…P-C3。
- partial-shared winner 依實際區域解析成 S-、D-、P- 候選。
- C3-P5、R1、e=0.25 或因素組合都不是本輪候選；看完 Float 結果後再由你決定。

候選 YAML 在 [configs/candidates](configs/candidates)，它們是 graft contract，且刻意標記
standalone_loadable: false；不能直接交給共享 YOLO(yaml) parser。

## 資料分割是什麼、在哪裡

### 原始資料（唯讀）

- BBAT5 Detect：/home/uxin/yolo/original/pose/detect_dataset/
- BBAT5 Pose：/home/uxin/yolo/original/pose/dataset/
- COCO2017：/home/uxin/yolo/coco2017/

### 衍生資料（已建立）

實際可訓練資料在 /home/uxin/yolo/original/pose/derived/bbat5-v1/。

| 項目 | 結果 |
|---|---:|
| 影像 | 6,647 |
| .rf. 前 prefix 群組 | 2,578 |
| formal train / val | 5,964 / 683 |
| search train / val（只取 formal train） | 5,364 / 600 |
| train/val group leakage | 0 |
| 修補的負座標 | 4 |
| 與 COCO train 重疊且固定留在 train 的 groups | 331 |
| test split | 不存在，也未建立 |

Pose labels 是唯一權威。衍生 Detect label 是每列修補後的前五欄；原始 Detect labels 只做
一對一 audit。影像使用 symlink，labels 只寫進衍生版本，原始兩個目錄沒有被修改。

bbat5-v1 建立於 spec 2.0.0，因此五份 manifest 保留當時的
75db239262de75998171a05a85c8755dea10c2bc76920dd2aabe8ff0dabb7a3b，不回溯改成目前 2.0.2。
目前 validator 同時接受這個已知歷史 lineage 與新建資料的 current lineage，但不允許五份
manifest 混用版本。

衍生目錄內有獨立中文 README、四份 Ultralytics dataset YAML 與五份 manifests。Git 保存兩層：

1. 可稽核 metadata：

- [BBAT5 metadata README](artifacts/datasets/bbat5-v1/README.md)
- [split manifest](artifacts/datasets/bbat5-v1/manifests/split-manifest.json)
- [patch manifest](artifacts/datasets/bbat5-v1/manifests/patch-manifest.json)
- [source audit](artifacts/datasets/bbat5-v1/manifests/source-audit-manifest.json)
- [COCO exclusion](artifacts/datasets/bbat5-v1/manifests/coco-exclusion-manifest.json)

2. 使用者明確核准的[完整 GitHub dataset snapshot](artifacts/datasets/bbat5-v1/github-dataset/README.md)：
   物化 Pose／Detect 影像與 labels，提供 portable YAML/splits，沒有 symlink、絕對 split path、test、
   cache、checkpoint、weight 或 run。

本機 canonical v1 仍可用 prepare-pose-data 重建且不可覆寫；GitHub snapshot 只改變儲存形式，
不改變既有 group assignment、四筆 patch 或 2.0.0 資料 lineage。

正式可讀 search wrappers 在
[BBAT5 Pose search YAML](configs/data/bbat5-pose-search.yaml) 與
[BBAT5 Detect search YAML](configs/data/bbat5-detect-search.yaml)；兩者只引用 formal train 內的
search lists。

    # 只做唯讀規劃
    $PY -m achitechure_2 prepare-pose-data

    # v1 尚不存在時才執行
    $PY -m achitechure_2 prepare-pose-data --execute

    # 驗證已建立版本
    $PY -m achitechure_2 validate-pose-data

    # 預覽／物化／驗證完整 GitHub snapshot；都不會啟動 Pose
    $PY -m achitechure_2 export-github-dataset
    $PY -m achitechure_2 export-github-dataset --execute
    $PY -m achitechure_2 validate-github-dataset

這些命令只處理資料，不會啟動 Pose 模型或訓練。

## YAML 設定怎麼看

所有正式檔案都列在 [configs/catalog.yaml](configs/catalog.yaml)。詳細欄位導覽見
[configs/README.md](configs/README.md)。

### Training templates

| 檔案 | 用途 | 現在能否執行 |
|---|---|---|
| [cpu-smoke.yaml](configs/training/cpu-smoke.yaml) | CPU 結構／數值驗證 | 可以 |
| [float-main.yaml](configs/training/float-main.yaml) | C0-Control 與 C1–C3 公平 Float 比較 | 等 handoff、Pose 選擇與 GPU 授權 |
| [float-extension.yaml](configs/training/float-extension.yaml) | late gate 後由未 strip continuation state resume | 等前一階段與完整 state |
| [quant-qat.yaml](configs/training/quant-qat.yaml) | eligible candidate 的 Q2 | 等 Float 決策與 GPU 授權 |

每份 template 都是一個完整、獨立的設定，不需要合併 overlay。winner 尚未交付前不能猜的值會明確寫成：

    batch:
      source: handoff
      value: null

這不是漏填，而是 fail-closed。收到 handoff 後，effective-config 會產生所有值都已解析的平面
effective YAML。

### 常用可調欄位

| 欄位 | 位置 | 意義 |
|---|---|---|
| batch size | adjustable.batch.value | 正式 Ultralytics 鍵名是 batch，不是 batch_size |
| 資料比例 | adjustable.fraction.value | 1.0 是完整 train；小於 1 只截取排序後 train 前段 |
| augmentation scale | adjustable.scale.value | 幾何縮放幅度，不是 model scale |
| cache | adjustable.cache.value | false、ram 或 disk |
| 影像尺寸 | adjustable.imgsz.value | 通常由 winner recipe 繼承 |
| 裝置／workers | adjustable.device/workers.value | 可依機器調整 |
| epochs／patience | adjustable.epochs/patience.value | 正式預算由 handoff 繼承 |

model_scale: m 是 YOLO26m 模型級別，與 augmentation 的 scale 完全不同。

fraction 不是隨機、分層或 grouped sampling，也不能取代 BBAT5 的 search YAML。cache=true／ram
可能破壞完全 deterministic；cache=disk 會在影像旁寫檔，只能使用獲授權的可寫衍生資料。

正式 ablation 的 learning fields 必須對所有候選一致。name/project/device/workers/cache 可在
runtime 層覆寫；若要改 batch、fraction、scale、LR 或 optimizer，應修改正式 YAML／spec revision，
不能只對某個候選用 CLI 偷改。

## Handoff 與候選流程

handoff 必須包含 winner checkpoint、state-dict builder、architecture/training/dataset/selection
hashes、fresh-process reload 證據、完整模型語意、Candidate Regions、protected/frozen paths。
範例見 [handoff-manifest.example.json](handoff-manifest.example.json)。

    # 只看 metadata，不會接受
    $PY -m achitechure_2 inspect-handoff --manifest /path/handoff.json

    # 必須真的 materialize graph 才能寫 accepted.json
    $PY -m achitechure_2 accept-handoff --manifest /path/handoff.json --model-loader yolo_combine.some_module:load_winner

    # 看 winner 實際解析出哪些候選
    $PY -m achitechure_2 resolve-candidates --manifest /path/handoff.json

    # 對指定 resolved candidate 做 CPU forward/backward/freeze/reload
    $PY -m achitechure_2 cpu-dry-run --manifest /path/handoff.json --model-loader yolo_combine.some_module:load_winner --candidate C0

沒有正式 handoff 時不能宣稱候選已建立，也不會猜測真正 layer paths。

## Pose 是否執行由你決定

目前允許：

- 建立／驗證資料。
- 讀取 Pose YAML。
- 用 fixture 或 handoff model 做 CPU interface、loss、gradient smoke。

目前不會自動做：

- Pose 長訓練。
- 正式 Pose validation。
- 因缺少 Pose 結果而偷偷只用 Detect 選 C_best。

將來若你選擇不跑 Pose，可以先得到 Detect 與成本成果，但報告必須標示「融合模型正式排名尚不完整」。
若你選擇跑 Pose，C0-Control 與所有候選必須使用相同 Pose 資料、head seed、task ratio、optimizer
steps 與 validation events。

## 子訓練、搜尋與 validation

可以做小規模子訓練與超參數搜尋，但現在沒有 winner／GPU 授權，因此 Phase A 只執行 CPU smoke。
進入正式階段後：

- BBAT5 搜尋只用 pose-search／detect-search YAML；它們完全位於 formal train 內。
- BBAT5 搜尋結果若用來決定設定，必須形成一份共用 recipe，再鎖定給 C0-Control 與所有候選；
  禁止每個候選各自調參。
- adjustable.fraction 小於 1 可作快速 plumbing，但不是 grouped search，也不能產生正式排名。
- formal val 不參與搜尋；鎖定設定後才用相同 validation events 比較候選。
- 是否把 Pose 納入搜尋與正式 validation 仍由你 opt-in；不會因為資料已準備好就自動執行。
- COCO train-only search contract 尚未建立，所以目前不能宣稱已具備完整雙任務融合超參數搜尋。

## AP、AP50、mAP 與 F1

- AP50：單一 IoU（box）或 OKS（keypoint）= 0.50 的 Average Precision。
- AP50-95：0.50 到 0.95 多個 threshold 的 AP 平均；通常更嚴格。
- mAP：對多個 classes 的 AP 再平均。本案會明確寫 mAP50 或 mAP50-95，不只寫模糊的 mAP。
- Macro F1：ball、bat 的 F1 未加權平均，是主要 F1，可避免多數類別掩蓋另一類。
- Micro F1：先合併所有 TP/FP/FN 再計算，保留整體應用表現。

結果必須分開記錄 COCO80 overall、COCO person、BBAT5 Pose box/keypoint、ball/bat per-class、
Precision、Recall、per-class F1、Macro F1、Micro F1 與 confidence threshold。F1 threshold 只用
C0-Control search-val 決定一次，之後所有候選固定相同值。

Stock Ultralytics Pose best.pt 依 Pose mAP50-95 + Box mAP50-95 combined fitness 選擇。本案若執行
Pose，必須另外保存 pose-only mAP50-95 best，並把 combined fitness 分欄報告。BBAT5 的兩點
keypoint 目前使用套件均勻 OKS sigma，不能宣稱已針對棒球校準。

第一輪不做自動 PASS/REJECT，也不自動選 C_best。舊 0.005/0.008 drop 只作描述性敏感度。完整
Float 與 Pareto 報告產出後，狀態固定是 c_best: null、selection_status: pending_user_decision。

## 量化

[w8a8-simulation.yaml](configs/quant/w8a8-simulation.yaml) 定義 Q0/Q1/Q2：

- C0：預設可做 Q0/Q1。
- C1/C2/C3 或 resolved variants：先等你看完 Float 結果後核准。
- Q2：任何候選都還需要額外 GPU 長訓練授權。
- standard Conv 使用 W8A8 fake quant；custom BinaryQK/PWL attention 排除。
- 全部標示 simulation_only=true，CPU plumbing 結果不是部署 latency。
- Fake quant 仍輸出浮點 tensor；沒有 backend convert 與目標硬體驗證時，不稱為 Bit-True 或部署 INT8。

    $PY -m achitechure_2 quant-check --candidate C0 --stage Q1
    # C1 尚未核准時會 fail closed
    $PY -m achitechure_2 quant-check --candidate C1 --stage Q1

## 目錄

    achitechure_2/
    ├── AGENTS.md / CONTEXT.md      本目錄工作規範與領域詞彙
    ├── EXPERIMENT_SPEC.md          唯一正式規格
    ├── README.md                   全專案入口
    ├── RUNBOOK.md                  可複製命令
    ├── TRAINING_GUIDE.md           訓練、公平性與失敗處理
    ├── configs/                    candidates/training/data/quant/schema
    ├── src/achitechure_2/          handoff、builder、資料、CPU、checkpoint、metrics、quant
    ├── tests/                      CPU pytest
    ├── artifacts/datasets/bbat5-v1/ metadata + 獨立 github-dataset 完整 snapshot
    ├── results/                    空白結果與報告模板
    └── docs/                       ADR、官方參考、工作紀錄

## Git 會上傳與不會上傳的東西

會提交：

- 程式、測試、正式 YAML/schema。
- README、規格、ADR、工作紀錄。
- BBAT5 split／patch／source audit／COCO exclusion／rebuild manifests。
- 僅限 `artifacts/datasets/bbat5-v1/github-dataset/` 的完整物化影像、labels、portable YAML/splits
  與 publication manifest。

不會提交：

- 上述固定 snapshot 以外的影像、labels 與 cache。
- `.pt`／`.pth`／`.onnx`／engine、checkpoint、runs、GPU profiles。
- 未驗證的 accepted.json 或正式結果。

本次完整修改、資料處理、驗證與未解風險記錄在
[2026-08-22 architecture_2 Phase A 工作紀錄](../../docs/worklogs/2026-08-22-architecture-2-phase-a.md)。
完整資料發布另見
[2026-08-22 GitHub dataset 工作紀錄](../../docs/worklogs/2026-08-22-architecture-2-github-dataset.md)。

最後操作與驗證順序見 [RUNBOOK.md](RUNBOOK.md)。目前可以可靠宣稱的是 Phase A 與資料成果；
正式架構精度、C_best、Pose 改善與量化可接受性都尚待後續實驗與你的決策。
