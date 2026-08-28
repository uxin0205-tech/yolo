# architecture_2：融合 YOLO26m 的 C0–C3 架構簡化評估

> **封存決策（2026-08-28）**：C1～C3 的精度下降過大，使用者決定不採用本階段候選。
> C2/C3 full、PTQ 與 QAT-lite 永久不執行；本目錄只保留負結果、重建工具與 BBAT5 v1 資料血緣。
> 這不否定上游 Full35 C0，也不影響獨立的 `yolo_activation` 工作。

這個資料夾承接 `/home/uxin/yolo/yolo_combine/final/full35/` 已選定的 YOLO26m Full35 融合模型，不重新做融合選型。它只做四件事：

1. 驗證並鎖定 immutable Full35 J3 EMA handoff。
2. 從同一 parent 各自建立 C0、C1、C2、C3。
3. 先用固定20% train-only資料公平比較架構精度與成本。
4. 封存 C1～C3 負結果，並禁止本 revision 接續 C2/C3 完整訓練、PTQ 與 QAT-lite。

目前 Full35 已確定是 23 層 shared graph 加 Detect／Pose 雙 head；本輪修改路徑固定由 handoff 驗證為 layers 6、8、13、19。`yolo_combine` 仍負責融合與雙權重／架構切換，本目錄不重做那些工作。

## 現在完成到哪裡

本專案已封存：J3 handoff、CPU驗收與Pose=true的Float20訓練／profile／匯出均已完成；後續GPU工作永久關閉：

- [EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md) 是唯一正式規格，版本 2.3.0。
- Full35 J3 EMA checkpoint 已嚴格載入 1,238 tensors；model contract 是 Detect nc=80、Pose nc=2、kpt_shape=[2,3]。
- C0～C3 都已通過 CPU 640×640 Detect／Pose／both forward、finite loss/gradient、freeze 與 fresh reload。
- Float20的C0～C3各完成20 epochs；正式結果在 [Float20報告](results/full35-j3-float20-seed0/REPORT.md)。
- 正式執行設定是 [full35-float-screen-20.yaml](configs/runs/full35-float-screen-20.yaml)，batch、fraction、scale、cache、LR、loss、workers、OOM 與 queue 全部集中在同一檔。
- profiling的CPU/GPU動態cache裝置錯誤已修正，正式RTX 5090 profile與hash manifest已成功產生。
- [後續設定](configs/runs/full35-c2-c3-auto-continuation.yaml)目前是`closed_rejected_by_user`；程式入口會fail closed，不得排入queue。
- Float20保留`c_best=null`。C2是三個簡化候選中最佳者，但四項精度下降都超過0.008門檻，因此沒有候選依原gate取得full／PTQ／QAT-lite資格。
- Pose gate仍是`true`；這只保存本輪已執行的Pose語意，不代表已授權新的GPU工作。
- BBAT5 v1、固定20% assignments 與 leakage 驗證均已完成；runtime label cache 只寫在子專案 artifacts，不寫 canonical 資料。

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
| Pose 主線 | ball=0、bat=1，kpt_shape=[2,3] | 正式 Pose 任務；本輪已選擇執行 |
| BBAT5 Detect | ball=0、bat=1 | 配對診斷 view，不取代 COCO80，也不能單獨決定 C_best |

所以「COCO80 Detect head 可以保留」和「BBAT5 有 2-class Detect」並不衝突：它們是兩個不同資料
與用途。融合 winner 的公開介面必須支援 model(images, task=detect|pose|both)。

## 專案邊界

    yolo_combine/final/full35
      └─ accepted J3 shared-dual-head EMA parent（不可變）
           └─ architecture_2
                ├─ C0：結構不改、同預算 control
                ├─ C1：只縮 e
                ├─ C2：只減 inner_n
                ├─ C3：只換第一個 kernel
                └─ Float20報告（已完成）→ 本階段不採用並封存

`yolo_combine` 負責融合、雙權重與架構切換；`architecture_2` 只評估這四個共享 C3k2 候選，不實作或挑選融合方式。

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
- Full35 handoff 固定解析成 C0/C1/C2/C3；本目錄不再展開 routed 或 partial-shared 候選。
- C3-P5、R1、e=0.25 或因素組合都不是本輪候選；看完 Float 結果後再由你決定。

候選 YAML 在 [configs/candidates](configs/candidates)，它們是 graft contract，且刻意標記
standalone_loadable: false；不能直接交給共享 YOLO(yaml) parser。

## 資料分割是什麼、在哪裡

### 原始資料（唯讀來源，禁止作新訓練入口）

- BBAT5 Detect：/home/uxin/yolo/original/pose/detect_dataset/
- BBAT5 Pose：/home/uxin/yolo/original/pose/dataset/
- COCO2017：/home/uxin/yolo/coco2017/

### 衍生資料（已建立）

實際可訓練資料在 /home/uxin/yolo/original/pose/derived/bbat5-v1/。
全專案唯一 registry 是 /home/uxin/yolo/configs/datasets/bbat5-v1.yaml；`yolo_combine`
與未來其他子專案都必須引用同一版本，不得從原始目錄另建 split。

`bbat5-v1` 根目錄是 Pose + 二類 Detect 的版本容器，不是可直接傳給 Ultralytics 的
dataset YAML。Pose 使用 `configs/pose.yaml`，ball/bat 二類 Detect 診斷使用
`configs/detect.yaml`；C0–C3 的 COCO80 Detect 主線仍使用 `/home/uxin/yolo/coco2017.yaml`。

| 項目 | 結果 |
|---|---:|
| 影像 | 6,647 |
| .rf. 前 prefix 群組 | 2,578 |
| formal train / val | 5,964 / 683 |
| search train / val（只取 formal train） | 5,364 / 600 |
| architecture screen train（seed 0） | COCO 23,657；BBAT5 1,073 / 410 groups |
| architecture screen search-val | COCO train-only 5,000；BBAT5 600 |
| train/val group leakage | 0 |
| 修補的負座標 | 4 |
| 與 COCO train 重疊且固定留在 train 的 groups | 331 |
| test split | 不存在，也未建立 |

Pose labels 是唯一權威。衍生 Detect label 是每列修補後的前五欄；原始 Detect labels 只做
一對一 audit。影像使用 symlink，labels 只寫進衍生版本，原始兩個目錄沒有被修改。

bbat5-v1 建立於 spec 2.0.0，因此五份 manifest 保留當時的
75db239262de75998171a05a85c8755dea10c2bc76920dd2aabe8ff0dabb7a3b，不回溯改成目前 2.3.0。
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

本機 canonical v1 仍可用 prepare-pose-data 重建且不可覆寫；它是正式訓練權威。
GitHub snapshot 只作可攜發行，不是第二個訓練來源，也不改變既有 group assignment、
四筆 patch 或 2.0.0 資料 lineage。

正式可讀 search wrappers 在
[BBAT5 Pose search YAML](configs/data/bbat5-pose-search.yaml) 與
[BBAT5 Detect search YAML](configs/data/bbat5-detect-search.yaml)；兩者只引用 formal train 內的
search lists。固定20% View 另由 [screening README](artifacts/datasets/architecture-screen-20-v1/README.md)
與 [screening manifest](artifacts/datasets/architecture-screen-20-v1/screening-manifest.json) 定義；只保存清單，
不複製影像或標註。

Ultralytics 的 label index 由本專案 adapter 固定寫到
`artifacts/datasets/architecture-screen-20-v1/runtime-cache/`。這是可刪除、可重建且不提交 Git 的
runtime View；不會在 `/home/uxin/yolo/coco2017/` 或 canonical `bbat5-v1` 旁建立 `.cache`。

    # 只做唯讀規劃
    $PY -m achitechure_2 prepare-pose-data

    # v1 尚不存在時才執行
    $PY -m achitechure_2 prepare-pose-data --execute

    # 驗證已建立版本
    $PY -m achitechure_2 validate-pose-data

    # 預覽／建立／驗證固定20%篩選 manifests；不啟動訓練
    $PY -m achitechure_2 prepare-screening-data
    $PY -m achitechure_2 prepare-screening-data --execute
    $PY -m achitechure_2 validate-screening-data

    # 預覽／物化／驗證完整 GitHub snapshot；都不會啟動 Pose
    $PY -m achitechure_2 export-github-dataset
    $PY -m achitechure_2 export-github-dataset --execute
    $PY -m achitechure_2 validate-github-dataset

這些命令只處理資料，不會啟動 Pose 模型或訓練。

## YAML 設定怎麼看

所有正式檔案都列在 [configs/catalog.yaml](configs/catalog.yaml)。詳細欄位導覽見
[configs/README.md](configs/README.md)。

### Training templates

| 檔案 | 用途 | 現在狀態 |
|---|---|---|
| [full35-float-screen-20.yaml](configs/runs/full35-float-screen-20.yaml) | 本輪Float20正式設定 | C0～C3訓練、profile與匯出已完成 |
| [full35-c2-c3-auto-continuation.yaml](configs/runs/full35-c2-c3-auto-continuation.yaml) | C2/C3 full、PTQ、QAT-lite與最終選型 | `closed_rejected_by_user`；封存且禁止執行 |
| [cpu-smoke.yaml](configs/training/cpu-smoke.yaml) | 通用 CPU 結構／數值模板 | 可以 |
| [float-screen-20.yaml](configs/training/float-screen-20.yaml) | 通用 Float20 契約模板 | 已由上列 Full35 run 具體化 |
| [float-main.yaml](configs/training/float-main.yaml) | 後續完整 Float | 通用模板；本 revision 不執行 |
| [float-extension.yaml](configs/training/float-extension.yaml) | late gate resume | 通用模板；本 revision 不執行 |
| [quant-qat-lite.yaml](configs/training/quant-qat-lite.yaml) | Q2L 通用模板 | 通用模板；本 revision 不執行 |
| [quant-qat.yaml](configs/training/quant-qat.yaml) | 正式 Q2 | 等使用者另行核准 |

Float20參數只看`configs/runs/full35-float-screen-20.yaml`；後續門檻、100 epochs／patience20、校正batch與QAT-lite steps只看`configs/runs/full35-c2-c3-auto-continuation.yaml`。`configs/training/`保留通用模板；`source: handoff, value: null`不是本輪run漏填。

### 常用可調欄位

| 欄位 | 位置 | 意義 |
|---|---|---|
| batch size | training.batch | 正式 Ultralytics 鍵名是 batch，不是 batch_size |
| 資料比例 | training.fraction | 必須1.0；20%由 immutable manifest決定 |
| augmentation scale | training.scale | 幾何縮放幅度，不是 model scale |
| cache | training.cache | false、true 或 ram；禁止 disk |
| 影像尺寸 | training.imgsz | 本輪固定640 |
| 裝置／workers | training.device 與 training.workers | GPU 0；Detect 4、Pose 8 |
| epochs | training.epochs | 本輪固定20；四候選相同 |
| OOM microbatch | training.batch.detect_physical_microbatch | 先32；C0完整macro OOM才全矩陣改16 |

model_scale: m 是 YOLO26m 模型級別，與 augmentation 的 scale 完全不同。

fraction 必須保持1.0，因為20%已固定在 manifest；不可再次抽樣。`cache: false` 只控制 image cache，label index 仍由 adapter 寫到可重建 runtime-cache，正式 run 禁止 source-adjacent disk cache。

正式 ablation 的 learning fields 必須對所有候選一致。name/project/device/workers/cache 可在
runtime 層覆寫；若要改 batch、fraction、scale、LR 或 optimizer，應修改正式 YAML／spec revision，
不能只對某個候選用 CLI 偷改。

## Handoff 與候選流程

本輪正式內容見[Full35 J3 handoff README](handoffs/full35-j3-seed0/README.md)。每個handoff revision
必須包含winner checkpoint、state-dict builder、architecture/training/dataset/selection hashes、
fresh-process reload證據、完整模型語意、Candidate Regions、protected/frozen paths。
範例見 [handoff-manifest.example.json](handoff-manifest.example.json)。

    # 只看 metadata，不會接受
    $PY -m achitechure_2 inspect-handoff --manifest handoffs/full35-j3-seed0/manifest.json

    # 必須真的 materialize graph 才能寫 accepted.json
    $PY -m achitechure_2 accept-handoff --manifest handoffs/full35-j3-seed0/manifest.json --model-loader achitechure_2.full35_adapter:load_full35_parent

    # 看 winner 實際解析出哪些候選
    $PY -m achitechure_2 resolve-candidates --manifest handoffs/full35-j3-seed0/manifest.json

    # 對指定 resolved candidate 做 CPU forward/backward/freeze/reload
    $PY -m achitechure_2 cpu-dry-run --manifest handoffs/full35-j3-seed0/manifest.json --model-loader achitechure_2.full35_adapter:load_full35_parent --candidate C0

本輪 handoff 已 materialize 驗收通過；accepted.json 是本機可重建驗收產物，不提交 checkpoint。

## Float20 計畫與安全佇列

正式計畫只讀檢查：

    $PY -m achitechure_2 float20-plan
    $PY -m achitechure_2 queue-status

目前完整設定是 logical Detect batch 128、physical microbatch 32、Pose batch 16、Detect／Pose validation batch 16、imgsz 640、20 epochs。佇列需同時滿足 GPU 無 compute process、free memory 至少30,000 MiB、utilization不超過10%，連續3次、每次間隔60秒才啟動。C0會先跑一個完整macro作 OOM probe；32若 OOM，整個 C0～C3 矩陣統一改16，禁止只替單一候選降batch。

`float20-run`不能直接啟動：它必須驗證父程序是持鎖的`yolo_combine.gpu_queue`。Float20後以同一安全gate執行`float20-profile`；原始Float20匯出本身仍不改寫C_best。

matrix 與 profile 都完成後，`$PY -m achitechure_2 export-float20-results --execute` 會產生
`results/full35-j3-float20-seed0/` 的 JSON、CSV、中文報告、lineage、Pareto 與圖表；任何 hash、候選順序、
F1 threshold 或 Pose 契約漂移都會 fail closed。

## 已取消後續：C2/C3 full、PTQ 與 QAT-lite

正式設定 [full35-c2-c3-auto-continuation.yaml](configs/runs/full35-c2-c3-auto-continuation.yaml)
已固定為 `closed_rejected_by_user`。背景服務保持 inactive，full／quant 執行入口皆會 fail closed，
不得重新排入 queue。

原計畫會先以四項精度下降不超過 0.008 且至少一項成本下降作 gate，再執行完整 Float、PTQ 與
200-step QAT-lite；但 C2 與 C3 都未通過 gate，且使用者已決定不採用本階段，所以這些工作沒有執行，
也不存在可宣稱的下游量化結果或 C_best。

已完成的 Float20 結果與 hash manifest 保留原樣，供負結果稽核。若未來要重訪架構簡化，必須建立新的
專案或 spec revision 與新的 run ID，不能切回本封存設定。

    $PY -m achitechure_2 config-check


## 本輪 Pose 決策

本輪已收到明確 opt-in，正式 run YAML 的 `authorization.pose` 與 `training.pose_enabled` 都固定為
`true`。因此 C0-Control、C1、C2、C3 都會執行相同的 BBAT5 Pose 訓練與正式 Pose validation，
並共用資料 assignment、head seed、task ratio、optimizer steps 與 validation events。

資料準備與CPU smoke本身仍不等於Pose長訓練；本輪與後續run都已明確固定Pose=true，且只有持鎖queue可解鎖CUDA。未來若建立另一份run仍須重新授權，兩個Pose gate不一致時一律fail closed。

## 子訓練、搜尋與 validation

可以做小規模子訓練與搜尋；Full35 J3 handoff、Pose=true 與 GPU queue 均已就緒。正式規則如下：

- BBAT5 搜尋只用 pose-search／detect-search YAML；它們完全位於 formal train 內。
- BBAT5 搜尋結果若用來決定設定，必須形成一份共用 recipe，再鎖定給 C0-Control 與所有候選；
  禁止每個候選各自調參。
- Float20 必須使用已提交的固定 manifests 且 training.fraction 維持1.0；小於1只可作 plumbing，不能產生初篩或正式排名。
- formal val 不參與搜尋；鎖定設定後才用相同 validation events 比較候選。
- 本輪正式 Float20 已明確納入 Pose；未來新 run 仍須各自記錄 opt-in，不會沿用本輪授權。
- COCO train-only 固定20% contract 已建立於 architecture-screen-20-v1；使用23,657張 train 與互斥的5,000張
  train-only search-val，官方 val2017 不參與初篩。此 contract 只支援架構初篩，不升格為一般超參數搜尋集。

## AP、AP50、mAP 與 F1

- AP50：單一 IoU（box）或 OKS（keypoint）= 0.50 的 Average Precision。
- AP50-95：0.50 到 0.95 多個 threshold 的 AP 平均；通常更嚴格。
- mAP：對多個 classes 的 AP 再平均。本案會明確寫 mAP50 或 mAP50-95，不只寫模糊的 mAP。
- Macro F1：ball、bat 的 F1 未加權平均，是主要 F1，可避免多數類別掩蓋另一類。
- Micro F1：在固定 threshold 下，由官方 precision/recall curves 與 class supports 估算合併 TP/FP/FN 後的整體 F1；報告會明標 estimated。

結果必須分開記錄 COCO80 overall、COCO person、BBAT5 Pose box/keypoint、ball/bat per-class、
Precision、Recall、per-class F1、Macro F1、Micro F1 與 confidence threshold。F1 threshold 只用
C0-Control search-val 決定一次，之後所有候選的同一 search-val 固定相同值。

Stock Ultralytics Pose best.pt 依 Pose mAP50-95 + Box mAP50-95 combined fitness 選擇。本案若執行
Pose，必須另外保存 pose-only mAP50-95 best，並把 combined fitness 分欄報告。BBAT5 的兩點
keypoint 目前使用套件均勻 OKS sigma，不能宣稱已針對棒球校準。

Float20原始報告不改寫C_best。後續run-specific gate原定要求C2/C3四項主要精度相對C0下降均不超過0.008且至少一項成本下降；本次沒有候選通過，流程已依使用者決定永久封存。

## 量化

[w8a8-simulation.yaml](configs/quant/w8a8-simulation.yaml) 定義 Q0/Q1/Q2L/Q2：

- C0：預設可做 Q0/Q1。
- C2/C3：本次未通過Float20 gate，且本 revision 已封存；不執行full、PTQ或QAT-lite。
- Q2L：只保留200-step短QAT通用模板，本 revision 不執行。
- Q2：只保留完整QAT通用模板，本 revision 不執行。
- standard Conv 使用 W8A8 fake quant；custom BinaryQK/PWL attention 排除。
- 全部標示 simulation_only=true，CPU plumbing 結果不是部署 latency。
- Fake quant 仍輸出浮點 tensor；沒有 backend convert 與目標硬體驗證時，不稱為 Bit-True 或部署 INT8。

    $PY -m achitechure_2 quant-check --candidate C0 --stage Q1
    $PY -m achitechure_2 qat-lite-check --candidate C0
    # C1 尚未核准時會 fail closed
    $PY -m achitechure_2 quant-check --candidate C1 --stage Q1

## 目錄

    achitechure_2/
    ├── AGENTS.md / CONTEXT.md      本目錄工作規範與領域詞彙
    ├── EXPERIMENT_SPEC.md          唯一正式規格
    ├── README.md                   全專案入口
    ├── RUNBOOK.md                  可複製命令
    ├── TRAINING_GUIDE.md           訓練、公平性與失敗處理
    ├── configs/                    runs/candidates/training/data/quant/schema
    ├── handoffs/full35-j3-seed0/  Full35 immutable handoff manifest 與 recipe
    ├── src/achitechure_2/          Full35 adapter、候選、training、validation、queue gate
    ├── tests/                      CPU pytest
    ├── artifacts/datasets/bbat5-v1/ metadata + 獨立 github-dataset 完整 snapshot
    ├── artifacts/datasets/architecture-screen-20-v1/ 固定20%清單、統計、hash與runtime cache
    ├── results/                    空白結果與報告模板
    ├── docs/                       本子專案ADR與官方訓練參考
    └── ../../docs/worklogs/        根repository中文工作紀錄

## Git 會上傳與不會上傳的東西

會提交：

- 程式、測試、正式 YAML/schema。
- README、規格、ADR、工作紀錄。
- BBAT5 split／patch／source audit／COCO exclusion／rebuild manifests。
- 僅限 `artifacts/datasets/bbat5-v1/github-dataset/` 的完整物化影像、labels、portable YAML/splits
  與 publication manifest。
- 依 2026-08-23 明確授權，根 repository `original/` 的 raw Pose、歷史 Detect 與 canonical
  lineage；此發布只供稽核／重建，不改變正式訓練入口。

不會提交：

- `.pt`／`.pth`／`.onnx`／engine、checkpoint、cache、runs、GPU profiles。
- `original/pose/detect_dataset.zip`；它超過 GitHub 單檔限制且與已發布解壓目錄重複。
- 未驗證的 accepted.json 或正式結果。

本次完整修改、資料處理、驗證與未解風險記錄在
[2026-08-22 architecture_2 Phase A 工作紀錄](../../docs/worklogs/2026-08-22-architecture-2-phase-a.md)。
完整資料發布另見
[2026-08-22 GitHub dataset 工作紀錄](../../docs/worklogs/2026-08-22-architecture-2-github-dataset.md)。
original source 發布另見
[2026-08-23 original 資料發布工作紀錄](../../docs/worklogs/2026-08-23-original-data-publication.md)。

固定20%初篩與 QAT-lite 的完整變更、驗證、困難及未解風險見
[2026-08-26 工作紀錄](../../docs/worklogs/2026-08-26-architecture-2-float20-qat-lite.md)。

本次 Full35 handoff、正式 Float20 runner、CPU 驗證、runtime cache 修正與 queue gate 見
[2026-08-27 工作紀錄](../../docs/worklogs/2026-08-27-architecture2-full35-float20-queue.md)。

C2／C3完整訓練、PTQ／QAT-lite自動接續、有限重試與未解風險見
[2026-08-27 自動接續工作紀錄](../../docs/worklogs/2026-08-27-architecture2-c2-c3-auto-continuation.md)。

Float20正式匯出、profiling裝置修正與結果判讀見
[2026-08-27成果與修正工作紀錄](../../docs/worklogs/2026-08-27-architecture2-float20-profile-device-fix.md)。

最終封存、清理與發布決策見
[2026-08-28封存工作紀錄](../../docs/worklogs/2026-08-28-architecture2-archive-and-publication.md)。

最後操作與驗證順序見 [RUNBOOK.md](RUNBOOK.md)。目前可可靠宣稱Float20訓練、RTX 5090成本profile與
正式匯出均完成；C1～C3不採用，C2／C3完整訓練、PTQ、QAT-lite與C_best均未執行並已封存。
