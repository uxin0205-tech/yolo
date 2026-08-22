# architecture_2 執行手冊

所有命令從本目錄執行。目前 GPU 有其他工作，本 revision 的 CLI 會強制隱藏 CUDA；這份手冊不包含
任何正式長訓練命令。

## 0. 環境

    cd /home/uxin/yolo/yolo_achitechure/achitechure_2
    export PYTHONPATH="$PWD/src"
    PY=/home/uxin/yolo/.venv/bin/python
    $PY -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__)"

正式固定 Ultralytics 8.4.90。

## 1. 目前狀態與設定

    $PY -m achitechure_2 status
    $PY -m achitechure_2 show-candidates
    $PY -m achitechure_2 config-check

預期：

- current_phase=A。
- readiness=ready_for_upstream_handoff。
- C0/C1/C2/C3 全部存在。
- Pose formal execution 與 GPU actions 維持 blocked／requires opt-in。

## 2. BBAT5 資料

來源固定唯讀：

- Pose：/home/uxin/yolo/original/pose/dataset/
- Detect audit：/home/uxin/yolo/original/pose/detect_dataset/

### 唯讀 dry-plan

    $PY -m achitechure_2 prepare-pose-data

這會稽核 6,647 對影像／labels、Detect 前五欄一致性、group assignment、4 個負座標與 COCO train
重疊，但不建立檔案。

### 建立 immutable v1

只有 /home/uxin/yolo/original/pose/derived/bbat5-v1 尚不存在時才能執行：

    $PY -m achitechure_2 prepare-pose-data --execute

v1 已存在時命令會拒絕覆寫。任何新規則必須建立 bbat5-v2，不可刪除 v1 後假裝是同一版本。

### 驗證

    $PY -m achitechure_2 validate-pose-data

目前已驗證結果：

- formal train/val：5,964 / 683 images。
- search train/val：5,364 / 600 images，且只來自 formal train。
- formal/search leakage：0。
- 331 個 COCO train overlap groups 不進入 formal/search val。
- Pose/Detect 共用 assignment。
- 4 個 patch 正確、原始 label tree hash 未變。
- test split 為 null。

正式 search dataset wrappers：

- configs/data/bbat5-pose-search.yaml
- configs/data/bbat5-detect-search.yaml

兩者都只引用 formal train 內的 search lists。COCO train-only search 尚未定義；在新增正式
config/spec 前，不得用 COCO formal val 做融合 recipe 搜尋。

### 匯出 Git metadata

只預覽：

    $PY -m achitechure_2 export-data-metadata

實際匯出：

    $PY -m achitechure_2 export-data-metadata --execute

目的地是 artifacts/datasets/bbat5-v1，只包含 README、dataset YAML 與 manifests；不包含 images、
labels 或 symlink。目的地存在時同樣拒絕覆寫。

## 3. yolo_combine handoff

將 [handoff-manifest.example.json](handoff-manifest.example.json) 複製到實驗產物目錄，換成
yolo_combine 實際 winner 的 paths/hashes/contract/regions。不要把私有 checkpoint 放進本目錄。

### 只檢查 metadata

    $PY -m achitechure_2 inspect-handoff --manifest /absolute/path/handoff.json

此命令會驗證所有 artifact hashes、環境、training recipe 與 fresh-process 證據，但因沒有 materialize
graph，輸出 accepted=false，也不會寫 artifacts/intake/accepted.json。

### 正式接受

    $PY -m achitechure_2 accept-handoff --manifest /absolute/path/handoff.json --model-loader yolo_combine.some_module:load_winner

model-loader 可使用 module:function 或 /absolute/path/builder.py:function，函式需接受 checkpoint Path
並回傳 torch.nn.Module。只有 contract、Candidate Regions、protected/frozen paths 與 C3k2 baseline
都由實際 graph 證明後，才會寫 accepted.json。

### 解析候選矩陣

    $PY -m achitechure_2 resolve-candidates --manifest /absolute/path/handoff.json

shared winner 會得到 C0/C1/C2/C3；routed/partial winner 會得到 D-/P-/S- resolved IDs。

## 4. CPU 候選 dry-run

先對 C0：

    $PY -m achitechure_2 cpu-dry-run --manifest /absolute/path/handoff.json --model-loader yolo_combine.some_module:load_winner --candidate C0 --output artifacts/cpu-validation/c0.json

再依 resolve-candidates 的輸出逐一測試，例如：

    $PY -m achitechure_2 cpu-dry-run --manifest /absolute/path/handoff.json --model-loader yolo_combine.some_module:load_winner --candidate C1 --output artifacts/cpu-validation/c1.json

驗證內容：

- detect／pose／both 三種介面。
- smoke forward 與 synthetic loss/backward。
- finite gradients。
- handoff frozen parameters、BN buffers 與 eval mode。
- 640×640 geometry forward。
- state-dict strict reload 與 contract 等價。

不測 accuracy、GPU latency 或 VRAM。

## 5. 解析正式 effective YAML

handoff 通過後：

    $PY -m achitechure_2 effective-config --manifest /absolute/path/handoff.json --template configs/training/float-main.yaml --output artifacts/effective-configs/float-main.yaml

可在 runtime 層調整：

    $PY -m achitechure_2 effective-config --manifest /absolute/path/handoff.json --template configs/training/float-main.yaml --device 0 --workers 8 --cache false --name c0-seed0 --project /path/runs --output artifacts/effective-configs/float-main-c0.yaml

只允許 name/project/device/workers/cache。batch、fraction、scale、LR、optimizer 等 learning fields 要
修改正式 YAML／spec，不能只對單一候選覆寫。此命令只解析並保存設定，不會開始訓練。

float-extension 只有在 runner 於 final strip 前另存完整 continuation state 時才能解析為真正 resume；
正常完訓的 stock last.pt／best.pt 不符合條件。缺少 optimizer、scaler、EMA、epoch 或 scheduler
position 證據時保持 blocked。

## 6. 正式訓練 gate

本 revision 刻意沒有 train 命令，避免在 winner topology、完整上游 recipe、GPU 狀態或 Pose 選擇
未確定時猜測 runner。以下條件全部滿足後才進 Phase C：

1. yolo_combine winner handoff accepted。
2. 全部 resolved candidates 通過 CPU dry-run。
3. 使用者明確授權 GPU。
4. 使用者決定是否執行 Pose。
5. effective YAML 已產生並通過 fairness/hash 檢查。

若 Pose 不執行，可以先做 Detect 成果，但不能宣稱完整融合模型 C_best。若 Pose 執行，所有候選與
C0-Control 必須使用相同資料、head seed、task ratio、optimizer steps 與 validation events。

## 7. Float 結果報告

依 [results/results-template.csv](results/results-template.csv) 與
[results/report-template.md](results/report-template.md) 填寫完整 AP/mAP/F1/成本與 lineage。
JSON input 格式對應 achitechure_2.decisions.CandidateMetrics，然後執行：

    $PY -m achitechure_2 assess --input /path/float-metrics.json --output artifacts/assessment/float.json

輸出只提供 C0 deltas、描述性 0.005/0.008 bands 與 Pareto front：

- c_best 永遠是 null。
- selection_status 永遠是 pending_user_decision。
- C1/C2/C3 quantization eligibility 永遠先是 pending_user_decision。

由使用者看完結果後決定。

## 8. 量化資格

C0 Q0/Q1：

    $PY -m achitechure_2 quant-check --candidate C0 --stage Q1

C1 未核准時會拒絕：

    $PY -m achitechure_2 quant-check --candidate C1 --stage Q1

使用者在 Float 報告後明確核准時：

    $PY -m achitechure_2 quant-check --candidate C1 --stage Q1 --approved C1

Q2 即使 C0 也需要 GPU 授權；quant-check 只檢查資格，不會真的開始 PTQ/QAT。所有量化都是
simulation_only。

## 9. CPU 驗證

    env CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" $PY -m pytest -q
    $PY -m achitechure_2 config-check
    $PY -m achitechure_2 validate-pose-data
    $PY -m ruff check .
    git status --short --branch

integration test 在沒有 ARCHITECHURE_2_HANDOFF 時會明確 skip，不能把 skip 誤寫成正式 winner 已通過。

## 10. Git 上傳範圍

提交程式、測試、YAML/schema、中文文件與 artifacts/datasets/bbat5-v1 metadata。禁止提交：

- pose/detect images 或 labels。
- .pt/.onnx/cache。
- runs、GPU profiles。
- 未經驗收的 accepted handoff。

GitHub Issue／push 若因 gh 未登入或遠端分支落後而阻擋，先保留本地範圍明確的 commit，不能用
destructive reset 或強制 push 覆蓋遠端。
