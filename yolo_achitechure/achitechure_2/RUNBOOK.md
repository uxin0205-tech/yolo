# architecture_2 執行手冊

> **封存狀態（2026-08-28）**：本階段 C1～C3 不採用。C2/C3 full、PTQ 與 QAT-lite 永久關閉，
> 不得啟動 service、script 或 queue。下列 Float20 命令只供既有結果稽核與重建，不是新 GPU
> 執行授權。

所有命令從本目錄執行。本專案不假設目前GPU工作何時結束；只允許由持鎖queue在連續3次idle檢查
通過後解鎖CUDA。一般CLI與所有CPU驗證仍強制隱藏CUDA。

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

- current_phase=B。
- fusion_winner=full35-final-j3-seed0-d67fb45c。
- C0/C1/C2/C3 全部存在且 CPU dry-run 已通過。
- GPU 與 Pose=true 已授權；queue 狀態以 `queue-status` 與 systemd service 為準。

## 2. BBAT5 資料

來源固定唯讀：

- Pose 原始來源（只供重建）：/home/uxin/yolo/original/pose/dataset/
- Detect audit 原始來源（只供重建）：/home/uxin/yolo/original/pose/detect_dataset/
- 全專案正式 registry：/home/uxin/yolo/configs/datasets/bbat5-v1.yaml

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
    $PY -m achitechure_2 validate-screening-data

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

兩者都只引用 formal train 內的 search lists。架構初篩另使用固定、不可覆寫的20% View：

    # 唯讀規劃
    $PY -m achitechure_2 prepare-screening-data

    # 只有目的地不存在時建立 manifests；不複製影像／labels
    $PY -m achitechure_2 prepare-screening-data --execute

    # 驗證 hash、COCO train/search-val互斥、BBAT5同 assignment與零group leakage
    $PY -m achitechure_2 validate-screening-data

目前固定結果：COCO train 23,657、COCO train-only search-val 5,000、BBAT5 train 1,073／410 groups、
BBAT5 search-val 600。官方 COCO val2017 與 BBAT5 formal val 都未使用；training fraction 保持1.0。

### 匯出 Git metadata

只預覽：

    $PY -m achitechure_2 export-data-metadata

實際匯出：

    $PY -m achitechure_2 export-data-metadata --execute

目的地是 artifacts/datasets/bbat5-v1，只包含 README、dataset YAML 與 manifests；不包含 images、
labels 或 symlink。目的地存在時同樣拒絕覆寫。

### 匯出完整 GitHub dataset

這是使用者明確核准的獨立 publication snapshot，不會改寫本機 canonical v1，也不會啟動 Pose：

    # 只計畫與驗證來源
    $PY -m achitechure_2 export-github-dataset

    # 物化一般影像檔、Pose/Detect labels、portable YAML/splits 與 manifest
    $PY -m achitechure_2 export-github-dataset --execute

    # 驗證零 symlink、零權重、零 test、split 可攜及整體 tree SHA256
    $PY -m achitechure_2 validate-github-dataset

固定目的地是 `artifacts/datasets/bbat5-v1/github-dataset/`。目的地存在時拒絕覆寫；需要重做時
不得直接刪除或覆蓋已發布版本，必須先確認 recovery path 與新版本政策。

## 3. yolo_combine handoff

本輪不再使用範例推測 winner；正式、不可變的 handoff 是
[full35-j3-seed0/manifest.json](handoffs/full35-j3-seed0/manifest.json)，checkpoint 與 builder 仍留在
`/home/uxin/yolo/yolo_combine/final/full35/`，本目錄只提交 manifest 與解析後 training recipe。
[handoff-manifest.example.json](handoff-manifest.example.json) 只供未來建立新 handoff revision 時參考。

### 只檢查 metadata

    $PY -m achitechure_2 inspect-handoff --manifest handoffs/full35-j3-seed0/manifest.json

此命令會驗證所有 artifact hashes、環境、training recipe 與 fresh-process 證據，但因沒有 materialize
graph，輸出 accepted=false，也不會寫 artifacts/intake/accepted.json。

### 正式接受

    $PY -m achitechure_2 accept-handoff --manifest handoffs/full35-j3-seed0/manifest.json --model-loader achitechure_2.full35_adapter:load_full35_parent

model-loader 可使用 module:function 或 /absolute/path/builder.py:function，函式需接受 checkpoint Path
並回傳 torch.nn.Module。只有 contract、Candidate Regions、protected/frozen paths 與 C3k2 baseline
都由實際 graph 證明後，才會寫 accepted.json。

### 解析候選矩陣

    $PY -m achitechure_2 resolve-candidates --manifest handoffs/full35-j3-seed0/manifest.json

本輪 Full35 shared winner 固定得到 C0/C1/C2/C3，實際修改 layers 6、8、13、19。

## 4. CPU 候選 dry-run

先對 C0：

    $PY -m achitechure_2 cpu-dry-run --manifest handoffs/full35-j3-seed0/manifest.json --model-loader achitechure_2.full35_adapter:load_full35_parent --candidate C0 --output artifacts/cpu-validation/c0.json

再依 resolve-candidates 的輸出逐一測試，例如：

    $PY -m achitechure_2 cpu-dry-run --manifest handoffs/full35-j3-seed0/manifest.json --model-loader achitechure_2.full35_adapter:load_full35_parent --candidate C1 --output artifacts/cpu-validation/c1.json

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

    $PY -m achitechure_2 effective-config --manifest handoffs/full35-j3-seed0/manifest.json --template configs/training/float-main.yaml --output artifacts/effective-configs/float-main.yaml

可在 runtime 層調整：

    $PY -m achitechure_2 effective-config --manifest handoffs/full35-j3-seed0/manifest.json --template configs/training/float-main.yaml --device 0 --workers 8 --cache false --name c0-seed0 --project /path/runs --output artifacts/effective-configs/float-main-c0.yaml

只允許 name/project/device/workers/cache。batch、fraction、scale、LR、optimizer 等 learning fields 要
修改正式 YAML／spec，不能只對單一候選覆寫。此命令只解析並保存設定，不會開始訓練。

float-extension 只有在 runner 於 final strip 前另存完整 continuation state 時才能解析為真正 resume；
正常完訓的 stock last.pt／best.pt 不符合條件。缺少 optimizer、scaler、EMA、epoch 或 scheduler
position 證據時保持 blocked。

## 6. Float20 Pose gate 與 GPU queue

已完成：Full35 J3 handoff accepted、C0～C3 CPU dry-run、原生 Detect/Pose loss、20% manifests、GPU 與 Pose=true 授權。正式矩陣已交由 queue 執行。

先只看計畫：

    $PY -m achitechure_2 float20-plan
    $PY -m achitechure_2 queue-status

`float20-plan`的`cpu_preflight.ready`必須是true、`passed_candidates`必須恰為C0～C3，且
`source_adjacent_caches_absent`必須是true。缺少accepted或CPU報告時，先重跑第3、4節命令；
preflight不會啟動GPU，也不會把CPU smoke冒充accuracy。

本輪 `configs/runs/full35-float-screen-20.yaml` 已選擇 joint；未來新 run 仍須明確選擇：

- joint Float20：`authorization.pose: true` 且 `training.pose_enabled: true`。
- Detect-only Float20：兩者都設 `false`；Pose metrics 標成 not_run，不能稱完整融合模型排名。
- pending 或兩欄不一致：fail closed，禁止排隊。

設定後先執行 `config-check`。只有檢查通過才可建立 transient user service：

    systemd-run --user --unit=yolo-architecture2-full35-j3-float20-seed0 \
      --property=WorkingDirectory=/home/uxin/yolo/yolo_achitechure/achitechure_2 \
      --setenv=PYTHONPATH=/home/uxin/yolo/yolo_achitechure/achitechure_2/src:/home/uxin/yolo/yolo_combine/final/full35/code/project/src \
      /home/uxin/yolo/.venv/bin/python -c 'from yolo_combine.gpu_queue import main; raise SystemExit(main())' \
      --queue-id architecture2-full35-j3-float20-seed0 --gpu-index 0 \
      --min-free-mib 30000 --max-utilization 10 --stable-polls 3 --poll-seconds 60 \
      --state /home/uxin/yolo/yolo_achitechure/achitechure_2/artifacts/queue/full35-j3-float20-seed0.json \
      --log /home/uxin/yolo/yolo_achitechure/achitechure_2/artifacts/queue/full35-j3-float20-seed0.log \
      --lock /home/uxin/yolo/yolo_achitechure/achitechure_2/artifacts/queue/full35-j3-float20-seed0.lock \
      --working-directory /home/uxin/yolo/yolo_achitechure/achitechure_2 -- \
      /home/uxin/yolo/.venv/bin/python -m achitechure_2 \
      --project-root /home/uxin/yolo/yolo_achitechure/achitechure_2 float20-run \
      --config /home/uxin/yolo/yolo_achitechure/achitechure_2/configs/runs/full35-float-screen-20.yaml \
      --queue-state /home/uxin/yolo/yolo_achitechure/achitechure_2/artifacts/queue/full35-j3-float20-seed0.json --execute

queue 僅在 GPU 無 compute process、free≥30,000 MiB、util≤10%，連續3次60秒檢查都通過後啟動。`float20-run` 另驗證 queue parent、child PID、queue_id 與 lock；直接呼叫會拒絕。

OOM 規則：先用 C0 跑 logical Detect 128 的完整macro，physical 32、累積4次，每macro兩個logical Detect batch；若 OOM，丟棄 probe 並讓整個矩陣統一改 physical 16、累積8次。Pose train與兩任務validation固定16。每候選20 epochs、93 optimizer steps/epoch、共1,860 steps。

`matrix-complete.json` 產生後，先保存訓練 queue 的完成 state／log，再以同一把鎖排入一次成本量測；
`float20-profile` 同樣不能繞過 queue：

    systemd-run --user --unit=yolo-architecture2-full35-j3-float20-profile-seed0 \
      --property=WorkingDirectory=/home/uxin/yolo/yolo_achitechure/achitechure_2 \
      --setenv=PYTHONPATH=/home/uxin/yolo/yolo_achitechure/achitechure_2/src:/home/uxin/yolo/yolo_combine/final/full35/code/project/src \
      /home/uxin/yolo/.venv/bin/python -c 'from yolo_combine.gpu_queue import main; raise SystemExit(main())' \
      --queue-id architecture2-full35-j3-float20-seed0 --gpu-index 0 \
      --min-free-mib 30000 --max-utilization 10 --stable-polls 3 --poll-seconds 60 \
      --state /home/uxin/yolo/yolo_achitechure/achitechure_2/artifacts/queue/full35-j3-float20-seed0.json \
      --log /home/uxin/yolo/yolo_achitechure/achitechure_2/artifacts/queue/full35-j3-float20-profile-seed0.log \
      --lock /home/uxin/yolo/yolo_achitechure/achitechure_2/artifacts/queue/full35-j3-float20-seed0.lock \
      --working-directory /home/uxin/yolo/yolo_achitechure/achitechure_2 -- \
      /home/uxin/yolo/.venv/bin/python -m achitechure_2 \
      --project-root /home/uxin/yolo/yolo_achitechure/achitechure_2 float20-profile \
      --config /home/uxin/yolo/yolo_achitechure/achitechure_2/configs/runs/full35-float-screen-20.yaml \
      --queue-state /home/uxin/yolo/yolo_achitechure/achitechure_2/artifacts/queue/full35-j3-float20-seed0.json --execute

它以各候選 `best-joint-screening` EMA、batch=1、imgsz=640、相同 RTX 5090，量測
Params／Detect-Pose-both GFLOPs、25 warmups、100 iterations latency 與 peak VRAM，寫入
`artifacts/runs/full35-j3-float20-seed0/profiles/cost-profiles.json`。

注意：shared head的`anchors/strides`是inference時建立的動態cache，未註冊為module buffer。GFLOPs必須
在model已搬到最終GPU後執行，否則CPU GFLOPs會留下CPU cache並使後續CUDA forward失敗。
`continue_full35_j3_float20.sh`可使用已封存的成功training queue state重試profile；失敗state／log應先
移入`artifacts/queue/history/`，不得覆寫歷史證據或把新舊log附加混用。

Float20原始runner、profile與報告均已完成，並保留`c_best=null`。C2/C3後續設定目前為
`closed_rejected_by_user`；full、Q1、Q2L與完整Q2都不執行。

## 7. Float 結果報告

完整 matrix 與 cost profile 都通過後，執行一次 fail-closed 匯出：

    $PY -m achitechure_2 export-float20-results --execute

固定輸出 `results/full35-j3-float20-seed0/`，包含 metrics JSON／CSV、lineage、selection、中文
`REPORT.md`、三張比較圖與檔案 hash manifest。匯出器會驗證 spec／training／parent／candidate
checkpoint hashes、C0 固定 F1 threshold、完整 C0～C3 順序與 Pose=true，不接受缺候選或舊 revision。

Float20輸出只提供完整實測、C0 deltas、描述性0.005/0.008 bands與Pareto front：

- c_best 永遠是 null。
- selection_status 永遠是 pending_user_decision。
- C1/C2/C3 quantization eligibility在此原始報告中仍是pending_user_decision。

後續run-specific流程原定只判定C2/C3，現已依使用者決定永久封存；狀態記在
`configs/runs/full35-c2-c3-auto-continuation.yaml`，不回寫Float20歷史報告。

## 8. 已取消接續 C2/C3 full、PTQ與QAT-lite

背景服務必須保持inactive，正式YAML是`closed_rejected_by_user`，程式入口會fail closed。只讀檢查：

    systemctl --user status yolo-architecture2-full35-j3-c2-c3-auto-seed0.service
    $PY -m achitechure_2 config-check


`scripts/continue_full35_c2_c3_auto.sh`不得執行。以下只保存原流程供稽核，不是待辦或重新啟動說明：

1. 驗證已完成的Float20 postprocess結果與匯出manifest。
2. 驗證Float20輸出manifest每個檔案hash與C0～C3順序。
3. 只讓C2/C3接受資格判定：四項精度相對C0下降均不超過0.008，且Params／GFLOPs／latency至少一項改善。
4. 通過者各自從相同J3 C0-Handoff重新建立，用完整COCO與BBAT5 v1、Pose=true，最多100 epochs、patience20。
5. full matrix完成後自動排入Q0、Q1 W8A8 PTQ simulation與200-step Q2L QAT-lite simulation。
6. 匯出完整訓練、量化與最終C_best中文報告。

本次C2、C3均未通過第3步的0.008精度門檻；依現行規則應停止，不排GPU。

full與quant各自再次經GPU idle gate：free≥30,000 MiB、util≤10%、連續3次、每次60秒，並使用獨立
queue state、log與lock。`full-run`與`quant-run`不能由互動shell直接啟動，只接受持鎖queue parent。

本 revision 不允許重新啟動；程式會在執行前因`closed_rejected_by_user` fail closed。
若未來另立新 revision，仍須重新驗證所有hash、資料、checkpoint與queue grant。

固定輸出：

- full：`artifacts/runs/full35-j3-c2-c3-full-seed0/`
- PTQ／QAT-lite：`results/full35-j3-c2-c3-quant-seed0/`
- 最終報告：`results/full35-j3-c2-c3-final-seed0/`
- queue state與log：`artifacts/queue/full35-j3-c2-c3-{full,quant}-seed0.*`

只有需要在全部前置產物已完成後手動重匯出報告時，才使用：

    $PY -m achitechure_2 export-downstream-results \
      --config configs/runs/full35-c2-c3-auto-continuation.yaml --execute

Q1與Q2L都是W8A8 fake-quant simulation；沒有backend convert與目標硬體驗證，不能宣稱部署INT8或
Bit-True。Q2L是本次授權的簡單QAT，完整Q2長訓練未排入。

## 9. 通用量化資格檢查

C0 Q0/Q1：

    $PY -m achitechure_2 quant-check --candidate C0 --stage Q1

C1 未核准時會拒絕：

    $PY -m achitechure_2 quant-check --candidate C1 --stage Q1

使用者在 Float 報告後明確核准時：

    $PY -m achitechure_2 quant-check --candidate C1 --stage Q1 --approved C1

Q2L 資格檢查（仍不會開始訓練）：

    $PY -m achitechure_2 qat-lite-check --candidate C0
    $PY -m achitechure_2 qat-lite-check --candidate C1 --approved C1 --gpu-authorized

Q2L 與 Q2 都需要執行時段的 GPU 授權；Q2L 固定200 steps，只判斷能否恢復部分 PTQ gap。
quant-check／qat-lite-check都只檢查通用資格，不會真的開始PTQ/QAT。本輪C2/C3的實際執行授權由
第8節的run-specific YAML與queue負責。所有量化都是simulation_only。

## 10. CPU 驗證

    env CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" $PY -m pytest -q
    $PY -m achitechure_2 config-check
    $PY -m achitechure_2 validate-pose-data
    $PY -m achitechure_2 validate-screening-data
    $PY -m achitechure_2 validate-github-dataset
    $PY -m ruff check .
    git status --short --branch

Full35 的正式 materialized 驗收與 C0～C3 CPU dry-run 已另外完成。pytest 中標記為 integration 的
重型重跑預設 skip；設定 `ARCHITECHURE_2_HANDOFF=1` 才會再次載入 2.35 GB release、建構 C1 並驗證契約。

## 11. Git 上傳範圍

提交程式、測試、YAML/schema、中文文件、bbat5-v1 metadata，以及使用者核准的
`artifacts/datasets/bbat5-v1/github-dataset/` 完整 snapshot。禁止提交：

- snapshot 固定目錄以外的 pose/detect images 或 labels。
- `.pt`／`.pth`／`.onnx`／engine／cache。
- runs、GPU profiles。
- 未經驗收的 accepted handoff。

GitHub Issue／push 若因 gh 未登入或遠端分支落後而阻擋，先保留本地範圍明確的 commit，不能用
destructive reset 或強制 push 覆蓋遠端。
