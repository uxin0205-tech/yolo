# 2026-08-22：完成 architecture_2 Phase A 與 BBAT5 v1

## 任務與範圍

重新定義 /home/uxin/yolo/yolo_achitechure/achitechure_2 的責任：承接 yolo_combine 最後選出的
YOLO26m Detect–Pose fusion winner，以 C0、C1、C2、C3 單因子方式評估 C3k2 簡化。本階段只做
CPU Phase A、資料、設定、契約與報告基礎，不執行正式 GPU 長訓練；Pose 正式訓練／validation
保留為使用者明確 opt-in。

## 變更內容與原因

### 規格與專案邊界

- 將 EXPERIMENT_SPEC.md 設為唯一正式規格，plan.md 只保留指向。
- 規格升至 2.0.1，最終 SHA256 是
  9897ece524fd62cd55e3afc46b6bd810abbef5858f75493d1a4ba895585a4f0c。
- yolo_combine 負責 fusion／雙權重／架構切換與 winner；architecture_2 不重做融合。
- 正式模型語意固定為 COCO80 Detect（應用關注 person）與 BBAT5 ball／bat Pose；BBAT5
  2-class Detect 只作 paired diagnostic。

### C0–C3、handoff 與 CPU 契約

- C0 不改 C3k2；C1 只把 e 由 0.5 改成 0.375；C2 只把 inner_n 由 2 改成 1；C3 只把第一個
  inner 3x3 改為 1x1。
- 候選位置不寫死 layer index，改由 yolo_combine handoff 的 shared／detect-specific／
  pose-specific Candidate Regions 動態解析。
- 建立 handoff v2 schema、可解析範例、artifact SHA256、state-dict-only checkpoint、
  materialized graph、weight-transfer、freeze/buffer/mode 與 fresh-process reload 契約。
- 建立 detect／pose／both、synthetic loss/backward、finite gradient、640 geometry 與 strict
  reload CPU dry-run。

### 正式 YAML 與結果

- 建立 C0–C3 candidate YAML、四份 training templates、五份 dataset YAML、W8A8 simulation YAML、
  catalog 與 schemas。
- batch、fraction、augmentation scale、cache、imgsz、device、workers、epochs、patience、optimizer、
  LR、nbs、loss、task ratio 與 freeze 都在 YAML 中明列來源和值。
- 新增 config-check，未知欄位、spec/hash 漂移、候選因素漂移、learning CLI override 與未授權
  Pose/GPU/QAT 都 fail closed。
- 結果範本分開保存 AP50、AP50-95、mAP50、mAP50-95、per-class P/R/F1、Macro/Micro F1、
  fixed confidence threshold、Params、GFLOPs、latency、VRAM 與完整 lineage。
- 第一輪只輸出 C0 delta、描述性 0.005／0.008 band 與 Pareto；C_best 固定 null，等待使用者決定。

### BBAT5 v1 衍生資料

- 原始 Pose：/home/uxin/yolo/original/pose/dataset/，唯讀。
- 原始 Detect audit：/home/uxin/yolo/original/pose/detect_dataset/，唯讀。
- 衍生版本：/home/uxin/yolo/original/pose/derived/bbat5-v1/，不可覆寫。
- Pose labels 是權威；Detect labels 由修補後前五欄產生。
- 依 .rf. 前 prefix grouped split：6,647 images、2,578 groups；formal train／val
  5,964／683；search train／val 5,364／600，且 search 完全位於 formal train。
- 331 個 COCO train overlap groups 固定留在 train；formal/search val overlap 為 0。
- 四個負座標只在衍生 labels clamp 為 0；原始 label tree hash 未改變。
- Pose／Detect 共用 assignment；未建立 test split。
- Git 只保存 README、四份重建 dataset YAML 與 split／patch／source-audit／COCO-exclusion／
  rebuild manifests，不保存影像、衍生 labels、cache 或 symlink。
- bbat5-v1 保留建立時 spec 2.0.0 與 hash
  75db239262de75998171a05a85c8755dea10c2bc76920dd2aabe8ff0dabb7a3b，不回溯改寫。

### 官方查核後修正

- Ultralytics 8.4.90 正常完訓會 strip stock last.pt／best.pt；真正 extension resume 必須在 strip
  前另存未 strip continuation state。缺少 optimizer、scaler、EMA、epoch 與 scheduler-position
  證據時保持 blocked。
- Stock Pose best.pt 依 Pose mAP50-95 + Box mAP50-95 combined fitness；研究流程若使用
  pose-only mAP50-95 必須另存 best，兩種指標分欄報告。
- fraction 只截取排序後 train 前段，不是 grouped split；cache=ram 可能不完全 deterministic，
  cache=disk 只能寫衍生資料。
- PyTorch fake quant 仍是浮點 simulation；沒有 backend convert 與目標硬體驗證時，不宣稱
  Bit-True 或部署 INT8。

## 驗證方式與結果

- env CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" /home/uxin/yolo/.venv/bin/python -m pytest -q
  ：51 passed、1 skipped，耗時 9.35 秒。
- 唯一 skip 是未提供 ARCHITECHURE_2_HANDOFF 的正式 fusion winner integration；不是 dataset skip。
- python -m ruff check .：All checks passed。
- python -m achitechure_2 config-check：valid=true，spec 2.0.1、Ultralytics 8.4.90、C0–C3 與四份
  training templates 全部通過。
- python -m achitechure_2 validate-pose-data：valid=true、formal_ready=true、6,647 Pose／Detect
  images、4 patches、331 overlap groups、source_untouched=true、test=null。
- compileall 通過；21 份 YAML 全部可解析；results CSV 共 6 列、62 欄且每列欄位一致。
- handoff-manifest.example.json 已由 HandoffManifest v2 loader 測試通過。
- 開發環境目前未安裝 jsonschema，因此沒有宣稱已執行獨立 Draft 2020-12 validator；pyproject 的
  dev extras 已加入 jsonschema，現有 config-check／loader 仍完成更嚴格的專案語意驗證。

## 困難與解法

1. GPU 正被其他工作使用。解法是固定隱藏 CUDA，只執行 CPU Phase A；沒有探測、占用或啟動 GPU。
2. yolo_combine 尚未產出 winner，真正 module paths 與上游 recipe 不能猜。解法是建立 fail-closed
   handoff、動態 Candidate Regions 與 null handoff fields，不提供 train 命令。
3. 原始 BBAT5 augmentation 可能同源跨既有 train／valid。使用者授權後，在獨立 derived/bbat5-v1
   依 source group 重建，原始來源維持唯讀，且留下完整 manifests 與中文 README。
4. 官方原始碼顯示 stock last.pt 完訓後被 strip，舊「直接 last resume」規則不成立。解法是把 spec
   升到 2.0.1，YAML/schema/loader 明確要求未 strip continuation state。
5. 根 Git 工作樹已有大量無關變更且 main 與 origin/main 分歧。發布時使用 origin/main 的暫存
   worktree，只複製本次獲授權範圍，禁止 reset、force push 或混入其他變更。
6. 執行環境的 bwrap 無法建立 loopback。解法是經核准後在 sandbox 外執行必要的唯讀命令與
   apply_patch；未使用 destructive reset。

## 清理與 Git 範圍

- .pytest_cache、.ruff_cache 與兩個 __pycache__ 只盤點、不刪除，且已由 .gitignore 排除。
- 淘汰舊 artifacts/datasets/pose_grouped 的 split／patch manifest，正式位置只有
  artifacts/datasets/bbat5-v1。
- commit 只包含 architecture_2、根層 README／AGENTS 的全域入口與追加規則、architecture_1 的
  全域文件入口、docs/agents/bbat5-datasets.md，以及兩份相關工作紀錄與索引；不包含其他既有
  dirty worktree 內容。

## 未解事項或風險

- 等待 yolo_combine 正式 winner handoff；目前沒有 C0–C3 實測精度、C_best 或量化接受結論。
- Pose 正式訓練／validation、GPU smoke、latency／VRAM、QAT 仍需使用者另外授權。
- 目前只有 BBAT5 formal-train-only search YAML；COCO train-only search contract 尚未建立，所以
  不能宣稱已完成雙任務融合超參數搜尋。
- BBAT5 keypoint mirror 語意與專用 OKS sigma 尚未確認，因此 fliplr=0，且目前均勻 sigma 不宣稱
  已針對棒球校準。
