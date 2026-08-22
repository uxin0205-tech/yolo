# architecture_2 訓練與公平性指南

[EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md) 是唯一正式規格。本文件只解釋為什麼這樣設定，以及正式
執行前要檢查什麼；不另立候選或訓練規則。

## 1. 為什麼 recipe 必須由 handoff 繼承

architecture_2 接收的是 yolo_combine 已選出的融合 winner。shared、routed 或 partial-shared
topology、兩個 task 的 sampling、loss、可恢復範圍與 optimizer 都可能不同。若在 winner 交付前先
填一組「看起來合理」的 LR、batch 或 stage，最後比較到的會同時是架構和訓練策略，不再是 C1–C3
單因子實驗。

因此：

- C0-Handoff 是 winner 完全不改的 immutable reference。
- C0-Control 與所有 resolved candidates 使用同一份 recovery recipe 與預算。
- 每個候選從 C0-Handoff 獨立建立新 optimizer，不承接另一候選。
- 上游決定的值在 template 中寫 source=handoff、value=null。
- handoff 後產生一份具體 effective YAML；缺值就停止。

這不代表 YAML 不能調。batch、fraction、scale、cache、imgsz、epochs、LR 等都明確列在 YAML；
只是正式 ablation 必須對所有候選同步調整並保存新 spec/config hash。

## 2. Main-task routes

融合模型有三種推論選擇：detect、pose、both。訓練資料的 loss 語意不能混淆：

- COCO batch 只計 COCO80 Detect loss。
- BBAT5 Pose batch 只計 Pose loss；BBAT5 沒有 person label，不能把人當成 Detect 背景。
- BBAT5 paired 診斷 batch 可以計 2-class Detect/Pose，但不能取代 COCO Detect 主線。

正式 runner 必須記錄每一 task 的 optimizer steps、effective samples、task ratio、accumulation、
physical batch 與 validation events。epoch 只能作為估算，不能證明兩個候選看過相同工作量。

## 3. C0-Handoff 與 C0-Control

C0-Handoff 用於證明：

- graph、contract、state_dict tensors 與 winner 完全相同。
- detect_nc=80、pose_nc=2、kpt_shape=[2,3] 不變。
- fusion topology、heads、protected/frozen modules 不變。

C0-Control 才參與恢復訓練比較。它不改架構，但接受與 C1/C2/C3 相同的 optimizer 建立方式、task
sampling、訓練 steps 與 validation budget。若拿未恢復的 C0-Handoff 直接對已恢復候選排名，差異不只
來自架構。

## 4. 動態 Candidate Regions 與 freeze

實際修改路徑由 handoff 宣告：

- shared：C1–C3 同時影響 Detect/Pose 共享區域。
- detect_specific：只影響 Detect branch。
- pose_specific：只影響 Pose branch。

每個 resolved candidate 只解凍：

- changed region。
- 該 region 對應 heads。
- handoff 明確允許的 recovery paths。

未修改的 task-specific branch 保持凍結。inherited MASF、attention、BinaryQK/PWL 的 parameters、
buffers 與 eval mode 必須永久保護。外部呼叫 model.train() 後要再次 enforce frozen eval，避免 BN
running_mean/variance 或 dropout state 漂移。

## 5. 可調參數與公平性

常用欄位在 training template 的 adjustable 區塊：

| 欄位 | 調整效果 | 正式比較要求 |
|---|---|---|
| batch | physical batch／記憶體 | 全候選相同，並另記 nbs/accumulation |
| fraction | 排序後 train 清單前段比例；val 永遠完整 | 正式固定 1.0；不能取代 grouped search |
| scale | geometry augmentation | 不是 model scale；全候選相同 |
| cache | I/O／RAM／disk | runtime 可因機器調整並記錄 |
| imgsz | 精度、成本、VRAM | 全候選相同 |
| workers | dataloader | runtime 可調 |
| epochs/patience | 最大預算與 early stop | 全候選同 gate |

recipe 區塊另保存 optimizer、lr0、lrf、momentum、weight_decay、warmup_epochs、nbs、losses、
augmentation、freeze_policy、AMP、deterministic、seed、optimizer_steps 與 validation_interval。

cache=true／ram 即使 deterministic 也可能不完全決定性；cache=disk 會在影像旁建立檔案，只能使用
可寫衍生資料。Ultralytics 的 fraction 不是隨機、分層或 grouped sampling；BBAT5 search 必須使用
已固定的 grouped search YAML。

只允許 name/project/device/workers/cache 作 runtime override。改 batch、fraction、scale、LR、loss、
optimizer 或 augmentation 時，要先更新 spec/YAML，不可只替單一候選改。

## 6. Smoke、Float 與 extension

### CPU Phase A

CPU smoke 只驗證：

- detect／pose／both forward。
- synthetic loss 與 backward。
- finite gradients。
- frozen parameters/buffers/mode。
- 64/128 fixture 與一次 640 geometry。
- state-dict strict reload。

它不讀真實 accuracy，也不宣稱 CPU/GPU latency。

### Float 主比較

Float 主比較要等：

1. handoff accepted。
2. 全候選 CPU dry-run 通過。
3. 使用者授權 GPU。
4. 使用者決定 Pose 是否執行。

第一輪 seed 0 跑完所有候選。若要正式確認某候選，再讓 C0 與該候選跑 seed 1，報告 mean 與
dispersion。不能只替喜歡的候選增加訓練預算。

### Extension

是否延長完全使用 handoff recipe 的 late-improvement gate。真正 resume 只能使用 runner 在
final strip 前另存的該候選 unstripped continuation checkpoint，並證明 optimizer、scaler、EMA、
epoch、best fitness 與 scheduler position 可恢復。正常完訓後的 stock last.pt／best.pt 已被 strip，
不符合此契約；缺少完整 state 時 extension 保持 blocked。若將來要從 weights 建立新 optimizer 延長，
必須另行升版並明稱 extension fine-tune。

## 7. Pose 是明確選擇

資料準備、YAML 檢查與 CPU interface smoke 不等於正式 Pose 執行。未收到明確 opt-in 時：

- 不啟動 Pose 長訓練。
- 不跑正式 Pose validation。
- 不把缺少 Pose 指標解讀成候選失敗。
- 不用只有 Detect 的結果自動選完整融合模型 C_best。

若啟用 Pose，C0-Control 與所有候選必須共用：

- BBAT5 v1 formal/search split。
- head seed 與初始化規則。
- task ratio、effective samples、optimizer groups。
- total optimizer steps 與 validation events。
- loss、augmentation、freeze/recovery policy。

目前 keypoint 左右／鏡像語意仍未被 kpt_names 與 mapping 證明，所以正式設定保持 fliplr=0。要開啟
水平翻轉，先修訂 spec、補 mirror mapping 與測試。

## 8. 指標與 best

每一候選分開保存：

- COCO80 box mAP50、mAP50-95。
- COCO person AP50、AP50-95。
- BBAT5 Pose box mAP50、mAP50-95。
- BBAT5 keypoint mAP50、mAP50-95。
- ball/bat 各自 box/keypoint AP50、AP50-95。
- Precision、Recall、per-class F1、Macro F1、Micro F1。
- F1 confidence threshold。

F1 threshold 只用 C0-Control search-val 決定一次，然後在所有候選的 formal val 固定。

Stock Ultralytics Pose best.pt 使用 Pose mAP50-95 與 Box mAP50-95 的 combined fitness。本案若以
Pose mAP50-95 為研究主指標，runner 必須另外保存 pose-only best，combined fitness 仍另欄保留。
BBAT5 的非 COCO 兩點 schema 目前使用均勻 OKS sigma；Micro F1 則是本案在固定 confidence／OKS
規則下聚合 TP／FP／FN 的衍生指標。

第一輪只做 measurement-first 報告。0.005/0.008 只能描述敏感度，不是自動 PASS/REJECT。成本與
精度 Pareto 產出後，C_best 仍是 null，等待使用者選擇。

## 9. Checkpoint 與 lineage

正式 checkpoint 只保存 state_dict 與重建契約，不 pickle 整個 model instance。每份都要包含：

- spec_version / spec_sha256。
- handoff revision / handoff manifest SHA256。
- architecture/training/dataset YAML SHA256。
- parent checkpoint SHA256。
- base/resolved candidate ID。
- changed paths／fields。
- model 與 architecture contract。

載入時先由 builder 重建 graph，再比對 contract 並 strict load。wrong builder、shape/key mismatch 或
lineage 缺欄位都要停止。

## 10. OOM、NaN、中斷與 resume

- OOM：記錄 requested/effective batch、candidate、route、imgsz、GPU、step 與 peak VRAM；stock trainer
  可能自動減半 batch，不可把該 run 悄悄視為 matched ablation。
- NaN/Inf：立即中止並保存有效 lineage/log；不要只替該候選改 LR/loss。
- 中斷：fresh run 從宣告 parent 重建；extension 只能從自己的未 strip continuation state 真正 resume。
- missing/shape mismatch：完整保存 matched/missing/unexpected/shape-mismatch，不宣稱全量 pretrained。
- frozen drift：任何 parameter、buffer 或 mode 改變都中止。
- 不同 recipe：公平性檢查失敗；先修正 effective YAML，不能事後在報告中淡化。

## 11. 量化

C0 預設可進 Q0/Q1；C1–C3 等使用者核准。Q2 另需 GPU 授權。W8A8 fake-quant 是 simulation：

- Conv weights：per-channel symmetric INT8。
- activations：per-tensor affine INT8。
- custom BinaryQK/PWL arithmetic：排除並列出。
- Q0/Q1/Q2 對每一個 AP/mAP/F1 指標分開記 gap/recovery。
- 不以固定 drop 自動接受；不宣稱實際部署 latency。
- fake quant 仍是浮點 quantize-dequantize simulation；沒有 backend convert、operator coverage 與
  目標硬體驗證時，不稱為 Bit-True 或部署 INT8。

官方公開原則與本機 8.4.90 對照見
[docs/OFFICIAL_TRAINING_REFERENCE.md](docs/OFFICIAL_TRAINING_REFERENCE.md)。
