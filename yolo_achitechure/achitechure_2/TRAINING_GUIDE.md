# architecture_2 訓練與公平性指南

[EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md) 是唯一正式規格。本文件只解釋為什麼這樣設定，以及正式
執行前要檢查什麼；不另立候選或訓練規則。

## 1. 固定承接 Full35 J3 handoff

本輪 parent 已不是待定 winner，而是不可變的
`/home/uxin/yolo/yolo_combine/final/full35/` accepted J3 EMA：

- 正式 manifest：`handoffs/full35-j3-seed0/manifest.json`。
- shared graph 23 層，Detect nc=80、Pose nc=2、kpt_shape=[2,3]。
- 以 final factory 重建 graph，再嚴格載入 1,238 個 EMA tensors。
- checkpoint、release、builder、architecture、training recipe、dataset 與 fresh-process 證據都以 SHA256 鎖定。
- J3 challenger 仍屬上游實驗，本輪不得在途中換 parent。

C0-Handoff 是上述模型完全不改的 immutable reference；C0-Control 才以與 C1～C3 相同的20%恢復
預算參與比較。每個候選都重新從同一 C0-Handoff 建立 graph 與 optimizer，不承接上一候選的權重。
通用 template 仍保留 `source=handoff` 欄位供未來 revision 使用；本輪實際數值已解析進
`handoffs/full35-j3-seed0/training-recipe.yaml` 與 `configs/runs/full35-float-screen-20.yaml`，缺值或 hash
不符就停止。

這不代表 YAML 不能調。batch、fraction、scale、cache、imgsz、epochs、LR、loss 與 augmentation 都
明列在正式 run YAML；但方法級變更必須先升級 spec，且 C0～C3 必須共用同一份新 hash。

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

## 4. 固定 Candidate Region 與 freeze

Full35 handoff 已把唯一 shared Candidate Region 驗證為四個 C3k2：

- `graph.model.6`
- `graph.model.8`
- `graph.model.13`
- `graph.model.19`

C1、C2、C3 都同時修改這四個 shared paths，因此結構會共同影響 Detect 與 Pose；本輪不再展開
routed、partial-shared、D-/P-/S- 候選。`graph.model.23.detect_head` 與
`graph.model.23.pose_head` 受保護，不能因簡化而改類別或輸出契約。

永久凍結並維持 eval mode 的 inherited modules 是：

- `graph.model.10.m.0.attn`
- `graph.model.16.p3_masf`
- `graph.model.22.m.0.1.attn`

外部呼叫 `model.train()` 後，runner 會重新 enforce frozen parameters、buffers 與 eval mode，避免
BN running statistics 或 dropout state 漂移。每個 epoch 與 checkpoint reload 都會再檢查 protected
contract；任何逸出四個 Candidate paths 的 missing、unexpected 或 shape mismatch 都直接失敗。

## 5. 正式 Run YAML 的可調參數與公平性

Float20可排隊設定是`configs/runs/full35-float-screen-20.yaml`；後續full／PTQ／QAT-lite設定是
`configs/runs/full35-c2-c3-auto-continuation.yaml`。下面常用值是Float20的`training`契約：

| 欄位 | 目前契約 | 理由／變更規則 |
|---|---|---|
| `batch.detect_logical` | 128 | 沿用 Full35 訓練語意；由 microbatch + accumulation 實現 |
| `batch.detect_physical_microbatch` | 32 | C0 完整 macro probe OOM 時，全矩陣統一 fallback 16 |
| `batch.pose_physical` | 16 | 只在 Pose=true 時使用 |
| `batch.validation_detect/pose` | 16／16 | 避免上游 validator 大 batch OOM |
| `fraction` | 1.0 | 20%已在 immutable manifest，禁止再次 fraction 抽樣 |
| `scale` | 0.5 | augmentation 幾何 scale，不是 YOLO26m model scale |
| `cache` | false | 可改 true／ram；禁止 disk |
| `imgsz` | 640 | 四候選與兩任務一致 |
| `workers.detect/pose` | 4／8 | I/O runtime 值，必須記錄 |
| `epochs` | 20 | 初篩不 early-stop，避免候選預算不同 |
| `optimizer`／`nbs` | MuSGD／64 | 明確固定，不使用 auto |

`cache` 控制 image cache；Ultralytics 即使 `cache:false` 仍可能建立 label index，因此本專案以
`RuntimeLabelCacheYOLODataset` 把 train／validation label cache 固定導向
`artifacts/datasets/architecture-screen-20-v1/runtime-cache/`。canonical COCO 與 bbat5-v1 旁不得出現
新 `.cache`。

`fraction` 不是隨機、分層或 grouped sampling。BBAT5 search 必須使用已固定的 grouped search YAML；
Float20 則必須完整讀取 screen manifest 所列樣本。logical batch、physical batch、accumulation、task
weights、optimizer steps、effective samples 與 validation events都會寫入 run manifest。

formal runner 不接受 CLI 覆寫 learning fields。調整 batch、fraction、scale、imgsz、epochs、LR、loss、
optimizer 或 augmentation 時，要先修改正式 spec/YAML並讓全部候選共用；device、workers、cache 的
runtime差異也必須留下 hash與工作紀錄。

## 6. CPU gate、Float20 與後續階段

### CPU contract gate（已完成）

Full35 J3 已在 CPU 通過：

- checkpoint 嚴格載入 1,238 tensors，C0 transfer matched 1,238、missing/unexpected/shape mismatch皆為0。
- C0～C3 都完成 Detect／Pose／both、640×640 三尺度輸出、finite synthetic loss/gradient、freeze與strict reload。
- C1只改 `e`、C2只改 `inner_n`、C3只改 `kernel_mode`，且都只出現在四個 Candidate paths。
- 真實 COCO／BBAT5 batch 已建立官方 Detect/Pose criterion；Pose 是 `PoseLoss26`，RealNVP/RLE路徑存在，
  loss有限且可反向傳播。
- runtime label cache 已證明寫在子專案 artifacts，source-adjacent cache不存在。

這些是結構與數值 smoke，不是 accuracy、GPU latency、VRAM或量化結果。

### Float 固定20%初篩（Pose=true，已完成）

C0-Control、C1、C2、C3 共用 architecture-screen-20-v1：

- COCO train 23,657，另有互斥 train-only search-val 5,000；不使用官方 val2017。
- BBAT5 train 1,073張／410 groups，沿用 search-val 600；不使用 formal val。
- 20 epochs、每候選93 optimizer steps/epoch、合計1,860 steps；每 epoch 都用相同 validation event。
- logical Detect batch 128、physical 32，若 C0 probe OOM則整個矩陣統一16；Pose train與兩任務validation為16。
- queue 必須觀察 GPU 無 compute process、free≥30,000 MiB、util≤10%，連續3次、每次間隔60秒才解鎖。
- 本輪兩個 Pose gate 已由使用者明確設為 true；未來 run 若 pending 或不一致仍 fail closed。

Float20結果只作架構初篩。若不跑 Pose，只能形成 Detect／成本成果，不能稱完整融合模型排名；若跑
Pose，runner會另外保存pose-only best與官方Pose+Box combined fitness。Float20原始矩陣與報告到此固定
`c_best=null`；獨立接續流程已依使用者決定封存，不再執行。

### C2/C3完整Float接續（已取消）

本次Float20中C2是最佳簡化候選，但C2與C3的四項主要精度下降均超過0.008，所以依原gate都不合格；使用者已決定不採用本階段，full、PTQ與QAT-lite均不得排入queue。

只檢查C2與C3；它們的COCO box mAP50-95、BBAT5 Pose box mAP50-95、keypoint mAP50-95與Macro F1
相對C0下降都不得超過0.008，且Params／GFLOPs／latency至少一項改善。通過者不是從Float20權重resume，
而是各自重新從同一J3 C0-Handoff建立model與optimizer，用完整COCO與BBAT5 v1、Pose=true訓練最多
100 epochs、patience20。physical32若共同probe OOM，整個full matrix統一fallback16；validation固定16。

額外seed與late extension仍未自動執行。Extension只接受runner在strip前保存的同候選continuation state，
並恢復optimizer、scaler、EMA、epoch、best與scheduler position；stock stripped`last.pt`／`best.pt`不能
冒充真正resume。若只從weights建新optimizer，必須另升版並明稱extension fine-tune。

## 7. Pose 是明確選擇

資料準備、YAML 檢查與 CPU interface smoke 不等於正式 Pose 執行。本輪已收到明確 opt-in，
`authorization.pose=true` 與 `training.pose_enabled=true`，所以 C0～C3 都執行 Pose 訓練與 validation。
這項Pose授權只屬於本輪。Float20報告不自行選C_best；若未來重新核准後續流程，才依固定joint與近似
tie-break規則產生本輪C_best，且不取代yolo_combine的融合winner選擇。

本輪 C0-Control 與所有候選必須共用：

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

Float20 的 F1 threshold 只用 C0-Control screen search-val 決定一次，之後 C1～C3 在同一
screen search-val 固定套用；本輪不把 formal val 偷渡進初篩。

Stock Ultralytics Pose best.pt 使用 Pose mAP50-95 與 Box mAP50-95 的 combined fitness。本案若以
Pose mAP50-95 為研究主指標，runner 必須另外保存 pose-only best，combined fitness 仍另欄保留。
BBAT5 的非 COCO 兩點 schema 目前使用均勻 OKS sigma。Macro F1 是兩類 F1 的算術平均；Micro F1
則由官方 precision/recall curves 與各類 support 在固定 confidence 下估算合併 TP／FP／FN，報告必須
標示 `estimated_from_curves_and_supports`，不可冒充逐預測重新配對的 exact Micro F1。

Float20原始報告維持measurement-first：0.005/0.008只描述敏感度，C_best仍是null。後續接續run則依
使用者另行授權，把0.008套在C2/C3四項主要精度與成本改善gate；完整訓練結果用固定joint score，
0.008內視為近似，再依latency、GFLOPs、Macro F1與candidate ID決勝。

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

- OOM probe：C0先以 physical32跑完整macro；只有辨識為CUDA OOM時才清空probe，改用16重跑。選定值寫入
  `shared-controls/batch-plan.json`，C0～C3全矩陣共用；fallback16仍OOM就停止，不再偷偷降batch。
- Validation：Detect/Pose都固定16，不啟用stock auto-batch；記錄requested/effective batch、route、imgsz、
  GPU、step與錯誤。
- NaN/Inf：立即中止並保存有效 lineage/log；不要只替該候選改LR或loss。
- 中斷：每epoch保存同候選的完整training snapshot。resume前必須比對run YAML hash、candidate、microbatch、
  optimizer、scheduler、EMA與epoch；不一致就停止。
- Extension：只可從自己的未strip continuation state真正resume；不能用另一候選、另一parent或stock
  stripped `last.pt`／`best.pt`。
- missing/shape mismatch：完整保存 matched/missing/unexpected/shape-mismatch，不宣稱全量pretrained。
- frozen drift：任何 protected parameter、buffer、mode或head contract改變都中止。
- 不同recipe：公平性檢查失敗；先修正正式YAML，不能事後在報告中淡化。

## 11. 量化

C0仍是通用設定中的Q0/Q1候選；但本次run已封存，C1～C3、Q0/Q1/Q2L與完整Q2均不執行。
以下只記錄原量化模板的技術語意，不代表本 revision 有量化成果或執行授權：

- Conv weights：per-channel symmetric INT8。
- activations：per-tensor affine INT8。
- custom BinaryQK/PWL arithmetic：排除並列出。
- Q2L 從 Q1 calibrated lineage 建立新 optimizer，固定200 steps、前50 steps更新 observer、每50 steps驗證；
  lr0為parent的0.1、AMP關閉、最多3 epochs且不early-stop。
- Q0/Q1/Q2L/Q2 對每一個 AP/mAP/F1 指標分開記 gap/recovery。
- Q2L只回答短QAT能否恢復PTQ gap，不取代正式Q2；最終報告會分開記錄各stage精度與gap。
- 不宣稱實際部署latency。
- fake quant 仍是浮點 quantize-dequantize simulation；沒有 backend convert、operator coverage 與
  目標硬體驗證時，不稱為 Bit-True 或部署 INT8。

官方公開原則與本機 8.4.90 對照見
[docs/OFFICIAL_TRAINING_REFERENCE.md](docs/OFFICIAL_TRAINING_REFERENCE.md)。
