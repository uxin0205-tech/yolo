# architecture_2 統一實驗規格

規格版本：`1.2.0`
狀態：唯一正式規格
生效日期：2026-08-19

本文件是 `achitechure_2` 的唯一規格來源，統一架構、訓練、選型、Pose、量化、資料與報告規則。
「必須／禁止」是強制條件，「應」是預設條件，「可」是選項。本文件的門檻是本研究的工程門檻，
不是通用研究標準。

## 1. 規格優先順序與版本追蹤

優先順序固定如下：

1. `EXPERIMENT_SPEC.md`：唯一正式定義。
2. `configs/**/*.yaml`：由本規格衍生的可執行設定。
3. `TRAINING_GUIDE.md`：理由、轉換與異常處理。
4. `RUNBOOK.md`：經驗證的實際命令。
5. `plan.pdf`、`YOLO26_Codex_Research_Protocol.md`：歷史與官方參考，不是第二份真相。

`plan.md` 只指向本文件。規則變更時，先修改本文件並升版，再更新 YAML。舊結果保留當時的
`spec_version` 與 `spec_sha256`，禁止回溯改寫。

所有正式 architecture、training、dataset、quantization 與 fusion YAML 都必須保存
`spec_version`、`spec_sha256`。每個 run manifest 與 checkpoint lineage 還必須保存 architecture YAML、
實際 training YAML、dataset YAML、parent checkpoint 的 SHA256。CLI 只允許覆寫
`model/name/project/device/workers/cache`；未知欄位、hash 漂移或偷改 learning fields 一律停止。

### 修訂歷史

| 版本 | 日期 | 內容 |
|---|---|---|
| 1.0.0 | 2026-08-18 | 建立 Detect、Pose、量化、設定驗證、grouped split 與 provenance 的統一規格。 |
| 1.1.0 | 2026-08-18 | 全面中文化使用者文件；Pose 改為使用者明確選用；新增未決的雙來源融合研究契約。 |
| 1.2.0 | 2026-08-19 | 訓練設定改為每階段一份完整正式 YAML；顯式列出 batch/fraction/scale/cache 等常用參數；新增設定目錄與單檔啟動介面；統一標準 schema 並修正 Pose26 head 契約。 |

## 2. 專案用語與繼承邊界

- **Float-PWL parent**：由 `architecture_1` 選出的可續訓 Float checkpoint，MASF variant 為 `full35`
  或 `partial75`。每個候選都必須獨立從同一份 parent 建立。
- **C0**：不改 C3k2 的 reference，保留 inherited MASF、BinaryQK 與 PWL attention。
- **候選（Candidate）**：相對 C0 只包含一個已宣告架構差異的模型，禁止從另一候選初始化。
- **C_best**：依第 5 節規則選出的 eligible 單因子候選；它不是 C0，也可能不存在。
- **Bit-True copy**：目前 Float EMA 的深拷貝，只轉換 inherited attention normalization。
- **新階段（Fresh stage）**：從上一階段 best weights 建立新 optimizer 與 scheduler。
- **續訓階段（Resume stage）**：從 `last.pt` 恢復 optimizer、scheduler 與 epoch。
- **Pose 路線（Pose route）**：選用的下游能力驗證，不是 Detect 主線的自動後續步驟。
- **雙來源融合研究（Dual-source fusion study）**：未來以來源 A／B 的權重與架構為輸入的獨立研究。
  本版只定義來源 provenance，不決定 weight interpolation、ensemble、routed switch 或 staged transfer。

Inherited graph 是本機 Ultralytics `8.4.90` 的 YOLO26m，包含 P3 MASF、BinaryQK、Float/Bit-True PWL
attention、head inputs `[16,19,22]`、strides `[8,16,32]`、`end2end=true`。MASF 與兩個 attention roots
永久保持 eval-mode frozen，包含 parameters 與 buffers。Full35/Partial75 的比較屬於 `architecture_1`，
本專案不重跑。

正式執行仍需要通過 `architecture_1` handoff gate，內容包含 Float/Bit-True checkpoints、hashes、metrics、
環境版本與 model-selection manifest。開發 fixture 不能成為正式 C0。

## 3. 架構候選矩陣

Baseline factors：`e=0.5`、`inner_n=2`、`kernel_mode=3x3_3x3`、`use_rep=false`。

| ID | 唯一架構變因 | 目標 layers | 執行條件 |
|---|---|---|---|
| C0 | 無 | 無 | 必跑 reference |
| C1 | `e: 0.5 -> 0.375` | 6、8、13、19 | 必跑 |
| C2 | `inner_n: 2 -> 1` | 6、8、13、19 | 必跑 |
| C3 | `3x3_3x3 -> 1x1_3x3` | 6、8、13、19 | 必跑 |
| C3-P5 | 只在 layer 8 套用 C3 kernel delta | 8 | C3 drop > 0.008 才可執行 |
| R1 | C2 factors 加 `RepBottleneck` | 6、8、13、19 | C2 為 CONDITIONAL 且 latency 或 GFLOPs 改善 >=8% |

Layers 2、4、16 保持 upstream；layer 22 因包含 inherited attention 而排除。1.2.0 禁止 C1+C2、
C1+C3、C2+C3 與其他組合。`e=0.25` 只有在 C1 drop <=0.005 且仍需壓縮時，才能列為未來候選。

每份 `configs/candidates/*.yaml` 都是 machine-readable architecture contract。Local builder 驗證後 graft，
並輸出 `standalone_loadable:false`、`builder:achitechure_2` 的 Ultralytics-like snapshot。Snapshot 只供
review/report，禁止傳給共享 `YOLO(yaml)` parser。

### 3.1 Architecture Delta

以下效果在量測前都是假設。Transfer report 必須列出 matched、missing、unexpected、shape-mismatch tensors。
Detect 與 Pose 共用 backbone/neck delta，但使用各自 task head。

| ID | Baseline -> Modified | 改變 | 不變 | 預期影響 |
|---|---|---|---|---|
| C0 | parent -> 完整深拷貝 | 無 | 全 graph | 結構與成本不變，所有相容 tensors matched |
| C1 | `e=.5 -> .375` | 6/8/13/19 hidden width | protected layers、MASF、attention、topology、head | Params/FLOPs/VRAM/latency 降低；可能損失 small-object capacity |
| C2 | `inner_n=2 -> 1` | 6/8/13/19 inner depth | CSP shell、channels、kernels、protected modules、head | 成本與有效深度下降；可能損失 context |
| C3 | `3x3_3x3 -> 1x1_3x3` | 6/8/13/19 第一個 inner conv | depth、width、CSP/residual、protected modules、head | 成本與 receptive field 降低；可能損失 spatial feature |
| C3-P5 | 只改 layer 8 | P5 backbone C3k2 | 其他 backbone/neck/scales 與 head | 成本收益較小，但 small-object 風險較低 |
| R1 | C2 Bottleneck -> RepBottleneck | 6/8/13/19 剩餘 block | C2 depth/width/kernels、protected modules、head | training 成本可能提高；fusion 必須 `max_abs_diff <=1e-4` |

## 4. 正式 YAML 與 config-check 契約

`configs/catalog.yaml` 是所有可用設定的機器可讀目錄；人類導覽位於 `configs/README.md`。正式設定分為：

- `configs/candidates/*.yaml`：一份檔案代表一個架構候選。
- `configs/training/detect/*.yaml`：D0、D1、D2、Q2 各一份完整訓練設定。
- `configs/training/pose/*.yaml`：P0–P4 各一份完整訓練設定。
- `configs/data/*.yaml`：Ultralytics dataset 設定與本案 provenance。
- `configs/quant/*.yaml`：量化模擬範圍與參數。
- `configs/fusion/*.yaml`：未來雙來源融合契約；本版不啟用。
- `configs/schema/*.schema.yaml`：標準 JSON Schema 文件。

訓練設定的正式外部 interface 是「每個階段一份完整 YAML」。禁止要求使用者自行合併 canonical、candidate
overlay 與 stage fragment。每份 `formal_training` YAML 必須完整保存 task、stage、中文用途、candidate policy、
execution policy、dataset、checkpoint transition、ranking contract、永久凍結模組與所有 forwarded Ultralytics
training arguments，至少顯式列出 batch、fraction、scale、cache、device、workers、optimizer、LR、augmentation、resume 與 validation 相關欄位。這些檔案由 `achitechure_2` launcher 讀取，`ultralytics_yaml_direct=false`，不可直接當成
共享 `YOLO(yaml)` 的模型架構檔。

推薦啟動方式是 `train --config <formal-training.yaml> --candidate <ID> ...`。舊的 `--task/--stage` 介面可作
相容入口，但必須解析到同一份正式 YAML，不能建立第二套設定。Runtime 仍只允許覆寫
`model/name/project/device/workers/cache`。

Schemas 位於 `configs/schema/`，全部採 JSON Schema Draft 2020-12 格式。`config-check` 必須：

- 確認本機 Ultralytics 恰為 `8.4.90`，且 `training` 下的 keys/values 可被本機 `get_cfg` 接受。
- 驗證 catalog 引用的檔案存在且完整，不允許漏列或引用未宣告的正式設定。
- 比對候選 factors、targets、protected layers、parent policy、triggers、Pose26/Detect heads 與禁止組合。
- 確認每個 training YAML 是完整單檔，dataset、task、stage、transition、ranking 與 execution policy 一致。
- 明確列出 accepted-but-inactive、optional routes 與 runtime-overridable keys。
- 對未知欄位、不支援值、缺候選、hash 漂移或 learning override fail closed。
- 驗證 dataset、quantization 與 fusion template 仍引用同一 spec。

`rle` 在 parser 層可接受，但只有 Pose head 含 `flow_model` 才 active；正式 Pose 啟動時再次檢查。

## 5. Detect 訓練與選型

Detect 使用 COCO2017、`imgsz=640`、physical `batch=16`、`nbs=64`、seed 0、AMP、deterministic 與
明確 MuSGD。這是本專案基準，不宣稱完整重製未公開的官方 pretraining。

- **D0**：獨立 3–5 epoch smoke，不參與排名。
- **D1**：最多 100 epochs、patience 20。
- **D2**：只有 D1 未 early-stop，且 best epoch 在 85–100，或 81–100 rolling best 比 61–80 至少提高
  0.001，才從 D1 `last.pt` resume 到總 epoch 140、patience 15。

每次 validation 都以 Float EMA 建立 Bit-True copy。Bit-True box mAP50-95 決定 best 與 early stopping。
正式結果必須 fresh-process reload。Manifest 保存 requested/resolved args、optimizer groups、best/last、
best epoch、stop reason、Params、FLOPs、latency、peak memory、hardware 與完整 hashes。

相對 C0：drop <=0.005 為 PASS；0.005 < drop <=0.008 為 CONDITIONAL，且只有 latency 或 GFLOPs 改善
>=8% 才 eligible；drop >0.008 為 REJECT。C3 rejection 才能觸發 C3-P5；eligible conditional C2 才能
觸發 R1。R1 沒有 fusion report 不得選為 C_best。C_best 依狀態、drop、latency、GFLOPs、Params 排序。
沒有 eligible candidate 時 C_best 為空，量化主線停止。

## 6. Pose：由使用者決定是否執行

Pose 是 **選用路線**，`enabled_by_default=false`。建立 grouped dataset 不會建立 Pose model，也不會開始
Pose training。只有使用者明確執行 `build-pose`、`train --task pose` 或 `validate-bittrue --task pose`，且同時提供 `--enable-pose --execute` 才能進入 Pose 路線。
任何自動化流程禁止因 Detect 完成而自動啟動 Pose。

若使用者選擇執行，正式比較仍是 C0 對 recorded C_best，兩者使用相同 grouped data、Pose26 head seed、
stages、optimizer groups 與 budget。未來官方 `yolo26m-pose.pt` 只記為 `P_official`，不參與因果排名。

Pose source 位於 `../../original/pose/bbt5.v1i.yolov8`，保持唯讀。Derived data 依檔名 `.rf.` 前 prefix
分組，以 seed 0 做 grouped 90/10 train/val；images 使用 symlink、labels 使用 copy。所有同源增強圖必須
在同一 split，禁止 leakage。觀察到的 4 個負座標只在 derived labels 裁切為 0，並寫入 patch manifest。
不建立不存在的 test split。因 keypoint semantics 未被命名，`fliplr=0`；只有新 revision 記錄
`kpt_names` 與 mirror mapping 後才能修改。

PoseLoss26：`box=7.5`、`cls=0.5`、`dfl=1.5`、`pose=12`、`kobj=1`、`rle=1`。

- **P0**：獨立 3 epoch smoke，不銜接 P1。
- **P1**：head-only，最多 10、patience 5。
- **P2**：freeze layers 0–10，neck+head，最多 20、patience 8。
- **P3**：除 MASF/attention 外完整 fine-tune，最多 100、patience 20。
- **P4**：通過 late-improvement gate 才從 P3 `last.pt` resume，總 epoch 130、patience 10。

P1->P2、P2->P3 使用上一階段 best weights 與新 optimizer/scheduler；只有 P3->P4 是 resume。
Pose mAP50-95 是 research best metric；官方 Pose+Box combined fitness 逐 epoch 另存。

## 7. 未來雙來源融合研究

本版不執行融合，只提供 `configs/fusion/source-pair.template.yaml` 保存：

- source A/B 的角色、checkpoint path/hash、architecture YAML path/hash、task/head contract。
- dataset/spec/training provenance。
- 尚未決定的 `fusion_mode` 與 `switch_policy`。

使用者未來必須先選擇融合語意：權重插值、ensemble、runtime routed switch 或 staged transfer。選定後必須
新增 spec revision、獨立候選 ID、baseline、測試與報告；禁止把融合結果混入目前 C0/C_best 單因子排名。

## 8. 量化

只對 C0 與 recorded C_best 執行 Q0 fused FP32、Q1 calibrated W8A8 PTQ fake quant、Q2 10–15 epoch
fake-quant fine-tuning（patience 5、0.1x architecture LR）。Observers 在 epochs 1–3 更新。Conv weights
採 per-channel symmetric INT8，activations 採 per-tensor affine INT8；MASF Conv 納入，BinaryQK/PWL custom
arithmetic 排除並列出。所有結果標示 `simulation_only=true`。報告 PTQ gap、QAT gap、recovery。

## 9. 驗證與報告

自動測試必須涵蓋 spec/YAML parity、single factors、conditional triggers、snapshots、Detect/Pose
forward/loss/gradients/strides/head inputs、frozen state、transfer/reload、stage transitions、group leakage、
MuSGD groups、Pose RLE、Bit-True selection、fusion source provenance 與 quantization regression。

本次交付不執行正式長訓練。正式完成後，`REPORT.md` 必須回答 C1/e=.25、C2 對 C3、conditional
fallback/recovery、C0/C_best quantization；Pose 只有在使用者選擇執行時才列入正式報告。
