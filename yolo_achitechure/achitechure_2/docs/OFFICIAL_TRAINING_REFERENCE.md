# 官方訓練依據與本機 Ultralytics 8.4.90 對照

更新日期：2026-08-22
適用範圍：architecture_2 的 transfer／resume／freeze、正式 training YAML、Pose 指標與 loss，以及 W8A8 fake-quant simulation

本文件是查證紀錄，不是第二份實驗規格。實驗邊界、候選、公平性與執行授權一律以 [`EXPERIMENT_SPEC.md`](../EXPERIMENT_SPEC.md) 為準；實際值一律以通過驗收的 handoff 與產生出的 effective YAML 為準。

## 1. 證據分層與本機環境

本文固定使用三種標籤：

- **官方公開原則**：Ultralytics／PyTorch 官方文件或官方原始碼可以直接支持的行為。
- **本機觀察**：2026-08-22 對目前環境實際 import 路徑及原始碼所做的唯讀比對。
- **本案決策**：為了 architecture_2 因果比較、可追溯性或授權邊界而採用的限制；不得反稱為官方 recipe。

本機查證結果：

| 項目 | 結果 |
|---|---|
| Python 套件環境 | `/home/uxin/yolo/.venv/bin/python` |
| `ultralytics.__version__` | `8.4.90` |
| 實際 import 路徑 | `/home/uxin/yolo/yolo_p2/ultralytics/__init__.py`，是本機 Git fork，不是 site-packages 中的乾淨 wheel |
| 本機 fork revision | `ea920761b1c6d531c2231d0033714301690cf67d`，branch `5090` |
| `torch.__version__` | `2.11.0+cu128` |
| 官方 tag 基準 | 本機 Git object `v8.4.90`，commit `ad7661d7cf8b58c5f338dc213a9d3b6696ef489a` |

逐檔 `git diff --quiet v8.4.90 -- ...` 顯示，`default.yaml`、trainer、MuSGD、data build/base/augment、loss、Pose trainer/validator 與 metrics 和官方 tag 一致。`nn/tasks.py`、`nn/modules/head.py` 有 fork 的 Detect classification tower/parser 修改，因此不能只用版本字串宣稱整份本機套件等同官方 tag；本次涉及的 `PoseModel`、`Pose26` 區段未被該差異改寫。官方基準可直接查看 [Ultralytics v8.4.90 source tree](https://github.com/ultralytics/ultralytics/tree/v8.4.90/ultralytics)。

可重做此檢查的唯讀命令：

```bash
/home/uxin/yolo/.venv/bin/python -c \
  "import inspect, ultralytics, torch; print(ultralytics.__version__, inspect.getfile(ultralytics)); print(torch.__version__)"

git -C /home/uxin/yolo/yolo_p2 diff --quiet v8.4.90 -- \
  ultralytics/cfg/default.yaml \
  ultralytics/engine/trainer.py \
  ultralytics/optim/muon.py \
  ultralytics/data/base.py \
  ultralytics/data/build.py \
  ultralytics/data/augment.py \
  ultralytics/utils/loss.py \
  ultralytics/utils/metrics.py \
  ultralytics/models/yolo/pose/train.py \
  ultralytics/models/yolo/pose/val.py
```

## 2. Transfer、fresh start、resume 與 freeze

### 2.1 Transfer 與 resume 不是同一件事

| 類型 | 官方公開原則 | 本機 8.4.90 觀察 | 本案決策 |
|---|---|---|---|
| Pretrained transfer | 官方範例允許直接載入 `.pt`，或先由 YAML 建圖再 `.load()`／指定 `pretrained=...` 轉移權重。[官方 Train 範例](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/en/modes/train.md#L61-L85) | `setup_model()` 會先建構 task model，再把指定 weights 傳入 `get_model()`；標準 loader 只轉移名稱存在且 shape 相同的 tensors，這條路徑不會從舊 run 恢復 optimizer。[官方 trainer source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/engine/trainer.py#L749-L783)、[官方 weight-transfer source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/nn/tasks.py#L291-L311) | C0-Control、C1、C2、C3 都從同一個已驗收 C0-Handoff state dict 建立，並建立**新的 optimizer**。候選 graft 是本案 loader 契約，不等於一般 `YOLO(yaml).load(pt)` 已自動支持融合 graph；manifest 仍須列 matched、missing、unexpected、shape mismatch。 |
| Resume | 官方文件定義為從中斷 checkpoint 接續 weights、optimizer、LR schedule 與 epoch；`resume=True` 應指向未完成 run 的 checkpoint。[官方 resume 說明](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/en/modes/train.md#L201-L234) | `check_resume()` 取回 checkpoint args；`_load_checkpoint_state()` 恢復 optimizer、scaler、EMA、updates、best fitness；`start_epoch` 由 checkpoint epoch 推進，scheduler 以 `last_epoch` 對回進度，而不是載入一份獨立 scheduler state dict。[官方 trainer source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/engine/trainer.py#L906-L1032) | 只有通過 late-improvement gate 的 extension，才從**該候選自己的** `last` 真正 resume。不同候選、不同 handoff revision 或 `best`/`last` 之間不得混接 optimizer state。 |
| New fine-tune run | `resume=False` 會開始新的 training session；pretrained weights 可以只是新 session 的初始化。[官方說明](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/en/modes/train.md#L232-L234) | 新 trainer 重新建立 dataloader、optimizer 與 scheduler。[官方 setup source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/engine/trainer.py#L264-L288) | 主 Float 比較固定為 fresh optimizer；這是 C0-Control 與候選公平性的必要條件。 |

`best.pt` 適合拿來做模型選擇、下一個**新**階段的權重初始化；`last.pt` 是「繼續同一 run」的最近 epoch。官方 trainer 每 epoch 寫 `last`，當 fitness 相等時另寫 `best`。[官方 checkpoint source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/engine/trainer.py#L660-L718) 不能把「從 best 權重開始一個新 optimizer」寫成 `resume`。

#### 正常完訓後的 `last.pt` 不一定還能 resume

這是本機 v8.4.90 最重要的 resume 限制：stock trainer 在 `final_eval()` 會直接對 `last.pt` 與 `best.pt` 執行 `strip_optimizer()`。[官方 final_eval source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/engine/trainer.py#L887-L900) `strip_optimizer()` 會清除 optimizer、scaler、EMA、updates、best fitness，把 epoch 設為 `-1`，並以 FP16 inference model 覆寫 checkpoint。[官方 strip source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/utils/torch_utils.py#L762-L827)

所以「中斷中的 raw `last.pt`」可以真正 resume；「正常完訓並跑過 final_eval 的 stock `last.pt`」不行。本案若要在主訓練結束、通過 late-improvement gate 後真正 extension resume，runner 必須在 strip 前另存不可變的 unstripped resume checkpoint，並測試 optimizer/scaler/EMA/epoch 能完整重載。沒有這份 artifact 時，只能標示為從 weights 建立新 optimizer 的 extension fine-tune，不能在 manifest 寫 `transition.mode: resume`。

### 2.2 Freeze 的實際涵義

**官方公開原則：** `freeze=N` 凍結前 N 個 model layers；list 則凍結指定 layer indices。[官方 train argument](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/macros/train-args.md#L30-L34)

**本機觀察：** trainer 以 `model.{index}.` 名稱比對，把參數 `requires_grad=False`，且 `.dfl` 永久列入 freeze；若全部參數都被凍結會直接報錯。[官方 freeze source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/engine/trainer.py#L303-L331) 每次切回 train mode 時，命中的 `BatchNorm2d` 會再設為 eval，避免 running mean/variance 漂移。[官方 BN freeze source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/engine/trainer.py#L652-L658)

**本案決策：** 融合 winner 的實際 module paths 由 handoff 宣告，不能假設仍是共享 parser 的固定 layer indices。本案 freeze validator 必須同時檢查：

1. frozen parameters 沒有 gradient、值沒有改變；
2. frozen BN／其他 buffers 沒有改變；
3. inherited MASF／attention／PWL 保持 eval state；
4. 未選定的 task-specific branch 未被意外解凍。

這比單獨設定 `requires_grad=False` 更嚴格，是 architecture_2 的公平性決策，不是 Ultralytics 對任意自訂 module path 的自動保證。

## 3. Training YAML 常用欄位的精確語意

正式 YAML 使用 Ultralytics 公開鍵名 `batch`，不是 `batch_size`。`model_scale: m` 是本案模型規模 metadata；augmentation 的 `scale` 是影像幾何增強，兩者完全不同。

| 欄位 | 官方／原始碼語意 | 本機細節與陷阱 | 本案用法 |
|---|---|---|---|
| `batch` | 正整數是實際 batch；`-1` 是約 60% GPU 記憶體的 AutoBatch；`0 < batch < 1` 是指定 GPU 記憶體比例。[官方 batch 說明](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/en/modes/train.md#L260-L267) | DDP 會把 global batch 除以 world size。單 GPU 第一個 epoch 發生 CUDA OOM 時，本機 trainer 最多三次把 batch 減半，並重建 dataloader、optimizer、scheduler。[官方 OOM source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/engine/trainer.py#L470-L494) | 正式候選比較要記錄 requested 與 effective physical batch。若某一候選觸發自動減半，該 run 不可悄悄與未減半 run 當 matched ablation，必須重設共同可行 batch 或標示無效。 |
| `nbs` | 官方定義為 nominal batch size，預設 64。[官方 train arguments](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/macros/train-args.md#L44-L52) | trainer 用 `round(nbs / physical_batch)` 設定 gradient accumulation，並依 `batch * accumulate / nbs` 縮放 weight decay；warmup 期間 accumulation 還會漸變。[官方 accumulation source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/engine/trainer.py#L264-L287) | `nbs` 不是 VRAM 中的 batch。報告必須同時保存 physical batch、accumulation、optimizer steps 與各 task effective samples。 |
| `cache` | `False` 關閉；`True`／`ram` 快取到記憶體；`disk` 快取到磁碟。[官方 train arguments](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/macros/train-args.md#L8-L14) | 本機在 `deterministic=True` 且 `cache=ram` 時明確警告可能非 deterministic；`disk` 會把 `.npy` 寫在每張影像旁邊並先檢查該路徑可寫與空間。[官方 cache source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/data/base.py#L133-L145)、[官方 disk-cache source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/data/base.py#L280-L340) | 預設 `cache=false`。BBAT5 原始目錄唯讀，不能對原始影像啟用 `disk` cache；若因 I/O 調整，須使用可寫衍生位置並寫入 effective YAML／manifest。cache 只能改載入方式，不能改資料 membership。 |
| `fraction` | 只指定訓練資料使用比例，預設 1.0。[官方 train arguments](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/macros/train-args.md#L30-L34) | build path 對 train 傳入 `cfg.fraction`，對 val 強制 1.0；dataset 是取排序後 file list 的前 `round(N*fraction)`，不是每次隨機、分層或 grouped sampling。[官方 build source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/data/build.py#L235-L280)、[官方 subset source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/data/base.py#L162-L184) | 正式比較固定 `fraction=1.0`。若 CPU smoke 使用較小 fraction，只能當 plumbing，不得當精度排名，也不得用它另造 BBAT5 split。 |
| `scale` | RandomPerspective 的縮放增強；float `s` 代表從 `Uniform(1-s, 1+s)` 抽樣，tuple 則是絕對 `(min,max)`。[官方 augmentation source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/data/augment.py#L1080-L1127) | 例如 `scale=0.5` 是 0.5–1.5 倍，不是「YOLO m scale」，也不是 dataset fraction。[官方 augmentation argument](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/macros/augmentation-args.md#L6-L12) | 值由 handoff effective recipe 決定；C0-Control 與全部候選必須相同。 |
| `multi_scale` | 每個 batch 改變 `imgsz` 的範圍，與 RandomPerspective `scale` 是不同設定。[官方 train arguments](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/macros/train-args.md#L24-L29) | 設為 0 才是關閉 multi-scale training。 | 若 handoff 未明示，不可因看到 augmentation `scale` 就推定已開啟 `multi_scale`。 |

## 4. YOLO26 MuSGD：官方行為與本案界線

### 4.1 官方公開行為

Ultralytics 將 MuSGD 定義為 SGD 與 Muon-style orthogonalized update 的混合 optimizer；`param.ndim >= 2` 的 weights 進 Muon+SGD 路徑，較低維度參數走 SGD。`optimizer=auto` 在估計 iterations 大於 10,000 時選 MuSGD，較短 run 選 AdamW。[官方 MuSGD 說明](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/en/modes/train.md#L240-L256)、[官方 optimizer builder](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/engine/trainer.py#L1042-L1126)

MuSGD 的 Muon 分量會把 2D／展平後的 Conv gradients 以五步 Newton–Schulz 近似正交化，再與 momentum SGD 更新組合；非 Muon group 只走 SGD。[官方 MuSGD implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/optim/muon.py#L9-L251)

公開 `BaseTrainer.build_optimizer()` 在 v8.4.90 固定以 `muon=0.2`、`sgd=1.0` 建立 MuSGD。[官方 builder source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/engine/trainer.py#L1100-L1126) 但官方 YOLO26 checkpoint metadata 列出的內部 `muon_w`／`sgd_w` 對 M/S/L/X 是 `0.436`／`0.479`，而且這些不是 `default.yaml` 可設定的 public arguments。[官方 internal recipe](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/en/guides/yolo26-training-recipe.md#L146-L160) 因此即使 optimizer 名稱同為 MuSGD，也不能把 public trainer run 宣稱為官方 pretraining 的 optimizer 精確重製。

本機 builder 還會將符合 `layer 23 + cv3`（以及 semantic auxiliary head pattern）的群組設為 `3 × lr`。[官方 parameter-group source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/engine/trainer.py#L1100-L1120) 這個規則依名稱 pattern，不保證在融合後的動態 graph 仍指向同一語意 head。

### 4.2 官方 YOLO26m recipe 只能作參考

官方公開表列的 COCO YOLO26m checkpoint 配方包含 `MuSGD`、`lr0=0.00038`、`lrf=0.882`、`momentum=0.948`、`weight_decay=0.00027`、`warmup_epochs=0.99`、`epochs=80`、`batch=128`、`imgsz=640`。[官方 YOLO26 recipe 表](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/en/guides/yolo26-training-recipe.md#L87-L105) 官方同頁也明確指出，released checkpoints 是由中間 pretrained weights 與搜尋後配方精煉而成，且內部 branch 有 public code 未暴露的設定，因此無法用公開套件精確重製 pretraining。[官方 training overview](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/en/guides/yolo26-training-recipe.md#L15-L35)、[官方 reproduction boundary](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/en/guides/yolo26-training-recipe.md#L240-L246)

**本案決策：** 上述數字是 sanity-check 參考，不直接覆蓋融合 winner。architecture_2 必須：

- 從 handoff 取得完整 resolved optimizer recipe；缺值保持 blocked，不用官方表格猜值；
- 若 handoff 指定 MuSGD，就在 effective YAML 明寫 `optimizer: MuSGD`，避免 `auto` 隨 iteration 數改選 optimizer；
- 對每個 resolved graph 匯出 parameter group、參數數量、effective LR、weight decay、Muon/SGD 路徑；
- 不把「參考官方 recipe」寫成「重製官方 pretraining」。

## 5. Pose：指標、loss 與翻轉

### 5.1 Box AP、Keypoint AP 與 F1 必須分開

官方 Pose validator 同時輸出 Box 與 Pose 的 Precision、Recall、mAP50、mAP50-95；API 分別由 `metrics.box.*` 與 `metrics.pose.*` 取得，`maps` 是 per-class mAP50-95。[官方 Pose validation API](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/en/tasks/pose.md#L108-L131)

- `map50` 是 threshold 0.50 的 AP 平均；`map` 是 0.50 到 0.95、step 0.05 的平均。[官方 Metric properties](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/utils/metrics.py#L930-L999)
- Box matching 使用 IoU；Pose matching 使用 keypoint OKS-like `kpt_iou`。[官方 Pose validator](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/models/yolo/pose/val.py#L157-L184)
- 非 COCO `[17,3]` keypoint shape 會使用 `ones(nkpt)/nkpt` 作 sigma；因此本案 `[2,3]` 會是兩個 0.5，而不是 COCO17 sigmas。[官方 Pose init source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/models/yolo/pose/val.py#L95-L105)
- 官方 `PoseMetrics.fitness` 是 Pose fitness 加 Box fitness；而單一 `Metric.fitness` 只給 mAP50-95 權重 1，所以這個 combined fitness 等於 pose mAP50-95 與 box mAP50-95 的和，不是 pose-only AP。[官方 fitness source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/utils/metrics.py#L1009-L1012)、[官方 PoseMetrics source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/utils/metrics.py#L1497-L1524)
- 官方 `ap_per_class()` 使用第一個 matching threshold（Box 是 IoU 0.50；Pose 是 OKS 0.50）建立 P/R/F1 curve，再在**各次 validation 自己的** smoothed mean F1 curve 上取最大值，回傳同一 confidence 點的 per-class P/R/F1。[官方 F1 selection source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/utils/metrics.py#L835-L889)

**本案決策：** 報告分開保存 COCO Box、BBAT5 Pose Box、BBAT5 Keypoint 的 mAP50 與 mAP50-95，以及 per-class AP。combined fitness 另存但不得冒充 pose-only 指標。Macro F1 是固定 threshold 下 per-class F1 的算術平均；Micro F1 必須先跨類別聚合 TP／FP／FN 再計算，stock API 沒有可直接代用的同名 scalar。F1 confidence threshold 只能由 C0-Control search-val 決定一次，再固定套用全部候選與 formal val，不能直接比較每個 run 各自最佳 F1 點。

若直接沿用 stock PoseTrainer，`best.pt` 依上述 combined fitness 保存；若研究流程要保存 pose-only mAP50-95 最佳 checkpoint，就必須另外實作與測試 checkpoint selection，而不能只把報告欄位改名。

### 5.2 Pose loss 與 RLE 是否真的 active

官方 v8.4.90 defaults 公開 `box=7.5`、`cls=0.5`、`dfl=1.5`、`pose=12.0`、`kobj=1.0`、`rle=1.0`。[官方 default.yaml](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/cfg/default.yaml#L93-L111)

YOLO26 是 DFL-free architecture；官方 recipe 明確說明，YOLO26 的 `dfl` 設定鍵改用來加權 normalized box-distance L1 loss，不是傳統 Distribution Focal Loss。[官方 YOLO26 loss 說明](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/en/guides/yolo26-training-recipe.md#L111-L121) 因此報告應保存實際 criterion 名稱與 component 語意，不能只看到鍵名就寫成 DFL。

本機與官方 source 的實際條件是：

1. 一般 `v8PoseLoss` 有 box、pose location、keypoint visibility/objectness、cls、dfl 五項；`pose` 與 `kobj` gains 套在 keypoint loss 上，且 `kobj` 只有 keypoint shape 具有 visibility/objectness 維度時才會產生。[官方 v8PoseLoss](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/utils/loss.py#L644-L799)
2. YOLO26 end-to-end PoseModel 才選 `PoseLoss26`；非 end-to-end 仍選 `v8PoseLoss`。[官方 PoseModel criterion](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/nn/tasks.py#L669-L707)
3. `PoseLoss26` 只有在 head 存在 `flow_model` 時才建立 RLELoss，且 prediction 還必須提供 `kpts_sigma`，使每點最後維度成為 4（xy+sigma）或 5（xyv+sigma），才能形成有效 RLE 項。[官方 PoseLoss26](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/utils/loss.py#L802-L975) 官方 Pose trainer 也只在 `flow_model` 存在時把 `rle_loss` 加入 loss names。[官方 Pose trainer](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/models/yolo/pose/train.py#L96-L105)
4. 官方 `Pose26` head 建立 RealNVP flow 與 sigma heads；fuse 後會清除 training-only flow/sigma modules。[官方 Pose26 head](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/nn/modules/head.py#L666-L760)

**本案決策：** defaults 只用來檢查 key 是否受支持；實際 loss gains 繼承 handoff effective recipe。只看到 YAML 中 `rle: 1.0` 不足以宣稱 RLE 生效，dry-run／manifest 必須證明 criterion、`flow_model`、`kpts_sigma` output、非零有限的 RLE loss 與有限 gradient 都 active；否則把它列為 `accepted-but-inactive` 並 fail closed，而不是靜默報告 RLE 訓練。

### 5.3 `fliplr` 與 keypoint mirror mapping

官方 Pose dataset 文件指出，對具有左右對稱語意的 keypoints，`flip_idx` 應定義翻轉後的 index 對應。[官方 Pose dataset YAML 說明](https://github.com/ultralytics/ultralytics/blob/v8.4.90/docs/en/datasets/pose/index.md#L40-L54) 本機 8.4.90 若 keypoint dataset 沒有 `flip_idx` 而 `fliplr`／`flipud` 大於 0，會直接把兩者設為 0 並警告；mapping 長度不等於 keypoint 數則報錯。[官方 transform source](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/data/augment.py#L2756-L2774)

**本案決策：** BBAT5 的兩個 keypoint 在 `kpt_names` 與 mirror mapping 未完成語意驗證前，effective recipe 明寫 `fliplr=0`、`flipud=0`，不依賴 runtime 靜默覆寫。未來只有在資料語意確認、更新正式 spec/handoff 並重做公平比較後才可開啟。

## 6. Fake-quant simulation 的能力與不能宣稱的事

### 6.1 PyTorch 官方定義

PyTorch `FakeQuantize` 的定義是在 training time 模擬 quantize→dequantize：以 observer 算出的 scale／zero-point 做 round、clamp，再轉回浮點表示；`observer_enabled` 與 `fake_quant_enabled` 可分開控制。[PyTorch 2.11 官方 API](https://docs.pytorch.org/docs/2.11/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)、[PyTorch v2.11 source](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/ao/quantization/fake_quantize.py#L128-L260)

`prepare_qat` 的角色是準備 QAT graph，`convert`／`convert_fx` 才依 mapping／backend config 把觀察與 fake-quant graph 轉為量化形式；兩步不是同一件事。[PyTorch 2.11 `prepare_qat` API](https://docs.pytorch.org/docs/2.11/generated/torch.ao.quantization.prepare_qat.html)、[`convert` API](https://docs.pytorch.org/docs/2.11/generated/torch.ao.quantization.convert.html)、[`convert_fx` API](https://docs.pytorch.org/docs/2.11/generated/torch.ao.quantization.quantize_fx.convert_fx.html)

因此 fake quant 能測量的是數值擾動與訓練恢復能力；單靠 fake quant 不能證明：

- graph 已使用整數 kernel；
- export backend 支援全部 operators；
- checkpoint／模型檔已縮成真正 INT8；
- 目標 5090、CPU、TensorRT 或其他硬體有 INT8 latency、throughput、VRAM 或能耗收益。

這些限制直接來自 FakeQuantize 仍執行 quantize-dequantize simulation、而非 `convert` 後 backend graph 的官方定義。

### 6.2 architecture_2 的 Q0／Q1／Q2 邊界

| Stage | 本案實際工作 | 允許的結論 | 禁止的結論 |
|---|---|---|---|
| Q0 | 建立 fused FP32 reference，檢查融合前後等價性 | FP32 graph reference 與數值差距 | INT8 已部署 |
| Q1 | 固定 calibration batches 收集 observers，凍結 range，啟用 W8A8 fake quant | PTQ-style simulation 的精度差距 | backend PTQ artifact、INT8 latency |
| Q2 | 在 fake quant 開啟時重新訓練，之後用相同 validation 比較 | QAT simulation 對 Q1 的精度恢復 | 已完成 `convert`、可部署 QAT model |

本案目前只包裝標準 `Conv2d`：weights 使用 per-channel symmetric qint8、axis 0；activations 使用 per-tensor affine qint8。BinaryQK／PWL／HardwareFriendlyAttention 等 custom roots 排除在 fake-quant scope 外。因此結果必須標示 `simulation_only=true`，並附 quantized／unquantized node 清單。即使標準 Conv 的 Q1/Q2 精度良好，也不能推論被排除的 custom arithmetic 已具有整數等價性。

本模擬也沒有逐項證明 target backend 的 rounding、accumulator width、saturation、requantization 與 custom op 次序，因此不得稱為 backend `bit-true`；「Float EMA 轉 Bit-True copy」若由上游 handoff 提供，必須是另一個有明確算術契約與 regression evidence 的 validator，不能用本案 FakeQuantize 結果代替。

要升級為「部署 INT8」至少還需要：明確 target backend、可轉換 graph、operator coverage、conversion/export artifact、fresh-process correctness、相同資料的正式精度驗證，以及目標硬體 latency／memory profiling；目前都不在 Phase A 可宣稱範圍內。

## 7. Effective YAML 與 run manifest 的查核清單

在 handoff 通過、且使用者另行授權正式訓練前，以下項目不得留給 CLI 臨時猜測：

- `transition.mode` 是 `fresh` 還是 `resume`；resume 必須指向同一候選自己的未 strip continuation checkpoint，stock 正常完訓的 `last.pt`／`best.pt` 必須拒絕。
- physical `batch`、`nbs`、實際 accumulation、optimizer steps、各 task effective samples。
- `fraction`、`cache`、augmentation `scale`、`multi_scale`；不得混淆 `model_scale=m`。
- optimizer 名稱與完整 LR／momentum／weight decay／warmup；MuSGD 必須輸出實際 parameter groups 與 effective LR。
- frozen module paths、parameters、buffers、BN state 與 task branch。
- Pose 的 box／pose／kobj／cls／dfl／rle gains，以及每個 loss component 是否 active。
- Box/Pose 的 mAP50、mAP50-95、per-class AP、固定 threshold 的 per-class／Macro／Micro F1；combined fitness 另欄保存。
- 量化 stage、observer/fake-quant 狀態、quantized/unquantized scope，以及 `simulation_only=true`。

若官方文件、目前本機 fork 與 handoff recipe 互相衝突，先停止執行並把差異寫入 config-check／run manifest；不得用官方 default 悄悄覆蓋 handoff，也不得用 handoff 值反稱官方公開 recipe。
