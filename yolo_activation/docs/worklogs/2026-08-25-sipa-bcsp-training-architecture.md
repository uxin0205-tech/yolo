# 2026-08-25：SIPA-BCSP Activation 訓練架構

## 任務與範圍

依使用者要求，說明 proposed activation 是否使用對稱性與特殊演算法優化，並把已成立的數學／
硬體概念完善成可執行的訓練架構資料夾。本輪建立 delivery contract、activation manifest、模型
安全替換、靜態 placement、硬體成本向量、受限 Pareto search、CLI 與 toy integration；因 baseline
資料夾尚未交付，沒有載入真實 YOLO checkpoint、啟動 COCO2017／BBAT5 訓練或產生 AP。

資料使用狀態：沒有讀寫影像或 labels、沒有建立 Runtime Dataset View、沒有重切／抽樣／合併
資料。Canonical BBAT5 v1 Detect 與 COCO80 Detect 只作不可混用的 machine-readable contract。

## 變更內容與原因

### 1. SIPA 對稱 activation

- 將既有 integral-polynomial family 明確命名為內部工作名稱 `SIPA`，不是新穎性宣告。
- 實作已使用 `A(x)=x/2+H(|x|)`，正負輸入共用同一 even residual；不是兩側各存一份係數。
- `poly_shift` 保留 `a=9/2,T=8`、exact-ReLU tails、四次資料乘法與 dyadic/APoT constant path。
- Q16.10 complementary rounding、0 LSB symmetry/tail 與 ONNX forbidden-op gate 沿用既有已驗證
  implementation。

### 2. BCSP placement 演算法

- 新增內部工作名稱 `BCSP`：不以任意加權總分混合 AP 與成本，而是對 `map_loss`、可選
  `ap_s_loss`、可選 target latency 及結構成本向量計算 non-dominated Pareto frontier。
- Planner 每次只產生下一個 stage：baseline reproduction → uniform zero-shot → equal-budget
  recovery → `poly_shift` region sensitivity → Pareto beam expansion → 最多二個 finalists。
- Beam 只從 frontier policy 增加一個 region，受 `beam_width`、最多 changed regions、最多三個
  deployment kernels 共同限制；避免 `K^L` 暴力枚舉。
- ReLU 固定為 cheap control；`poly_quality` 固定為 uniform accuracy ablation；region search 預設只
  擴展 `poly_shift`，降低完整訓練數量。
- seed 1 是必要 stage，seed 2 只在 finalists 啟用。兩個資料集各自維護 observations/frontier，
  不平均 raw AP。

### 3. 深模組與安全 seam

- 新增 `src/activation_lab/training/`，外部以 `TrainingArchitecture.inspect/apply/plan/audit_delivery`
  四個 interface 使用大量 implementation；此設計依 `codebase-design` 的 deep-module、locality 與
  interface-as-test-surface 原則。
- `inspect()` 只盤點原生 `nn.SiLU`，未匹配與多重 region match 直接阻擋；output sigmoid、softmax
  與其他 activation 不會因名稱猜測被替換。
- Manifest 必須 `reviewed=true`；`apply()` 在任何 mutation 前一次性檢查所有 path 仍存在且仍為
  SiLU。預設 deepcopy，確保 baseline model 不受候選污染。
- Policy 支援 uniform、region 與精確 site override，但 site override 不可指定 ineligible path。
- `cost_weight` 可於交付後填入 activation elements/profile 統計；未提供時只作 site-count proxy。

### 4. Training 資料夾

- 新增 `training/README.md`：記錄 symmetry datapath、BCSP state flow、介面、契約、資料規則、指令
  與未完成邊界。
- 新增 `training/configs/pipeline.yaml` 與 `region-rules.example.yaml`。
- 新增 delivery、manifest、observations YAML 範本與四份 JSON Schema。
- Delivery audit 要求 model/source/checkpoint/hash/version、兩資料集各自 fingerprint/baseline
  metrics/checkpoint/recipe；placeholder、全零 hash、錯誤 canonical YAML 或缺失路徑都不能通過。
- 新增 `training/.gitignore`，排除 run、output、weights 與 deployment binaries。

### 5. CLI 與 serialization

- 新增 `scripts/activation_training.py` 三個 subcommands：`audit-delivery`、`compile-plan`、
  `toy-dry-run`。
- Manifest 與 TrainingPlan policy 已提供 YAML/JSON round-trip；plan 中 region/site assignments 輸出
  為 mapping，可原樣放入下一輪 observations，不會出現 tuple/list 格式不相容。
- 現有 `ultralytics` import 指向 sibling `yolo_p2`，不是未來交付 baseline 的確認來源；因此沒有
  建立會載錯模型的 production adapter。資料夾到位後依其 framework commit 再補真實 adapter。

## 驗證方式與結果

1. `/home/uxin/yolo/.venv/bin/python -m pytest`：結果 `26 passed`；覆蓋既有 activation、
   fixed-point/ONNX，以及 training manifest、transactional replacement、dataset guard、BCSP 全 stage
   progression、delivery audit、JSON schema、region regex、plan/observation 與 manifest round-trip。
2. `scripts/activation_training.py toy-dry-run`：exit 0；合成 detector 的 `early/deep/neck/head` 四個
   SiLU 全部替換成 `poly_shift`，原 model 仍為 SiLU、輸出 shape `[1,3,8,8]` 且 finite；初始計畫
   僅為兩資料集各自的 baseline reproduction。沒有讀資料集或 checkpoint。
3. `audit-delivery --delivery training/contracts/delivery.example.yaml --no-check-files`：預期 exit 2；
   成功拒絕所有 `REPLACE_WITH`／絕對 placeholder path、全零 hash 與 null baseline mAP。
4. `compile-plan --manifest training/contracts/activation-manifest.example.yaml`：預期 exit 2；成功拒絕
   空且未 reviewed manifest，不產生任何 experiment。
5. Canonical dataset tests 確認 BBAT5 必須使用正式 `nc=2 detect.yaml`，COCO80 必須使用
   `/home/uxin/yolo/coco2017.yaml`；plan 為兩者產生不同 experiment 與 YAML。

6. `/home/uxin/yolo/.venv/bin/python -m ruff check src tests scripts` 輸出 `All checks passed!`；
   `ruff format --check` 輸出 `19 files already formatted`。

7. 依 `finish-work` 完成末段稽核：完整核對根層 AGENTS、domain vocabulary、兩份 accepted ADR 與
   BBAT5 dataset contract，沒有衝突；權重、ONNX、engine、log、temporary、cache、run 與 output
   inventory 皆為空，`git diff --check -- .` 通過。未執行 commit、push、下載、GPU job 或真實訓練。

## 困難與解法

- Sandbox 持續發生 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`：必要命令改用經
  核准、限定工作區或 `/tmp` 的執行。
- `apply_patch` 對既有檔案讀取仍受同一 helper 故障阻擋：新檔案均以 `apply_patch` 建立；既有
  檔案在嘗試失敗後使用精確、完全匹配的 `perl -0pi/-pi` 小範圍替換，並立即唯讀核對。
- Region YAML 使用單引號時最初保留了多餘反斜線，且 one-to-many pattern 可能涵蓋
  `one2one_cv2`：改為正確單反斜線，並要求 `cv2/cv3` 是獨立 path segment；六個代表 path 測試
  均只匹配一個 region。
- 第一版 TrainingPlan 以 dataclass `asdict()` 輸出 tuple assignments，與 observations schema 的
  mapping 不一致：在 deep module 內集中正規化為 mapping，加入 YAML round-trip test。
- 最終 CLI 演練第一輪把全域 `--config` 放在 subcommand 後，並誤把 delivery 參數寫成
  `--contract`，因此 argparse 正確回傳 exit 2：以 `--help`／原始 parser 契約核對後，改成
  `activation_training.py --config ... <subcommand>` 與 `audit-delivery --delivery ...` 重跑；toy
  dry-run 為 exit 0，兩個故意未完成的契約案例則如設計為 exit 2。
- 沒有真實 baseline adapter：這是已知且刻意保留的阻擋，不使用 sibling Ultralytics 版本規避。

## 未解事項或風險

- 尚未收到 baseline folder，production model loader、strict checkpoint restore、實際 region mapping、
  train/val/export adapter 與完整 recipe 仍無法驗證。
- 現有硬體 cost 是 activation-site proxy；未取得 tensor elements、memory traffic、fusion、target
  compiler 與實測 latency 前，不能宣稱 BCSP 選出硬體最快 policy。
- `max_map_loss/max_ap_s_loss` 目前為 `null`；須在 pilot 後、正式比較前由使用者與 baseline 尺度
  凍結，不得事後調門檻。
- `poly_shift` 的 curve／fixed-point 成績尚未轉換為 COCO2017 或 Canonical BBAT5 v1 AP 證據。
- SIPA、BCSP 與組合名稱只是工程工作名稱；prior-art 報告已指出 symmetry、PWL/PoT 與 YOLO
  placement 均有相鄰工作，不能單憑本架構宣稱 novel。
