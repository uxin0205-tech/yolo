# 2026-09-01：YOLO26 P3 MASF 無實質增益專項診斷

## 任務與範圍

針對使用者設計於 YOLO26 P3 的 Full35／Partial75 MASF 為何沒有提高精度，檢查正式圖結構、
checkpoint dependency、尺度／類別 AP、訓練 scope 與資料目標，並提出可驗證的解法。

本輪屬診斷，不修改模型實作、不啟動 GPU training/validation、不改 checkpoint 或資料集。

## 變更內容與原因

1. 新增 `docs/research/2026-09-01-yolo26-p3-masf-no-gain-diagnosis.md`：

   - 記錄目前 in-place graft 會同時改變 P3/P4/P5 的圖結構證據。
   - 記錄正式 Full35／Partial75 A2 checkpoint 的 CPU alpha-on／alpha-zero 差分。
   - 拆解 canonical COCO overall、AP_S/M/L、sports-ball 與 baseball-bat 差值。
   - 區分已確認根因、高機率機制及尚未證實的「高頻平滑」假設。
   - 提出 Detect-only fork、learned selector、matched control 與 P2→P3 lateral 的優先順序。
   - 補上 production-oriented before/after pseudocode、舊 checkpoint 相容邊界、計算圖 Jacobian、
     stride/channel 資訊推導、論文差異與必要 regression tests。

2. 更新 `scripts/audit_accuracy_regressions.py`：

   - 加入目前 YOLO26 Full35／Partial75 正式 CSV。
   - 加入最佳 MASF 必須達 `+0.001` 的 material-gain red gate。
   - 輸出 canonical overall、AP_S、sports-ball、baseball-bat 相對 A0 的差值。

3. 同步根 README、研究索引、工作紀錄索引與既有 MASF 診斷的專項報告入口。

4. 沒有修改 `yolo_achitechure/achitechure_1` 的原始碼、模型、權重或正式報表；所有輸入均唯讀。

## 驗證方式與結果

### 1. 快速紅燈

讀取 `final/reports/full35-partial75-ap.csv`，以 `+0.001` material-gain gate 比較同機 COCO internal：

- A0：`0.506753642`。
- Full35 A2：`0.506391251`，delta `-0.000362391`。
- Partial75 A2：`0.506754491`，delta `+0.000000849`。
- 最佳 MASF 未達 `+0.001`，red loop 穩定重現。

### 2. 正式圖結構

使用 `torch.load(weights_only=True)` 與明確 allowlist 載入兩個正式 Float A2 checkpoint，列出
layer 15–23 的 `from` 與 class：

- layer 16 是 `C3k2P3MASFFull35`／`C3k2P3MASFPartial75`。
- layer 17 `from=-1`，所以 MASF output 直接進 P3→P4 downsample。
- layer 20 同樣把已受影響的 P4 傳到 P5。
- Detect layer 23 讀取 `[16, 19, 22]`。

結果確認「P3 MASF」不是 Detect-only P3 branch，而是 P3/P4/P5 的共同上游。

### 3. CPU alpha dependency

固定 torch seed `20260901`、輸入 `1×3×128×128`，同 checkpoint 先正常 forward，再只把記憶體內
`alpha` 暫設為零 forward；完成後恢復 alpha，沒有保存 checkpoint。

| 模型 | alpha | P3 layer16 | P4 layer19 | P5 layer22 |
|---|---:|---:|---:|---:|
| Full35 A2 | +0.170532 | 0.271428 | 0.174071 | 0.085438 |
| Partial75 A2 | -0.361084 | 0.210857 | 0.234279 | 0.098079 |

表內是 alpha-on 相對 alpha-zero 的 feature L1 dependency。另一個 probe 得到：

- Full35：MASF output/input relative L1 `0.295976`、cosine `0.952471`、模型輸出 dependency
  `0.118066`。
- Partial75：relative L1 `0.241157`、cosine `0.863623`、模型輸出 dependency `0.120828`。

這排除 branch 沒進 forward、alpha 沒學到或作用太小；synthetic probe 不當成 AP 證據。

### 4. Canonical COCO 指標拆解

相對 A0：

- Full35：overall `-0.000223`、AP_S `-0.001226`、sports-ball `-0.001871`、baseball-bat
  `-0.002900`。
- Partial75：overall `+0.000107`、AP_S `+0.001559`、sports-ball `+0.002763`、baseball-bat
  `-0.005135`。

Partial75 有局部 small／ball 訊號，但被 bat 與其他目標抵銷；Full35 沒有同樣收益。

### 5. 訓練 scope 稽核

`phases.py` 的 `parameter_role()` 先辨識 `.p3_masf.`，A1/A2 確實只解凍 MASF；Phase B 解凍
MASF + layer 11 之後的 Neck/Detect 並凍結 attention。optimizer 另檢查 trainable parameter identities。
未發現「MASF 參數沒有進 optimizer」的程式錯誤。

Full-data Phase B 相對各自 A2 的 delta 為 Full35 `-0.002889`、Partial75 `-0.002745`；兩者都 rollback。
這支持共同適應沒有成功，但因缺 matched no-MASF control，不能單獨歸因於 scheduler 或 MASF。

### 6. 文件檢查

- `scripts/audit_accuracy_regressions.py` 重新執行後穩定輸出 4 個紅燈；exit code `1` 是預期
  fail-closed 結果。
- AST parse：`syntax-ok`。
- Ruff check：`All checks passed!`；formatter 首次修正長行後，format check 為
  `1 file already formatted`。
- 初次 6 份 Markdown 相對連結檢查：`broken_links=0`；追加 before/after 與推導後再查 5 份交付
  文件，仍為 `broken_links=0`、`text_problems=0`。
- 7 份本輪檔案的尾端空白／conflict marker 檢查：`text_problems=0`。
- 研究索引、工作紀錄索引、根 README 與既有 MASF 診斷入口已同步。

### 7. Detect-entry CPU prototype

只在記憶體中把 MASF owner 從 `model.16` 移到 `model.23 Detect`，沒有保存模型：

- parent state tensors 全數保留；新 state path 為 `model.23.p3_masf.*`。
- gate=0 時 P3 enhancement 與 raw P3 bit-exact，完整模型 output 也與 parent bit-exact。
- gate 開啟後，P3 Detect feature 改變，但 layer 16 raw P3、layer 19 P4、layer 22 P5 均 bit-exact。
- 同 parent、同 Partial75 權重、`alpha=0.2` 的 A/B 中，現行 seam 的 P4/P5 feature change 為
  `0.006724 / 0.004886`；Detect-entry 為 `0 / 0`。
- 兩個插入位置的完整 output 相對差為 `0.043793`，證明 routing 本身是 material computation change。

這驗證 proposed seam 的結構性質，不是 accuracy validation；尚未跑任何 GPU AP。

### 8. 原論文一手查核

查核 arXiv `2504.18136`：原 MASF-YOLO 使用 YOLOv11-s／VisDrone2019，包含 P2 detection layer、
backbone MFAM、shallow/deep Fusion、IEMA、DASI。MFAM 描述四種尺度輸出與 `k=7,9` strip
convolutions；本地只保留 P3 DW3/DW5 residual，是不同變體。

論文 ablation 的 mAP50:95 依序約為 baseline `0.294`、P2 `0.307`、P2+MFAM `0.319`、完整系統
`0.329`。因此論文證據不能直接外推成單一 YOLO26 P3 block 的預期增益。

## 困難與解法

1. 系統 `python` 別名不存在：改用 `/usr/bin/python3` 做 CSV 計算，並使用
   `/home/uxin/yolo/.venv/bin/python` 載入與正式環境一致的 PyTorch `2.11.0+cu128`、Ultralytics
   `8.4.90`。
2. 專案套件未安裝到共用環境：唯讀設定 `PYTHONPATH` 指向 `achitechure_1/src` 與
   `yolo_attention/src`。
3. checkpoint 含自訂類別：先用 `get_unsafe_globals_in_checkpoint` 列出類別，限制 prefix 為
   `torch`、`ultralytics`、`achitechure_1`、`yolo_attention`，再以 `weights_only=True` 載入；沒有使用
   `weights_only=False`。
4. `apply_patch` 讀取既有工作區檔案持續回報
   `bwrap: loopback: Failed RTM_NEWADDR`：確認環境有 `CODEX_SANDBOX_NETWORK_DISABLED` 後，只對
   `apply_patch` 子程序使用 `env -u CODEX_SANDBOX_NETWORK_DISABLED`，成功恢復既有檔案更新；外層仍
   使用受控權限，patch 不使用網路，也沒有改用 `cat`、重導向或其他寫檔方式。
5. 共用 venv 的 Ultralytics 實際來源是 editable 的 `/home/uxin/yolo/yolo_p2/ultralytics`，不是預期的
   `site-packages` 實體檔：改用 `inspect.getfile/getsource` 取得運行中 `Detect.forward`、
   `forward_head` 與 `BaseModel._predict_once`，避免讀錯版本。
6. 首次 same-weight A/B 用 Detect `forward_pre_hook` 比較 P3，shared-seam 量到 MASF 後 tensor，
   Detect-entry 卻量到內部 MASF 前 tensor，因邊界不同而非 bit-exact：改在兩個 MASF module 的
   input/output hooks 比較，確認輸入與輸出皆 bit-exact；P4/P5 routing 結論不變。

## 資料集紀錄

- 沒有讀取或修改 BBAT5 影像、labels、split、cache 或 registry。
- ball 尺寸統計與歷史 BBT5 指標只讀既有報告，不建立新 dataset view。
- 後續新 BBAT5 detection、pose、融合實驗只可使用不可變
  `/home/uxin/yolo/original/pose/derived/bbat5-v1/` 與正式 configs，不得重新切分。

## 未解事項或風險

1. 尚未對正式 J3 執行 alpha-on／alpha-zero 的完整八項 GPU validation。
2. 尚未用 matched no-MASF control 與至少三個 paired seeds 驗證 Detect-only fork。
3. 尚未執行真實影像 feature-spectrum／ball-centered activation probe，因此高頻平滑仍是假設。
4. A0 未用同一 staged schedule 重訓，現在的微小 delta 不能作純架構因果結論。
5. 本輪只提出 production-oriented 修改設計並做記憶體 prototype，尚未修改
   `yolo_achitechure/achitechure_1` 或建立新 checkpoint lineage。
