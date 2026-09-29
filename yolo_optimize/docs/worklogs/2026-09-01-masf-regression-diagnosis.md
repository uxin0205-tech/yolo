# 2026-09-01：MASF 精度回歸原因診斷

## 任務與範圍

分析既有 YOLO11m／YOLO26 MASF 實驗為何沒有超過 baseline，分離 P2 graph/head、MFAM、
訓練排程、初始化擾動、尺度／類別 trade-off 與 seed variance，並提出可驗證的解法。

本輪只做既有結果與程式碼稽核、synthetic-input CPU differential probes 及一手文獻查核；
沒有啟動 GPU training/validation，沒有修改模型、權重或任何資料集。

## 變更內容與原因

1. 新增 `docs/research/2026-09-01-masf-regression-diagnosis.md`：

   - 修正 P2-PaperFormula 相對 B0 的錯誤單因子歸因。
   - 記錄舊 P3 MFAM 初始化、完整 graph 與 trained-checkpoint activation 差分。
   - 分開說明 YOLO11m 非 identity-safe MFAM 與 YOLO26 identity-safe residual MASF。
   - 排列已證實原因、待驗證機制及最小解法矩陣。

2. 更新 `scripts/audit_accuracy_regressions.py`：

   - 加入 P2-Direct control，分別輸出 P2 graph/head 與 MFAM 邊際效果。
   - 加入 P3-Partial25 seeds 42/43 paired delta，避免只看兩 seed mean。
   - 保留原 fail-closed 回歸狀態，但把訊息改為「最佳含 MASF 架構未超過 B0」，不再暗示
     `-0.008189` 全由 MASF 單獨造成。

3. 更新子專案 README、研究索引及工作紀錄索引，提供本報告入口。

## 主要結果

### P2 因果拆解

- B0 → P2-Direct：`-0.014104`，代表新增 P2 graph/head 與其最佳化的合併效果。
- P2-Direct → P2-PaperFormula：`+0.005914`，MFAM 在相同 P2 family 內其實回補精度。
- P2-PaperFormula → B0：合計 `-0.008189`，不能全算成 MASF 退化。

### P3 初始化與完整 graph

- 舊 legacy/full MFAM 模組初始化 relative L1 約 `1.057`、cosine 接近 0，並非 identity-safe。
- 舊 Partial25 初始化 relative L1 `0.263360`。
- 現行 YOLO26 Full35／Partial75 residual 初始化 relative L1 只有 `0.001305`／`0.000323`。
- 完整 YOLO11m P3-Partial25 相對 B0 的 layer 16 feature relative L1 `0.301194`；
  decoded prediction relative L1 `0.185187`。

### Trained checkpoint 與最佳化

- P3 Full 類 trained branches 對 P3 feature 的 relative L1 約 `1.55–1.78`，cosine 約 `0.05–0.14`。
- P3-Partial25 兩 seeds 約 `0.31`、cosine 約 `0.80`，也是 P3 family 最佳者。
- P3-Partial25 相對 B0：seed 42 `-0.032639`、seed 43 `+0.010316`，方向相反。
- P3-Lite35-F7／Partial50 seed 42 都在 epoch 1 最佳、epoch 31 早停，seed 43 則可訓練到
  epoch 49–53 最佳，證明強烈 seed/optimization interaction。

### 現行 YOLO26

- 同機 COCO A0／Full35 A2／Partial75 A2 為 `0.506754`／`0.506391`／`0.506754`；
  差異在 `0.001` tie band，沒有 material regression 或 gain 證據。
- Full-data Phase B 的 Full35／Partial75 都下降約 `0.002889`／`0.002745`，較像共同
  unfreeze schedule／parent drift；現有證據缺 matched no-MASF retraining control。
- J3 `best_joint` 的 `p3_masf.alpha=0.110659`，不能直接拔除，先做同 checkpoint
  `alpha=0` 完整八項 validation。

## 驗證方式與結果

### 回歸迴路

~~~bash
python3 scripts/audit_accuracy_regressions.py
~~~

更新前後都穩定重現 MASF、BinaryQK 三個紅燈，exit code `1`。更新後新增以下輸出：

- B0 → P2-Direct `-0.014104`。
- P2-Direct → P2-MFAM `+0.005914`。
- P3-Partial25 seed 42／43 `-0.032639`／`+0.010316`。

exit code `1` 是預期的 fail-closed 訊號，不是腳本錯誤。

### CPU differential probes

- 固定 seed 模組級 probe：完成，舊 Full 約 `105%` relative L1 擾動；YOLO26 residual 低於
  `0.14%`。
- 完整 YOLO11m graph P3-Partial25 probe：完成，feature/prediction relative L1 為
  `0.301194`／`0.185187`。
- 10 個正式 P3 best checkpoints：使用 `torch.load(weights_only=True)` 與顯式 safe-global
  allowlist 完成，沒有使用 `weights_only=False`。
- training curve、per-class AP50/AP75/AR/AP_S 與 per-seed paired delta：均由既有正式
  JSON/CSV 唯讀計算。

### 文件與程式

- 使用 `ast.parse` 做無 pycache 語法檢查：`syntax-ok`。
- Ruff check 使用 `--no-cache`：`All checks passed!`。
- Ruff format 首次找到兩個過長輸出行；執行 formatter 後重查：`2 files already formatted`。
- 7 份 Markdown 相對連結檢查：`broken_links=0`。
- 尾端空白與 merge-conflict marker 掃描：無命中；`rg` exit code `1` 代表沒有找到問題。

## 困難與解法

1. 預設 sandbox 多次回報 `bwrap: loopback: Failed RTM_NEWADDR`：使用核准的 escalated
   唯讀命令；文件修改仍只透過 `apply_patch`。
2. 首次一次載入多個完整模型的 probe 沒有回傳結果，疑似程序資源／buffer 問題：縮成
   `128×128` 固定輸入並一次比較一個 variant，成功取得差分。
3. native checkpoint 首次 `weights_only=True` 因自訂類別未 allowlist 而拒絕：先用
   `get_unsafe_globals_in_checkpoint` 列出類別，再只 allowlist repo 與 Ultralytics 已知類別，
   成功載入；沒有改用不安全反序列化。
4. 首次新增研究報告的 patch 終止標記多一個 `+` 而被完整拒絕：修正 patch 格式後重新套用，
   未留下半成品。
5. Ruff formatter 建立可重建的 `.ruff_cache/`：最終檔案檢查發現後，以精確路徑刪除；
   沒有移除任何原始碼、報告、模型、資料或實驗結果。

## 資料集紀錄

- 沒有讀取或修改 BBAT5 影像、labels、split、cache 或 registry。
- 所有 CPU probes 使用固定 synthetic tensors；資料數字只讀既有歷史正式報告。
- 後續所有新 BBAT5 detection、pose 與融合實驗仍只能使用
  `/home/uxin/yolo/original/pose/derived/bbat5-v1/` 的既定 assignment 與
  `configs/detect.yaml`／`configs/pose.yaml`，不得重新切分。

## 未解事項或風險

1. 尚未執行 J3 `alpha=0` 的完整八項 validation，所以不能宣稱可無損移除現行 MASF。
2. 尚未以 canonical BBAT5 跑 matched no-MASF／zero-gated MASF 三 seed 實驗。
3. 「高頻被平滑」只有 class/size AP 與 activation 間接證據，仍缺真實影像 feature-spectrum probe。
4. 兩 seed historical study 只能定位不穩定來源，不能形成穩定泛化結論。
5. 本輪只提出解法，沒有實作或訓練任何 MASF 變體。
