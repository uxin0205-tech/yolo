# 2026-08-29：Full35 Activation 收尾、權重發布與量化交接

## 任務與範圍

依使用者要求，把本子專案從數學設計、資料規範、實驗順序、訓練配置、正式／失敗／中止結果到權重
lineage 全部統整，準備以 commit subject `5090 Finish 0829` 發布 GitHub。此次不恢復 qSiLU finalist
訓練；將 activation-only 階段凍結為可供下一條 quantization 工作列讀取的交付。

正式資料入口保持不變：

- COCO2017 Detect：`/home/uxin/yolo/coco2017.yaml`，118,287 train／5,000 val。
- BBAT5 Pose／box branch：`/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml`，
  5,964 formal train／683 formal val。
- BBAT5 registry：`/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。
- `fraction=1.0`、`resampling=false`；沒有改動樣本、labels、split，也沒有建立 30% 資料版本。

## 變更內容與原因

1. 新增 `docs/research/full35-activation-mathematical-derivations.md`，完整推導 registry 六種 activation。
   控制組包含 SiLU 偶殘差、Hardswish 分段式與 ReLU；proposed 部分包含 SIPA 一般四約束線性解、
   `poly_quality`／`poly_shift` 的逐系數展開、qSiLU 截斷平方基底與 exact-tail 三條件、各區段展開，
   以及 fixed-point `A(n)-A(-n)=n` 的整數證明。
2. 新增最終中文分析報告、JSON 與 CSV。JSON 保留 full Float／Bit-True zero-shot 數值、八項 selector、
   11-region sensitivity、queue、OOM probes、profiling 與權重 lineage；CSV 提供 12 個關鍵節點的八項
   mAP50-95／delta 寬表。
3. 新增 `scripts/export_full35_results.py`，從本機原始 artifacts 重建 committed JSON／CSV，避免人工
   抄寫誤差；用 `/tmp` 重生成並以 `cmp` 證明兩份報告逐位元可復現。
4. 發布兩個 inference 權重：accepted SiLU 與完成 10 epochs、通過 gate 的 qSiLU。每個原檔
   106,825,541 bytes，因超過 GitHub 100 MB 單 blob 上限，切成 90,000,000 + 16,825,541-byte 兩片；
   新增 `weights.json`、`SHA256SUMS` 與安全重建程式。沒有發布中止 qSiLU finalist、425 MB resume
   checkpoints 或已被 gate 淘汰候選。
5. 新增 release contract tests，防止後續把 provisional qSiLU 標成 final、改變關鍵 pass/fail，或
   產生超過 GitHub 單檔上限的分片。
6. 同步子專案 README、研究索引、Full35 配置 README、報告索引與工作紀錄索引，使最終結論、數學、
   權重及量化入口可從根層一路追蹤。
7. 在根層 `.gitignore` 僅放行 `yolo_activation/release/weights/`，讓已核准且有 checksum 的分片可提交；
   重建出的 `*.pt` 仍由全域規則排除。
8. 機械格式化 `scripts/full35_queue.py` 的既有長路徑表達式；沒有改變 selector 或 queue 語意。

## 最終實驗結論

- 靜態 queue：`5 completed / 0 pending / 14 blocked`。14 個 blocked 是 `poly_shift` uniform
  prerequisite 失敗後的預註冊 gate 傳播，不是執行錯誤。
- 10-epoch recovery：qSiLU worst delta `-0.008635`，八項通過；Hardswish `-0.016884`、
  `poly_quality -0.020970`、`poly_shift -0.030138`，均未過。
- SiLU finalist control：第 6 個已完成 epoch early-stop，gate-passing best worst delta `-0.012872`。
- qSiLU finalist：只完成 epoch 1，並在 epoch 2 macro 106 後依使用者要求中止；epoch 1 provisional
  worst delta `-0.015966`，沒有 completion marker，不作最終勝負或量化父權重。
- qSiLU physical batch 128／64／32 均在第一個 forward OOM；16 可穩定完成。logical Detect batch
  仍為 128，靠 gradient accumulation 保持 macro 語意。
- 後續先跑 activation × quantization PTQ 分析，只有超出相同八項預算才做 matched QAT；目前不繼續
  activation-only 調參或補跑被 gate 封鎖的 policy。

## 驗證方式與結果

- `/home/uxin/yolo/.venv/bin/python -m pytest`：`61 passed`；4 個既有 legacy ONNX exporter
  deprecation warnings，無功能失敗。
- `python -m ruff check .`：`All checks passed!`。
- `python -m ruff format --check .`：`64 files already formatted`。
- `scripts/validate_phase0.py`：六種函數報告完成；qSiLU／`poly_shift` 的 symmetry、exact tails、Q16.10
  與 ONNX gates 通過。
- `scripts/activation_training.py toy-dry-run`：manifest、policy replacement 與 finite output 通過。
- `scripts/full35_activation.py preflight`：`ready=true`、`blockers=[]`；資料比例 1.0、正式路徑與父權重
  hash 正確。TensorBoard 未安裝只影響額外 UI，不影響 JSONL／CSV／PNG；base config 的 VRAM warning
  已由本輪真實 qSiLU probes 與 per-job physical microbatch override 明確處理。
- 報告重生成：`reports/*.json/csv` 與 `/tmp` 重建檔 `cmp` 一致；JSON parser 通過。
- 權重：四個分片 `sha256sum -c` 全部 OK；兩個 `/tmp` 重建 `.pt` 分別回到原 SHA
  `d67fb45...ec74c` 與 `767918...190e`。
- Markdown：26 份文件的相對鏈接檢查，`missing=[]`；字面 `\\n` 殘留檢查為空。
- 訓練程序：唯讀 process 查核沒有本專案 queue/train；未重啓 GPU 作業。

## 困難與解法

1. 修改既有 Markdown 時，`apply_patch` 再次因環境 `bwrap: loopback: Failed RTM_NEWADDR` 失敗。
   依既有工作流程改用舊內容必須唯一命中的精確 Perl replacement。
2. 第一次 Perl fallback 使用不插值的 `q{}`，在研究／訓練 README 寫入字面 `\\n`；立即唯讀檢查發現，
   精確轉換為真實換行並再次檢查全專案無殘留。README 多段 replacement 第一次命中數為 0，未寫入；
   改用 `qq{}` 後三段必須全部唯一命中才完成。
3. 初次完整 Ruff 找到新增 exporter 的 executable bit、import order 與三個格式問題；設定 CLI 可執行、
   自動整理 imports 與機械格式化後重跑完整檢查，全數通過。
4. 權重單檔超過 GitHub blob 限制且倉庫沒有可驗證的 Git LFS remote；採用無損分片、逐片 hash 與最終
   full-file hash，既能一般 Git push，也保留逐位元可追溯性。

## 清理與保留

- 沒有刪除本機 29 GB 原始 experiments、OOM probes、quarantine 或 checkpoints；它們受 `.gitignore`
  排除並保留作稽核證據。
- 發布只包含 source/config/tests/docs、約 51 KB 正規化報告及約 214 MB 的兩個必要權重分片。
- `/tmp` 的報告與重建權重只用於驗證，發布前移除，不影響本機正式 artifacts。

## 未解事項或風險

- qSiLU seed-1 finalist 未完成，seed 2／claim ablation 未執行；不得宣稱多 seed 最終優越性。
- 尚未完成 PTQ、QAT、全輸入碼 Q8/Q12/Q16 枚舉與 compiler/HLS/RTL bit-exact 比對。
- 沒有指定 target FPGA／ASIC、clock、precision 與 synthesis flow，尚無 latency、power、LUT/DSP/BRAM
  板上結論。
- qSiLU eager PyTorch 的 OOM 不代表 fused kernel 的極限；未來若優化 kernel 必須重新實測。
