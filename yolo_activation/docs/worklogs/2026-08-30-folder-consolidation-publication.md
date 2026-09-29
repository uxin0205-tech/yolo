# 2026-08-30：Activation 子專案完整整理與重新發布

## 任務與範圍

依使用者要求，整理 `yolo_activation/` 全部已追蹤內容，統一繁體中文、重建文件入口、把 LaTeX 密集的
數學式改寫成可直接閱讀格式，重新驗證結果與權重，最後以 commit subject `5090 Finish 0830`
發布 GitHub。範圍不包含其他 YOLO 子專案，也不啟動新的訓練、PTQ 或 QAT。

Canonical BBAT5 v1、COCO2017、formal split、影像、標註、父 checkpoint 與既有 artifacts 均未改動。

## 變更內容與原因

1. 新增 `docs/README.md` 作為文件總導覽，清楚區分目前權威結論、數學、machine-readable 結果、
   訓練配置、權重、工作紀錄與歷史研究。
2. 重寫子專案 `README.md`，加入一分鐘結論、資料夾用途、資料契約、GitHub 發布邊界與驗證指令。
3. 重寫 `reports/README.md` 與 `docs/research/README.md`，只把目前整合報告列為權威入口；舊分析與
   Phase 0 文件明確歸為歷史或預註冊資料。
4. 重寫 `training/README.md`，分開說明 SIPA activation、BCSP placement 支援與未完成的 mixed-policy
   搜尋，避免把 BCSP 支援架構誤寫成已驗證搜尋成果。
5. 修復 `docs/research/README.md` 與 `training/full35/README.md` 中真的字面 `\n` 排版缺陷；Full35
   README 尾段同步成 2026-08-30 最終 queue、finalist、OOM 與量化交接狀態。
6. 重寫 `docs/research/full35-activation-mathematical-derivations.md`：
   - 不再使用 `\frac`、`\quad`、`\int` 或單行壓縮限制式。
   - 先解釋每個符號與 C0／C1／C2，再逐步推導六種 activation。
   - SIPA 四個限制各自拆開說明，並列出一般線性方程、係數解、積分後曲線與 C2 接合檢查。
   - qSiLU 保留截斷平方、三個 exact-tail 抵消條件、四段實際公式、dyadic 分解與誤差。
   - fixed-point invariant 改成逐行 floor／ceil 證明。
7. 重寫權威整合報告的數學章節，使用普通分數、Unicode 次方、表格與中文說明；移除主要兩份文件
   中所有 LaTeX 反斜線指令。
8. 對全部已追蹤 Markdown、Python、YAML、JSON、CSV 與 TOML 執行 ICU `Hans-Hant` 機械式繁體
   轉換；程式識別字、schema key、路徑、hash、數值與二進位權重不變。
9. 為原始需求規格與 2026-08-29 詳細分析加入歷史狀態提示，避免舊文中的「尚未收到模型」被當成
   目前狀況。
10. 更新母專案 README，連到新文件導覽與權威整合報告。
11. 更新權重 README，記錄本次四分片校驗與兩組完整權重重建結果；沒有重複加入權重 blob，也沒有
    發布失敗候選、中止 finalist 或 optimizer checkpoint。

## 驗證方式與結果

- `python -m pytest`：`61 passed`；4 個既有 legacy ONNX exporter deprecation warnings。
- `python -m ruff check .`：全部通過。
- `python -m ruff format --check .`：README Python 範例修正後，71 份檔案格式正確。
- `scripts/validate_phase0.py`：六種 activation curve／hardware proxy 重新產生；qSiLU 與
  `poly_shift` 的 symmetry、exact tails、Q16.10 證據維持通過。
- `scripts/activation_training.py toy-dry-run`：manifest、planner、policy replacement、原模型不變與
  finite output 全部通過。
- `scripts/full35_activation.py preflight`：`ready=true`、`blockers=[]`、`fraction=1.0`、
  `resampling=false`，父權重與正式資料入口解析正確。
- `scripts/export_full35_results.py`：透過唯讀 artifacts 重新輸出到 `/tmp`，JSON 與 CSV 均和提交版
  `cmp` 逐位元相同。
- 權重：`sha256sum -c SHA256SUMS` 四個分片全部 OK；兩組 `/tmp` 重建權重均為 106,825,541 bytes，
  SHA-256 分別為 `d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c` 與
  `7679186695317e431cd7deb17289f426f4b39b7a4993e4548e74f5ba2766190e`。
- 繁體中文機械檢查：所有納入文字檔再次經 `Hans-Hant` 後皆逐位元不變，`not_hant=[]`。
- UTF-8／Markdown：32 份文件、106 個相對連結全部有效；無亂碼、無控制字元、無行尾空白；
  主要報告與數學文件無 LaTeX 反斜線指令。
- `git diff --check`：通過。
- 本次沒有啟動 GPU 訓練、PTQ、QAT 或硬體 synthesis。

## 困難與解法

1. 環境的 `bwrap: loopback: Failed RTM_NEWADDR` 仍使 `apply_patch` 無法更新部分既有檔案。新文件與
   完整替代內容先由 `apply_patch` 建立，再精確取代舊檔；少量唯一命中文字使用 UTF-8 Perl replacement。
2. 第一次把新數學章節插入權威報告時，Perl 輸出層未明確指定 UTF-8，立即被亂碼掃描發現。從上一次
   已發布的乾淨報告副本復原，改用 `-Mutf8 -CSD` 重做；後續 `mojibake=[]`。
3. 結果重建第一次多寫一層相對路徑，唯讀 artifacts symlink 未建立，匯出器因找不到輸入而停止。
   移除空白 `/tmp` 目錄後以正確路徑重跑，JSON／CSV 逐位元一致。
4. Ruff 會檢查 Markdown 內的 Python code fence；新訓練 README 的單行 import 被判定需格式化。改成
   標準多行 import 後，完整 format check 通過。
5. 一次整合命令從臨時母專案根目錄執行 Ruff，掃到其他不在本次範圍的子專案既有問題；沒有修改
   那些專案，改從 `yolo_activation/` 根目錄重跑，`ruff check` 與 71 份格式檢查全部通過。

## 清理盤點與處理

| ID | 路徑 | 類型 | 大小／數量 | 決定 |
| --- | --- | --- | ---: | --- |
| C01 | `.pytest_cache/` | 可重建 cache | 約 36 KB | 未取得指定刪除授權，保留 |
| C02 | `.ruff_cache/` | 可重建 cache | 約 60 KB | 未取得指定刪除授權，保留 |
| C03 | `artifacts/` | 原始 run／queue／OOM／稽核證據 | 約 29 GB／4,800 檔 | 保留；多份正式結果仍需 provenance |
| C04 | `release/weights/` | 量化交付 | 約 204 MB／兩組權重 | 保留並重新驗證 |

本次驗證建立的 artifacts symlink、`/tmp` JSON／CSV 與兩個重建權重副本已精確移除。沒有刪除本機正式
artifacts、資料、checkpoint、權重、歷史文件或使用者檔案。

## 未解事項或風險

- qSiLU 20-epoch seed-1 finalist 仍是人工停止狀態；seed 2 未執行。
- BCSP 沒有 mixed-policy 或 normalized-regret 搜尋結果。
- PTQ、QAT、Q8／Q12／Q16 全輸入碼與 compiler／HLS／RTL bit-exact 比對尚未執行。
- 沒有指定 FPGA／ASIC target、clock、precision 與 synthesis flow，尚無板上硬體效能證據。
- C01、C02 仍保留，等待使用者明確授權後才可刪除。
