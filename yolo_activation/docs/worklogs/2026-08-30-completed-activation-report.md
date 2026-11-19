# 2026-08-30：已完成 Activation 權威整合報告

## 任務與範圍

依使用者要求，重新整理本專案實際完成與使用過的 activation，重寫一份可獨立閱讀的權威報告並準備
上傳 GitHub。範圍包含 SiLU、Hardswish、ReLU、`qsilu_pq`、`poly_quality`、`poly_shift`，
以及 SIPA 已完成成果與 BCSP 尚未完成的邊界。

本次只整理、修正與驗證既有成果；沒有啟動訓練、驗證 GPU job、PTQ 或 QAT，也沒有改動資料。
COCO2017 與 Canonical BBAT5 v1 的影像、標註、formal split 及 registry 均維持不變。

## 變更內容與原因

1. 新增 `reports/completed-activation-integrated-report.md`，以「實際完成狀態」重新編排六種
   activation，而不是按歷史實驗時間排列。
2. 報告新增完成矩陣，逐項標示實作、zero-shot、10-epoch recovery、finalist、權重與目前定位。
3. 整合六種 activation 的核心公式、共同 symmetry invariant、qSiLU 四段係數、SIPA 一般四約束、
   `poly_quality`／`poly_shift` 直接式與 fixed-point invariant。
4. 以同一份 machine-readable JSON 重建 baseline、五種 zero-shot 與四種 recovery 的八項 delta
   表，避免只報 overall AP 或手抄部分指標。
5. 明確區分 SIPA 與 BCSP：SIPA 已完成數學、實作與真實 Full35 實驗；BCSP 完成支援架構，但
   `poly_shift` prerequisite 失敗後 14 個 region／mixed jobs 被 gate 封鎖，沒有搜尋結果。
6. 把 `poly_quality` 的證據範圍收緊為實際完成的 PyTorch float reference；不把它誤寫成已有
   fixed-point／ONNX gate。SIPA 的 Q16.10／ONNX 證據屬於 `poly_shift`。
7. 同步更新母專案 README、子專案 README 與 reports README，讓新報告成為第一個人類閱讀入口。
8. 稽核舊數學文件時找到 `poly_shift` 式 (21) 的兩個 `\frac` 被寫成 form-feed、`\quad`
   遺失反斜線；精確修復三處。公式數值與程式實作從未改變，修正的是 Markdown／LaTeX 顯示。

## 驗證方式與結果

- 以 `reports/full35-activation-results.json` 自動核對：
  - 六種 activation 與 SIPA／BCSP 術語全部存在。
  - 八項 accepted baseline 數值全部存在。
  - 五種 zero-shot 的全部八項 delta 均以六位小數出現在新報告。
  - 四種 10-epoch recovery 的全部八項 delta 均以六位小數出現在新報告。
  - 結果為 `missing_values_or_terms=[]`。
- 掃描所有 Markdown 控制字元：修正前只找到數學文件兩個 form-feed；修正後
  `control_characters=[]`。
- Markdown 相對連結檢查：27 份文件，`missing=[]`。
- `python -m pytest`：`61 passed`；4 個既有 legacy ONNX exporter deprecation warnings，無功能失敗。
- `python -m ruff check .`：全部通過。
- `python -m ruff format --check .`：69 份 Python 檔案格式正確。
- `scripts/validate_phase0.py`：六種 activation 的 curve／hardware proxy 重新產生；qSiLU 與
  `poly_shift` 的 symmetry、exact tails、Q16.10 evidence 維持通過。
- 沒有本專案訓練程序因本次報告工作而啟動。

## 困難與解法

1. 修改既有 README 時，`apply_patch` 再次被環境
   `bwrap: loopback: Failed RTM_NEWADDR` 阻擋。先保留失敗證據，再改用舊內容必須唯一命中的
   精確 Perl replacement；新報告與新工作紀錄仍使用 `apply_patch` 建立。
2. 舊數學文件含不可見 form-feed，普通文字 diff 不容易辨認。以 Unicode control-character 枚舉定位
   offset，再以精確替換修復，最後重新掃描全專案確認為空。
3. 初稿把 `poly_quality` 概括寫成 PyTorch／ONNX；重新對照 `validate_phase0.py` 與
   `fixed_point_emulator_supported=false` 後，修正成 PyTorch float reference，避免擴大證據。

## 清理盤點

| ID | 路徑 | 分類 | 判斷 |
| --- | --- | --- | --- |
| C01 | `.pytest_cache/`、`.ruff_cache/`、`__pycache__/` | 可重建 cache | 受 ignore 排除；未經額外授權不刪除 |
| C02 | `artifacts/runs/full35/` | 原始實驗、queue、OOM 與稽核證據 | 保留；新報告數值仍需其 provenance |
| C03 | 已發布 SiLU／qSiLU 權重分片 | 量化交付 | 保留 |

本次沒有刪除任何 cache、run、checkpoint、資料或使用者檔案。

## 未解事項或風險

- qSiLU 20-epoch seed-1 finalist 仍是人工中止狀態，不能視為正式完成；seed 2 未執行。
- BCSP 沒有 mixed-policy 搜尋結果，不能宣稱已完成 placement 演算法驗證。
- PTQ、QAT、Q8／Q12／Q16 全碼枚舉與 FPGA／ASIC 板上量測仍待量化／硬體階段處理。
- 本次 commit subject 由使用者指定為 `5090 Activation Report 0830`；完成後遠端 hash 由交付說明記錄。
