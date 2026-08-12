# MASF-YOLO：BBT5 球棒／球偵測實驗首頁

本 repo 是 MASF-YOLO Phase 1 的可重現實驗工作區。資料固定使用 `bbt5-detect-baseline/dataset/`，初始化固定使用 `bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`。該 pose-derived 權重已看過 BBT5，因此本階段是資料暴露下的操作性消融，不能解讀為無洩漏泛化能力。

## 先看這裡

| 內容 | 直接入口 |
|---|---|
| 本輪完整中文報告 | [B1R/P2/P3 REPORT](artifacts/b1r-p2-p3-retest/REPORT.md) |
| 所有 val/test 比較表 | [comparison.csv](artifacts/b1r-p2-p3-retest/comparison.csv) |
| JSON 統整結果 | [summary.json](artifacts/b1r-p2-p3-retest/summary.json) |
| 結果完整性檢查 | [final_audit.json](artifacts/b1r-p2-p3-retest/final_audit.json) |
| checkpoint 路徑與 SHA-256 | [checkpoints.json](artifacts/b1r-p2-p3-retest/lineage/checkpoints.json) |
| 硬體成本 | [profiles/summary.json](artifacts/b1r-p2-p3-retest/profiles/summary.json) |
| 整理後 study 封面 | [b1r_p2_p3_study/](b1r_p2_p3_study/README.md) |
| 第二輪設計規格 | [B1R/P2/P3 spec](docs/superpowers/specs/2026-08-11-b1r-p2-p3-retest-design.md) |
| 論文實作稽核 | [paper audit](docs/research/2026-08-11-masf-paper-implementation-audit.md) |

## 歷史紀錄（第一輪 Phase 1）

舊的 M0–M7、P3M、SP2、SP2P、B0/B1 紀錄已保留並接入目前首頁；它們仍與本輪 B1R/P2/P3 重測分開，不會被誤當成同一組公平排名。

| 內容 | 入口 |
|---|---|
| 歷史結果摘要與指標 | [LEGACY_RESULTS.md](b1r_p2_p3_study/results/LEGACY_RESULTS.md) |
| 第一輪完整中文報告 | [EXPERIMENT_RESULTS_ZH.md](EXPERIMENT_RESULTS_ZH.md) |
| 第一輪機器報告 | [artifacts/static-phase1/report.md](artifacts/static-phase1/report.md) |
| 第一輪產物索引 | [artifacts/static-phase1/README.md](artifacts/static-phase1/README.md) |

## 目前正式結論

統一 test 的 B0 Original 3Scale 為 0.770812；最佳新增模型為 P3 Partial25-35，
為 0.754951。P3 PaperFormula-Full 為 0.750617，P2 Control-Full 為 0.747240。
因此目前證據支持「P3 partial 比完整 P2 更穩定」，不支持「加入 P2 必然提升」。

完整模型、Ball/Bat、AP_S/AP_M/AP_L、FP、recall 與硬體數值均在上方報告與 CSV。

## 專案導覽

| 路徑 | 用途 |
|---|---|
| `MFAM_plan.md` / `codex_plan.md` | 研究設計與執行清單 |
| `configs/` | 正式設定與模型 YAML |
| `bbt5-detect-baseline/` | BBT5 detect view 與來源權重 |
| `masf_yolo/` / `tests/` | 實驗程式與自動測試 |
| `artifacts/` | 正式權重、評估、profile 與報告 |
| `docs/` | agent 規則、spec、計畫與 ADR |

`artifacts/static-phase1/` 是第一輪正式結果；`artifacts/b1r-p2-p3-retest/` 是本輪
B1R/P2/P3 統一後處理結果。原始資料與正式 checkpoint 不移動、不覆寫。

重要資料夾各有 `README.md`；自動生成的 leaf 不重複放文件，以免污染模型輸出。

## 操作

```bash
env PYTHONPATH=. ../.venv/bin/pytest -q
env PYTHONPATH=. ../.venv/bin/python -m masf_yolo.cli pipeline status --config configs/static-phase1.yaml
env PYTHONPATH=. ../.venv/bin/python -m masf_yolo.cli report --config configs/static-phase1.yaml
# dry-run：只列清單與 hash
env PYTHONPATH=. ../.venv/bin/python -m masf_yolo.cli cleanup --config configs/static-phase1.yaml
# pipeline 完成且服務停止後才實際清理
env PYTHONPATH=. ../.venv/bin/python -m masf_yolo.cli cleanup --config configs/static-phase1.yaml --apply
```

正式 native 權重在 `artifacts/static-phase1/runs/<run>/weights/{best,last}.pt`；canonical 權重在 `training/<stage>/canonical.pt`；val/test 在 `evaluation/{val,test}/<model>/metrics.json`；硬體資料在 `profiles/<model>/profile.json`。正式 checkpoint、CSV、metrics、profiles、selection 與 audit 保留；smoke/preflight/gate checkpoint、cache 與可重建預覽圖才在清理白名單。

全部模型只有單一 seed，且 initializer 已看過 BBT5。若要主張穩定提升與泛化能力，需多 seed 並加入未接觸 BBT5 的乾淨 initializer。
