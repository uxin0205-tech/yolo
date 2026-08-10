# MASF-YOLO：BBT5 球棒／球偵測實驗

本 repo 是 MASF-YOLO Phase 1 的可重現實驗工作區。資料固定使用 `bbt5-detect-baseline/dataset/`，初始化固定使用 `bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`。該 pose-derived 權重已看過 BBT5，因此本階段是資料暴露下的操作性消融，不能解讀為無洩漏泛化能力。

正式 pipeline `cb7e0392cc32` 已完成 34 個階段，final audit 通過。完整數值與分析見 `EXPERIMENT_RESULTS_ZH.md`；機器證據見 `artifacts/static-phase1/report.md`。

## 目錄導覽

| 路徑 | 用途 |
|---|---|
| `MFAM_plan.md` / `codex_plan.md` | 研究設計與執行清單 |
| `configs/` | 正式設定與模型 YAML |
| `bbt5-detect-baseline/` | BBT5 detect view 與來源權重 |
| `masf_yolo/` / `tests/` | 實驗程式與自動測試 |
| `artifacts/` | 正式權重、評估、profile 與報告 |
| `field_check/` | 正式管線外的探索性檢查 |
| `docs/` | agent 規則、spec、計畫與 ADR |

重要資料夾各有 `README.md`；自動生成的 leaf 不重複放文件，以免污染模型輸出。

## 本輪模型與結果

B1 是 YOLO11m P2 baseline，採凍結 backbone 0–10 共 10 epochs，再全模型 90 epochs。M7 是 P2 的 DW3×3 + DW5×5 + DW(1×7→7×1)；M0/M1/M2/M3 比較 kernel 與 partial channels；P3M 只在 P3 使用 3、5、factorized 7，沒有 9×9；SP2 是 Ball-only 輕量高解析 head；SP2P 再加入 validation 選出的 M2，並採同樣 10+90。

公平 test 中 M7 mAP50–95 為 0.7386、P3M 為 0.7340、B1 為 0.7303。SP2/SP2P 雖降低理論 GFLOPs，RTX 5090 實測更慢且 Ball 誤報大增，目前不建議部署。

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
