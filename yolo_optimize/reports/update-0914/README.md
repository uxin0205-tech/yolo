# 0914 更新：MASF → BinaryQK → RepConv

本次 Git commit 名稱：`5090 Uodate 0914`（依使用者原字拼寫）。這是完成後的整理發布，不新增 GPU 訓練、不改正式選用權重。

## 先看哪一份

1. [完整總表與結論](<../../experiments/post_binary_rep_v1/STATUS-20260914.md>)：六階段、八項 AP、設定、採用閘及權重入口。
2. [實際串接與訓練設計](<../../experiments/post_binary_rep_v1/README.md>)：三個 Rep 都從 BinaryQK E5 開始。
3. [scale/bias 硬體設計核對](<../../experiments/post_binary_rep_v1/SCALE-BIAS-HARDWARE.md>)：固定常數／小表與尚未完成的純整數部分。
4. [完整比較 CSV](<comparison.csv>)：保留精確差值，可自行檢查。
5. [checkpoint 路徑／大小／SHA-256](<checkpoint-manifest.json>)：53 個本機權重與快照全部核對，權重不納入一般 Git。

![本輪血緣與架構](<architecture.svg>)

## 本輪結論

全部 GPU 排程於 2026-09-14 04:49:57（台北）正常結束。MASF B5 的收益不能獨立歸因於 MASF；BinaryQK 新 scale/bias 有 COCO／person／ball 收益，但 bat 退步；Rep17 較有利於 person／Pose，雙層較有利於 COCO。Rep 三組都未通過採用門檻，不把局部提升稱為全面勝出。

目前保留的正式研究版本仍見[選用模型說明](<../current-model/README.md>)。本轮候選有兩套 MASF，正式舊權重只有 Detect P3 MASF，兩者不能混稱。

## 整理與發布邊界

- `experiments/pose_masf_training_v1/`：Pose MASF B5 程式、設定、結果與圖。
- `experiments/post_masf_hardware_v1/`：已完成 BinaryQK scale/bias；舊 B5 直出 Rep 排程已被取代，保留紀錄。
- `experiments/post_binary_rep_v1/`：有效新 Rep 三組及完整結果。
- 本目錄：統一閱讀入口、CSV、架構圖、checkpoint 與發布驗證索引。

本機全部 checkpoint 保留原位；Git 只發布程式、報告、必要圖、設定與小型數值證據，不含 `.pt`、runtime 資料集、cache、原始大 log 或 PDF。不建立另一份訓練資料，不重新切分 BBAT5 v1。GitHub 上本機權重連結會明確改為未上傳註記。

## 驗證與限制

三組 Rep 皆 CPU／GPU smoke／5E／全量驗證／Float＋BitTrue 重載完成；本次再做 53 份權重 SHA 稽核，未重跑 GPU。發布副本另檢查 Python AST、JSON 有限值、CSV、SVG／PNG 與連結。完整硬體 latency／energy、多 seed 與獨立 test 尚未做。

清理盤點見 [保留清單](<CLEANUP.md>)，本次不刪實驗產物。發布檢查結果與完整檔案白名單保存在本目錄 publication-check.json。
