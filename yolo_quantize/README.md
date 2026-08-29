# Full35 Activation量化預選證據包

本目錄是2026-08-29 Full35 Detect＋Pose activation-output量化預選的可公開證據包。它回答「為什麼第一輪weight sensitivity先固定A8」並提供老師版PNG／PDF／SVG、完整中文論證、30格source JSON／CSV、machine-readable契約及可重建繪圖腳本。

這不是正式activation winner、完整mAP validation、QAT或硬體效能報告，也不包含checkpoint或weights。

## 直接取用

- [老師版PNG](deliverables/activation-preselection-teacher-v1.png)：適合投影片與即時分享。
- [老師版PDF](deliverables/activation-preselection-teacher-v1.pdf)：單頁附件。
- [老師版SVG](deliverables/activation-preselection-teacher-v1.svg)：向量排版。
- [完整預選報告](docs/reports/2026-08-29-activation-preselection-report.md)：實驗契約、指標、A3至A8判讀、SD4設計與限制。
- [30格smoke報告](docs/reports/2026-08-29-activation-output-smoke-v2.md)：原始測試範圍與完整矩陣。
- [發布manifest](PUBLICATION_MANIFEST.yaml)：本次Git範圍與檔案雜湊。

## 決策摘要

| 類別 | 決策 | 原因 |
|---|---|---|
| 第一輪主線 | qSiLU＋A8 | 保留上游short-recovery的mAP headroom角色 |
| 第一輪主線 | poly_quality＋A8 | 這批smoke的A8 raw NRMSE與minimum TopK overlap最佳 |
| 第一輪主線 | poly_shift＋A8 | 保留shift／APoT硬體取向 |
| 後續探索 | qSiLU＋A6、poly_quality＋A7、poly_shift＋A7 | 等mixed weight policy確定後再耦合重驗 |
| 不晉級 | A3至A5 | Top-300候選集合變化過大 |
| 不新增 | SiLU、Hardswish、ReLU | SiLU只作歷史control；其他不占後續矩陣 |

`0.3797`是量化誤差RMS相對參考輸出RMS約0.38倍，不是mAP下降37.97%。`0.0967`表示較差任務的Top-300 class–anchor pair約只保留29個相同候選。因此A5不作目前主線，先以A8隔離weight誤差。

## 證據鏈

    activation-smoke-v2.json（30格原始證據）
    ├─ activation-smoke-v2-summary.csv（人工可讀摘要）
    ├─ activation-preselection-v1.yaml（決策與SD4契約）
    ├─ 2026-08-29-activation-preselection-report.md（完整論證）
    └─ render_activation_preselection.py
       ├─ activation-preselection-teacher-v1.png
       ├─ activation-preselection-teacher-v1.svg
       └─ activation-preselection-teacher-v1.pdf

30個`passed`只表示graph、observer、輸出結構與有限值檢查成功，不代表精度gate通過。最終仍須以COCO box、BBAT box、BBAT pose完整mAP驗證。

## SD4定位

- W-SD4：weight codebook主線；未來和W8、W4、Fixed SD4、LS-SD4在相同activation parent、region與budget下比較。
- A-SD4：activation output探索支線；必須和同為4-bit的LSQ+ A4使用相同parent、boundary、calibration與budget公平比較。
- 本公開包只有實驗設計，沒有SD4量測結果。

## 重建老師版圖表

需求：Python 3.12、`matplotlib==3.11.1`及Noto Sans CJK字型。腳本目前固定讀取：

- `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
- `/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc`

執行：

    cd yolo_quantize
    python3 scripts/render_activation_preselection.py --input artifacts/reports/activation-smoke-v2.json --output-dir /tmp/activation-preselection-rebuild

腳本不使用GPU，也不啟動validation、QAT或訓練。固定PDF metadata與SVG ID salt後，指定環境可逐位元重建三種正式成品。

## 資料與發布邊界

- Detect證據使用COCO2017；Pose證據使用不可變Canonical BBAT5 v1。
- 每task僅2張canonical train exemplar校正與1張canonical val exemplar probe；沒有重切、抽樣或修改BBAT5。
- 本包不含`.pt`、checkpoint、run、cache、dataset副本或PTQ結果。
- 原始smoke runner與Full35／yolo_activation本機bundle不在這個「預選報告先行發布」範圍；raw evidence與產物重建所需資料已保留。

## 工作紀錄

- [Activation耦合與30格smoke](docs/worklogs/2026-08-29-activation-coupled-smoke-matrix.md)
- [老師版報告與圖表](docs/worklogs/2026-08-29-activation-preselection-figure.md)
- [GitHub發布收尾](docs/worklogs/2026-08-29-activation-preselection-github-publication.md)
