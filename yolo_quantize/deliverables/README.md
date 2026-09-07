# Activation預選可分享成品

另見[2026-09-07全模型量化盤點附件](full-model-audit-2026-09-07/README.md)：V36實測數據、148路格式、四天規劃與圖表。以下activation圖保留歷史版本。

本目錄保存同一張Full35 activation預選老師版圖表的三種格式。完整解釋見[預選報告](../docs/reports/2026-08-29-activation-preselection-report.md)。這是2026-08-29的凍結歷史成品；2026-08-31後active shortlist已改為qSiLU／Hardswish／poly_shift，見[修訂報告](../docs/reports/2026-08-31-hardswish-policy-revision.md)。舊圖不重繪或改hash，以免把新決策偽裝成當時結果。

| 檔案 | 用途 | SHA-256 |
|---|---|---|
| `activation-preselection-teacher-v1.png` | 投影片、訊息與一般預覽 | `3bfe999ce0fd9ed2fde79500cc03a5484d3459647ced29a28ae58023d5c50684` |
| `activation-preselection-teacher-v1.pdf` | 單頁正式附件 | `8b1ca9e44cf1679d924cfe08197782ce575d2c9d1c2ee41965b22e791002a551` |
| `activation-preselection-teacher-v1.svg` | 可縮放向量排版 | `82ea786c5be8558659a790f3a85ee834a48bb12f8b77e191ddbe15459b7ec103` |

三者均由[`scripts/render_activation_preselection.py`](../scripts/render_activation_preselection.py)讀取固定source JSON產生；沒有手動改數值。PNG為2471×1361 RGBA、PDF為1頁。
