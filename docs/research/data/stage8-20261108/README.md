# Stage8 最終比較資料

資料截點：2026-11-08T15:04:45.379480+08:00。詳見 [無 Rep／Rep／ACB 比較與推薦](../../stage8-comparison-2026-11-08.md)。Stage8 已完成，主線推薦 rep17_20_silu E1，最終部署使用 qSiLU／D1；Stage9 未開始。

[metrics.csv](metrics.csv) 含 11 列：無 Rep qSiLU／PolyQuality、舊輪三組、新輪五組及 SiLU 衍生硬體適應。兩個無 Rep 模型來自 Stage7 各 10E，未新增 control 加訓；[epochs.csv](epochs.csv) 僅含 Stage8 舊輪 15E、新輪 25E 及硬體適應 3E，共 43 筆。硬體適應未選定，沒有另做獨立最終重載驗證。PolyQuality 為不同 activation 的參考，不加入 Stage8 正式選型。

CSV 的 AP 為 0–1，報告與圖為百分比，差值為百分點。[snapshot.json](snapshot.json) 保存原始精度、模型狀態、來源 SHA、運算成本與限制；[verification.json](verification.json) 記錄本次無 GPU 查核。Stage8 五組最終 AP／folding／可攜驗證已完成。

執行 `python redraw.py` 重畫三張圖的 PNG／SVG，只需 NumPy／Matplotlib，不需 GPU、checkpoint 或研究套件。本資料夾保存小型證據與重畫來源，適合納入 Git；不放權重、資料影像、cache 或重複的封存快照。
