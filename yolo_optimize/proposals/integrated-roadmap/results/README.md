# 第一輪可重建結果表

本目錄保存由實際新實驗 summary 產生的交付表，不混入歷史消融數字，也不代表增準成功。

使用 `/home/uxin/yolo/yolo_combine/.venv/bin/python scripts/build_round1_evidence.py` 重新建立：

- `epoch-comparison.csv`：每個完成 epoch 的 EMA／live 八項 AP、joint、對原 BEST 差值；EMA 另列對相同 epoch native 的 joint 差值。
- `evidence-index.json`：原 BEST hash、必要 checkpoint 存在檢查、各 run 已完成預算與 scope。

輸入固定為 `artifacts/direction1-20260908/` 下的既有 summary／checkpoint；腳本不訓練、不改 checkpoint、不讀 GPU。重新執行只覆寫本目錄兩份可重建表。

Git 邊界：可保存小型 CSV／JSON／Markdown 與 generator；原始影像、checkpoint、cache、validation 大型產物不隨此目錄發布。未獲 commit／push 授權，本次不執行 Git 發布。
