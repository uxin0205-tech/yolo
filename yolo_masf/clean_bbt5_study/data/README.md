# 資料

- 來源：`../../bbt5-detect-baseline/dataset/`，只讀且完整保留。
- 固定證據：`../../artifacts/locked-bbt5-dataset/`。
- Train：更新權重。
- Validation：checkpoint、超參數判斷與 selection。
- Historical test：selection freeze 後才讀取，不進 trainer YAML。
- Final holdout：尚未建立；建立前禁止 unseen generalization 宣稱。

Clean 與舊實驗使用同一個已鎖定 BBT5 split，以便架構比較；差異是 Clean 使用未接觸 BBT5 的官方 COCO80 initializer。
