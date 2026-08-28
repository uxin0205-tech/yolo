# architecture-screen-20-v1

這是 architecture_2 的可重建篩選 View，不是新的 canonical dataset，也沒有複製影像或標註。

- 目的：供 C0-Control、C1、C2、C3 使用相同資料做 Float 20% 初篩，並供通過候選做 PTQ／QAT-lite 模擬。
- COCO：從 train2017 以 seed 0 隨機抽取 23,657 張作訓練，另取互斥的 5,000 張 train-only search-val；官方 val2017 未使用。
- BBAT5：從 canonical bbat5-v1 的 search-train 依 `.rf.` 前 prefix 整組抽取 1,073 張，共 410 groups；沿用既有 search-val 600 張。
- Pose 與二類 Detect 使用相同影像 assignment。
- `fraction` 在 training YAML 保持 1.0；20% 由這些固定 manifest 決定。
- Ultralytics train／validation label index只寫入本目錄的`runtime-cache/`；可重建且不提交。
- COCO與bbat5-v1來源旁出現`.cache`視為錯誤，CPU preflight與正式runner都會fail closed。
- 禁止用此 View 宣稱正式 C_best，正式 val 仍保留給完整資料確認。
- 原始與 canonical 資料修改：無。

完整來源雜湊、輸出雜湊、類別統計與 leakage 證據見 `screening-manifest.json`。
