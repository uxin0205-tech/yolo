# 2026-09-16 combine Pose 方法比較

變更與原因：使用者希望參考 yolo_combine 改善官方 Pose 訓練。讀取公開 policy、實際 args.yaml／results.csv、正式獨立 baseline 與模型來源，完成[研究報告](../research/2026-09-16-combine-pose-training-review.md)。另產生只改 lr0 的 MuSGD 候選 YAML，沒有加入現有 queue。

驗證方式與結果：正式 P3 實跑88輪，第68輪綜合最佳；與本次對照時標示架構、初始化、解析度及 backend 差異。候選 YAML 經官方 get_cfg 解析，程式斷言與現有 Stage 4 只差 name、lr0，確認初始 checkpoint 存在。未執行 GPU、未修改資料集或 yolo_combine。

困難與解法：底層 stage 定義含被覆寫的舊預設，改以實際 args.yaml 交叉核對；不同架構的成績不可直接用來證明 recipe 因果。

未解事項與風險：候選改善尚未實測；單 seed 與反覆使用 formal val 限制仍存在。沒有承諾提升幅度。
