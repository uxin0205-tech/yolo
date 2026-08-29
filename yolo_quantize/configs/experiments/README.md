# Activation預選實驗契約

- [`activation-smoke-v2.yaml`](activation-smoke-v2.yaml)：模型、checkpoint、資料、校正、probe、矩陣、graph與產物契約。
- [`activation-preselection-v1.yaml`](activation-preselection-v1.yaml)：A8主線、A6／A7探索、停止項目與W-SD4／A-SD4公平比較設計。

兩份YAML都標示`formal_training: false`或等價狀態；它們不授權或啟動GPU工作、QAT或正式訓練。
