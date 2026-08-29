# 報告重建腳本

`render_activation_preselection.py`只讀取既有`activation-smoke-v2.json`並輸出PNG、SVG、PDF，不使用GPU、不載入模型，也不執行validation或訓練。

需求：Python 3.12、`matplotlib==3.11.1`與Noto Sans CJK字型。完整命令、字型路徑與限制見[子專案README](../README.md#重建老師版圖表)。

這個先行公開包不含產生30格raw smoke的Full35 runner；重建範圍是「由固定source evidence重建報告圖表」。
