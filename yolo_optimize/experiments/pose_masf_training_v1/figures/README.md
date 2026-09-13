# 架構與梯度圖

- full-architecture.svg／png：layer 0–23 的完整架構、640 輸入時各層尺寸、固定／可訓練範圍。
- training-gradient.svg／png：兩條 Pose 訓練分支與 α／β 的不同作用。
- DOT 是可編輯原始圖；SVG 可放大，PNG 方便直接檢視；preview.jpg 是本機目視檢查用的輕量版本。

圖已更新為「B 組 CPU 通過，GPU 尚未訓練」。β=1 僅實作於新 training_b.py，既有移接候選程式未修改。A 加訓已取消；虛線只表示固定 E2 的原接法。完整解釋見[主計畫](<../README.md>)。圖片為 Graphviz 依真實接線產生，沒有使用生成式圖片猜測架構。
