# weights

本目錄存放本機模型 checkpoint，不提交 Git。

- `yolo26m.pt`：官方 Ultralytics YOLO26m baseline 與所有研究階段的初始權重。
- training YAML 統一使用 `weights/yolo26m.pt`，命令應從 repository root 執行。
- checkpoint 下載後以 SHA-256 記錄 provenance；重新下載或更換權重時必須更新實驗 manifest。

目前 checkpoint：

- 來源：Ultralytics assets `v8.4.0/yolo26m.pt`。
- 下載時使用的 Ultralytics：`8.4.90`。
- 大小：44,255,705 bytes。
- SHA-256：`401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`。

確認權重可載入：

~~~bash
../.venv/bin/python -c "from ultralytics import YOLO; YOLO('weights/yolo26m.pt')"
~~~
