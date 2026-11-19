# 第 1 階段 B2：運算量實際減少多少

**每張 640 × 640 輸入，整網浮點 MAC 小計減少約 0.109%，不是大幅整網減算。** 移除的是兩處 Q·K 的浮點乘加，另加入二值比對／popcount、Hadamard 與尺度／分支合併成本。計入新增浮點算術後，所列浮點運算小計淨降幅約 0.103%；這不是包含二值運算的總成本降幅，也不是實測速度提升。

## 口徑與 CPU 核對

相同原始 YOLO26m parent 與本輪交付 B2，batch 1、640 × 640、CPU FP32 eval、no_grad、未 fuse。以 Conv hooks 加 TorchDispatchMode 計數完整實際前向，與唯讀 optimize `accounting-v2/fp.json` 相同口徑。兩模型輸出均有限、參數數量相同；未啟動 GPU。

兩個 Attention sites 的輸入均為 `[1,256,20,20]`，每處 4 heads、Q/K 維度 32、V 維度 64；每個 head 為 400 tokens。這裡不是 40 × 40。驗證集 rect padding 可能產生其他大小，以下不是全驗證集的平均運算量。

1 MAC 表示一次乘加；換算常用 FLOPs 小計時採 1 MAC＝2 FLOPs。下表不含共同 BN、activation、Softmax、pooling、decode、sort／NMS、記憶體搬移。現行未 fuse head 路徑仍會執行 one2many 與 one2one 分支；此值不代表移除 one2many 後的匯出圖。

| 浮點 MAC 項目 | 原始 parent | B2 | 減少 |
| --- | ---: | ---: | ---: |
| 全模型 Conv | 37,382,758,400 | 37,382,758,400 | 0 |
| 兩處 Q·K | 40,960,000 | 0 | 40,960,000，改用二值分支 |
| 兩處 P·V | 81,920,000 | 81,920,000 | 0 |
| 整網小計 | **37,505,638,400** | **37,464,678,400** | **0.109210%** |

Conv 並未二值化。QKV projection、Attention output projection 與其他 Conv 都包含在第一列，保持相同計算量；模型參數仍為 21,896,248。

## 被替換與新增的運算

原始 Q·K 的兩處總量：

```text
2 sites × 4 heads × 400 × 400 token pairs × 32 channels
= 40,960,000 浮點 MAC
```

B2 有 sign 與 Hadamard 兩條分支，因此為 **81,920,000 個 1-bit pair products**，以 XNOR 與 popcount 形成 score；不是沒有計算。

若未來將長度 32 的向量打包成 32-bit word，可表示為 **2,560,000 次 32-bit XNOR＋2,560,000 次 popcount**，另有每個 dot 的 `2×count−32`、sign／packing、轉型與資料搬移。這只是未來打包的運算數估計，不能當作已實作 kernel，也不能直接換算為相同數目的浮點 FLOPs 或硬體 cycles。

目前實作以布林 tensor 比對、int64 sum 模擬 popcount，另有重複 sign 比較及轉型，並未打包為 32-bit 核心。因此不能用理想 bit packing 倍率宣稱目前 GPU 加速；本次沒有量測 B2 推論延遲或能耗。

每張圖另有以下浮點工作：

| B2 項目 | 次數 | 公式 |
| --- | ---: | --- |
| Q/K Hadamard 加減 | 1,024,000 | 2 sites × 2（Q、K）× 4 heads × 400 tokens × 32 × log₂32 |
| 兩分支固定 scale 乘法 | 2,560,000 | 2 sites × 4 heads × 400² × 2 bases |
| 兩分支合併加法 | 1,280,000 | 2 sites × 4 heads × 400² |

原生 Q scaling 的 102,400 次乘法同時移除。本輪推論 Hadamard 不做 `1/√32` 乘法，因後續 sign 不受該正常數影響；訓練的代理路徑另計。

```text
浮點運算淨減少小計
= 2 × 40,960,000 + 102,400 − 1,024,000 − 2,560,000 − 1,280,000
= 77,158,400 FLOPs

相對「原整網 MAC×2＋原 Q scaling」小計 = 0.102862%
```

此淨減少仍不含二值／整數成本，共同未統計算子也不在分母。只看 Q·K＋P·V，浮點 MAC 減少 33.33%；再包含 Attention 內的 QKV、PE 與 output projection，局部浮點 MAC 降幅約 14.52%。這些局部比例不能當作整網比例。

## 結論與可重現資料

本階段把 Q·K 的浮點乘加替換成二值運算，但 Q·K 位於兩處低解析度特徵圖，原本只占整網很小部分。以本輪官方 AP 從 0.524686 降到 0.509889、整網浮點 MAC 只減少約 0.109% 而言，目前不能宣稱已取得理想的整網精度／減算交換，也不能宣稱實際加速。

後續 Q10／二次冪 scale、PWL／reciprocal 與 qSiLU 各自處理不同算子；它們的好處須再依位寬、乘除器、LUT、資料搬移及實測成本判斷，不能只以 FLOPs 排序。本次只補成本分析，第 2 階段仍未開始。

- [完整公式與數據](operation-cost.json)、[兩模型 CPU 計數](cost-model-check.json)。
- [獨立分析腳本](../../../scripts/stage1_cost.py)：預設只讀小型 JSON，不需 checkpoint、PyTorch 或 GPU；加 `--verify-models` 才載入模型執行 CPU 核對。
- [工作紀錄](../../../../docs/worklogs/2026-10-18-binaryqk-operation-cost.md)。
