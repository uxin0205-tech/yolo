# 特徵圖與硬體成本的第一輪分析

本文件是 CPU 相容性、完整 validation 影像尺寸統計及算子檢查的結論。正式 AP 以 GPU queue 結果為準；不把可前向、估算 MAC 或二次冪尺寸當作精度／加速的證據。

## 為何會有 40×40？

40×40 是 640×640 輸入經累積 stride16 的結果。模型不要求這一層永遠為 40 或永遠方形。這份實際模型的 P3／P4／P5 位於 layer16／19／22；兩個 Attention 都在 stride32，原固定640時是20×20。Hadamard 的長度是 key channel32，並非空間 token。

| 固定輸入 H×W | P3 | P4 | P5／Attention | 每站 tokens | Conv＋P·V GMAC | 相對固定640 |
|---|---|---|---|---:|---:|---:|
| 640×640 | 80×80 | 40×40 | 20×20 | 400 | 47.408 | 1.000 |
| 512×512 | 64×64 | 32×32 | 16×16 | 256 | 30.322 | 0.640 |
| 384×768 | 48×96 | 24×48 | 12×24 | 288 | 34.117 | 0.720 |
| 512×1024 | 64×128 | 32×64 | 16×32 | 512 | 60.711 | 1.281 |
| 1024×1024 | 128×128 | 64×64 | 32×32 | 1024 | 121.691 | 2.567 |

上述 GMAC 是 CPU trace 的 both routing Conv 加上浮點 P·V 小計，不含 binary QK、逐元素運算、前處理、資料搬移與延遲。reference_rect640 沿原始 batch 最小補邊，canvas 可能因 stride／pad 規則變動，不能用固定640的成本代替。所有尺寸的輸出有限，矩形切換後切回的完整輸出逐位相同。

## 長方形與二次冪的不同效果

兩種長方形都是 stride32 對齊；512×1024 的各級 H、W 都是二次冪，384×768 則不是。512 平方省成本主要因面積縮小，不能全歸因二次冪。1024 平方雖然各軸整齊，但面積是640平方的2.56倍，Attention 的 N² 增幅更大。

COCO val5000 中3602張橫向、1228張直向、170張正方；BBAT val683 中454張橫向、229張正方，沒有直向。約2:1的canvas讓BBAT平均補邊從平方的28.9%降至約24.3%，但COCO平均補邊從28.4%升至37.9%。因此較適合棒球影像的比例未必適合 person／完整 COCO；必須同時保護兩個任務。

為區分現有尺寸和固定 canvas 的效果，原 rect640 與 fixed640 都只驗證不加訓。固定 canvas 使用一次原生 LetterBox、等比例縮放、pad114；bbox／keypoint 共用同一 gain，已測真實 transform／inverse。正式來源、split、標註不改動。

## 不應直接向 Attention 補零 token

將20×20空間 buffer補成32×32，容量會從400增加到1024個token。若把補零也放進注意力，會新增 score、改變 row-max／分母及位置bias；補零不是數值等價的捷徑。這輪只估算硬體 buffer／tile 容量，沒有加入這種未遮罩token。

既有相對位置表每軸最大32，最大N1024。exp(0)的整數權重是32768，最大分母1024×32768＝33554432，需要unsigned26。候選全部在目前位寬內，D1不回退浮點除法。N288均勻score的row-sum誤差約0.001738；已分開記錄近似誤差，未用最後一次浮點除法掩蓋。

## 完整部署運算檢查

| 部位 | 目前實際處理 | 剩餘限制或可再處理事項 |
|---|---|---|
| Rep17／20 | 已 fold 成單 Conv3×3+bias，後接 qSiLU | 不把 qSiLU fold 進線性 Conv；保留本輪權重與位置契約。 |
| activation | 固定 qsilu_pq 分段與係數 | PyTorch 任務路徑仍使用浮點張量，不是全網整數部署。 |
| QK | B2 sign＋Hadamard | Hadamard 形成與動態資料搬移仍需成本；不是整個 Attention 都只做位元運算。 |
| scale／bias | S1 Q10、signed16/1024 分解式 bias | 部署常數固定，沿用 manifest。 |
| normalization | D1整數PWL exp20＋每列LUT256 reciprocal | 實際整數div／移位須按owner解讀；沒有最後浮點除法補正。 |
| QKV／P·V／projection／其他Conv | 浮點 | 未進行整網PTQ／QAT，不以局部簡化冒稱整網整數。 |
| 非Rep ConvBN | 仍保留推論BN | 可離線fold，但需獨立等價與部署格式驗證；此次尺寸矩陣不混入另一個圖變因。 |
| head confidence／visibility | 原sigmoid語意 | 任意換hard近似會影響threshold／AP，需要獨立誤差及任務驗證，未默默替換。 |
| 前處理／座標還原 | 等比例resize、pad、RGB/255與反算 | 固定係數可預計算；可變影像gain仍依輸入決定。外部後處理成本另外計。 |

以上依 `cpu-audit.json` 引用的選定模型部署算子稽核。未做 ONNX／TensorRT 轉換、RTL／HLS 或板端量測；不能由軟體算子數宣稱硬體加速。

## 適應研究

沿用 optimize 已核對的 AdamW、warmup／cosine、fresh EMA與FP32 CIoU。形狀適應本身是新的研究延伸，固定 backbone／Attention／BN running，調整 neck／雙 head。完整 E0 後各成本組最多選一組，10E再半LR10E，最多40E，每輪完整AP；採用看十項AP及成本，不因二次冪而預設勝出。

依據：[CPU trace／D1](artifacts/shape-20261108-v1/cpu-audit.json)、[完整影像尺寸統計](artifacts/shape-20261108-v1/data-audit.json)、[optimize 核對](reference-review.json)、[固定 protocol](protocol.json)。
