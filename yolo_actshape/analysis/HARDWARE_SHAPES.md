# 以硬體實作為目的的特徵圖尺寸補充

使用者於 2026-11-08 再次強調：研究目的在於讓硬體容易實作，不能只比較輸入解析度或 AP。以下把邏輯特徵圖、實體記憶體排列與運算分塊分開。本文件最初記錄 CPU 查核；後續已獲授權並於 2026-11-11 完成初輪 40E＋擴充 60E，以及七組尺寸的 GPU replay。完整精度、成本與採用理由見 [Stage9 最終結果](../result/stage9/README.md)，原始來源鎖保持。

## stride2 與二次冪是兩件事

實際讀取選定模型，layer0、1、3、5、7都是kernel3、padding1、stride2。尺寸減半表示卷積視窗步進2個位置，**不是把每個像素值除以2**，因此沒有必要為此引入通用除法器。保留這種金字塔通常讓目前neck的2倍upsample與concat保持規律。

特徵圖40×40也不是硬體無法處理的形狀。平面逐列索引可用counter，下一行基址加40；即使直接計算y×40+x，也能寫為(y<<5)+(y<<3)+x。二次冪有利於部分位址切位，但不應為省一點位址邏輯就增加大量空白儲存、token或resize。

## 分塊比「全圖一定二次冪」更值得先檢查

40×40配8×8 tile為5×5塊，沒有空白格。若固定16×16，則需3×3塊、容量2304，邏輯格子1600，配置容量多44%（格子利用率69.44%）。這些數字只算空間分塊，不含Conv halo、通道tile、雙緩衝或實際bank安排；8×8因此不是已證明的全網最佳tile。

也可以保持模型數學形狀，把記憶體pitch或channel packing對齊，讓硬體逐塊處理。只有配置容量改變、且運算排除無效位置時，才可能保持數值等價；不能讓padding token參與Attention分母。這與把特徵圖真正resize成另一個大小是不同方法。

AMD HLS官方文件把memory access、partition／reshape列為效能瓶頸與優化點，line buffer可只保留卷積所需的前幾行。NVDLA則以32-byte資料atom及line/surface stride對齊定義儲存方式。這些例子支持先看記憶體與資料流，不支持「每個H/W都必須二次冪」。[AMD陣列存取](https://docs.amd.com/r/2024.1-English/ug1399-vitis-hls/Array-Accesses-and-Performance)、[AMD line buffer](https://docs.amd.com/r/2022.1-English/ug1399-vitis-hls/Vertical-Convolution)、[NVDLA資料格式](https://nvdla.org/hw/format.html)。

## 新增的硬體候選

以下P3／P4／P5均為**內部特徵圖**。左欄canvas只說明在既有stride2架構下，怎樣產生這組尺寸；尚未實作固定640輸入、只改某一級的替代架構。

| Canvas H×W | P3 → P4 → P5 | 每站token N | 面積／固定640 | 形狀上的價值與代價 |
|---|---|---:|---:|---|
| 256×512 | 32×64 → 16×32 → 8×16 | 128 | 0.32 | 各軸二次冪、容量小；小球／球棒解析度風險較大。 |
| 384×512 | 48×64 → 24×32 → 12×16 | 192 | 0.48 | 各級寬度二次冪、N可分為3個64-token塊；部分8×8空間tile有尾塊。 |
| 512×768 | 64×96 → 32×48 → 16×24 | 384 | 0.96 | 五級H/W皆可整除8、較接近640平方的像素面積；較寬row可能增加line-buffer。 |
| 已測512×512 | 64×64 → 32×32 → 16×16 | 256 | 0.64 | 最規整的平方方案；E0已有精度下降，尚未做本輪適應訓練。 |
| 已完成 20E 的 512×1024 | 64×128 → 32×64 → 16×32 | 512 | 1.28 | 長方形且各軸二次冪；儲存和計算比固定640更大。 |

512×768的N²是N400參考的92.16%；512×1024則是163.84%。所以512×1024雖然各軸二次冪，並不自動更省。真正硬體優先的研究應加入tile利用率、row/channel對齊、片上容量、外部流量和tail控制，再與AP一起選型。

三個head輸入的C實測為256／512／512，原模型的通道數已容易對齊16／32等常見並行組。FP32單份P3+P4+P5容量：固定640為10.15625MiB、256×512為3.25MiB、384×512為4.875MiB、512×768為9.75MiB；這不是實際峰值SRAM，因為存活時間、skip、串流、雙緩衝及其他tensor另計。INT8情境只在JSON另外估算，不代表已完成整網量化。

## 優先順序與界線

1. 先分析原模型的8／16／32 tile、line buffer、channel packing與實體pitch，尋找不改模型數學形狀的做法。
2. 以512×768、384×512作新的幾何／資源候選，256×512作更低容量但高精度風險候選；先做相容與誤差查核。
3. 固定輸入、獨立改內部尺度，以及不均勻stride或減少尺度，是另外的架構延伸：需處理feature對齊、取樣相位、感受野及head stride，不能靠插一個resize就宣稱免費省資源。

即使N為128／256／512，Softmax分母仍是各權重的總和，不是常數N，不能直接改成固定右移。沿用D1 PWL exp＋LUT256 reciprocal，qSiLU／B2／S1與既有bias軸上限32仍須保護。

目前尚未指定目標 FPGA／ASIC／NPU 型號、SRAM／BRAM 容量、並行 lane、位寬及外部 bus，因此不宣稱某一個尺寸是硬體最優。新增三組尺寸均已完成正式 AP 與各 20E 適應，但沒有通過全部採用門檻，主線保留原 rect640。CPU 容量估算、GPU replay 與同步推論時間，均不等於硬體綜合或實板量測。

資料與重算：[純CPU估算程式](hardware_shapes.py)、[JSON](hardware-shapes.json)、[CSV](hardware-shapes.csv)、[實際模型CPU相容結果](hardware-shapes-cpu.json)。執行 `python hardware_shapes.py` 不需要GPU或checkpoint。唯讀參考 optimize 的analytical profiler、runtime profiler、既有hardware-codebook研究；其SHA記在CPU結果內。本輪是新延伸，沒有冒稱optimize已驗證這些尺寸。
