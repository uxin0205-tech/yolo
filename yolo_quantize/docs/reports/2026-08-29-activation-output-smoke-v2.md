# 2026-08-29 Full35 activation-output無訓練GPU smoke報告

## 結論

5個activation乘LSQ+ A3至A8共30格已全部執行完成：30 passed、0 failed。Full35 graph、Detect／Pose輸出結構、124個部署activation observer及受保護的Binary Q/K、MASF、attention PWL、RealNVP與end-to-end head契約都沒有被破壞。

這批結果也否定「直接把上游activation最佳者接到量化」的做法。qSiLU是上游舊0.015 gate唯一通過的非SiLU候選，但在耦合的A7／A8 proxy中並非全面最佳；poly_quality多項raw與TopK proxy較佳。現階段沒有正式winner。

本次是無optimizer、無backward、無epoch、FP32 weights的GPU smoke，不是正式validation或QAT。NRMSE與TopK overlap不能換算成mAP，因此不能用來判斷0.04 absolute mAP gate。

## 實驗契約

- 模型：accepted Full35 Detect＋Pose，checkpoint SHA-256為d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c。
- activation來源：只使用yolo_activation；不存在也不依賴yolo_activation2。
- 矩陣：SiLU、qSiLU、poly_quality、poly_shift、Hardswish乘A3、A4、A5、A6、A7、A8。
- activation quantizer：per-boundary LSQ+，以observer min／max把實數端點映到整數端點。
- weights：全程FP32；沒有執行INT8、INT4、SD4或ternary weight量化。
- calibration：每個task各2張canonical train exemplar；BBAT5兩張來自不同.rf. source group。
- probe：每個task各1張canonical val exemplar。
- 資料治理：沒有建立新split、30% view或資料版本，也沒有更改BBAT5 assignment與labels。
- 裝置：NVIDIA GeForce RTX 5090；PyTorch 2.11.0＋CUDA 12.8。
- 比較：同一activation函數的quantization-disabled輸出，對比observer freeze後的fake-quant輸出。

## Full35量化範圍校正

上游manifest共有190個可替換SiLU函數位置，但不是190個都應加入部署activation-output quantizer。

- 124個部署會走到的位置：加入LSQ+ activation-output adapter。
- 66個training-only位置：detect_one2many 18、pose_one2many 24、pose_flow 24，不加入部署output quantizer。
- 所有124個observer在校正後都有有效且非退化range。
- protected module class signature在adapter前後完全一致。
- graph contract仍為23個shared layers、head inputs 16／19／22、strides 8／16／32、reg_max 1、end2end true、Detect nc 80、Pose nc 2、keypoint shape 2乘3、pose flow RealNVP。

這個校正避免把只在訓練使用的支路成本誤報為部署成本，也避免因pose_flow在inference不執行而把observer缺值誤判成模型錯誤。

## 30格矩陣

每格格式是worst Detect／Pose one-to-one raw NRMSE，再接minimum Detect／Pose TopK selected-pair overlap。NRMSE越低越好，overlap越高越好。

| Activation | A3 | A4 | A5 | A6 | A7 | A8 |
|---|---:|---:|---:|---:|---:|---:|
| SiLU | .6482 / .003 | .6648 / .010 | .3920 / .070 | .2049 / .233 | .1514 / .403 | .0911 / .680 |
| qSiLU | .6328 / .000 | .6564 / .000 | .3851 / .050 | .2045 / .230 | .1598 / .430 | .1019 / .613 |
| poly_quality | .6319 / .000 | .6020 / .003 | .3797 / .097 | .2058 / .237 | .1387 / .433 | .0816 / .693 |
| poly_shift | .6296 / .000 | .6398 / .023 | .3816 / .067 | .2124 / .247 | .1579 / .443 | .0989 / .687 |
| Hardswish | .8189 / .000 | .6809 / .000 | .4453 / .020 | .2770 / .090 | .1970 / .367 | .1194 / .617 |

所有格都能執行，不等於所有格都有可接受精度：

- A3／A4的TopK selected-pair overlap幾乎崩潰，只適合作低位元壓力控制。
- A5仍有明顯raw誤差及低Pose TopK overlap。
- A6開始有恢復跡象，但需要完整validation或QAT才能判斷。
- A7可作探索，A8是目前最安全的後續validation anchor。
- decoded output NRMSE會因TopK索引改變而非單調，不能只看decoded tensor。v2因此分開保存decoded、one-to-one raw與TopK overlap。

## A8細節

| Activation | Detect raw NRMSE | Pose raw NRMSE | Detect pair overlap | Pose pair overlap |
|---|---:|---:|---:|---:|
| SiLU | .091132 | .075442 | .8400 | .6800 |
| qSiLU | .094659 | .101851 | .8533 | .6133 |
| poly_quality | .081644 | .068198 | .8600 | .6933 |
| poly_shift | .098934 | .079368 | .8200 | .6867 |
| Hardswish | .119412 | .091203 | .8100 | .6167 |

poly_quality在這個極小probe的A8 raw與minimum pair overlap最好，但樣本量不足、校正只用min／max，也沒有mAP，因此只能作優先序線索。qSiLU仍值得保留，因為它有上游short-recovery證據；poly_shift則保留hardware-oriented角色。

## 建議但未執行的下一步

若使用者要繼續，先在不訓練的完整validation比較少數policy：

1. SiLU＋A8：matched control。
2. qSiLU＋A8：上游領先seed。
3. poly_quality＋A8：本次accuracy-oriented proxy。
4. poly_shift＋A8：hardware-oriented proxy。
5. 探索格可另保留poly_quality＋A7與qSiLU＋A6。

完整validation後才決定是否對少數policy啟動W8A8／QAT。weight的INT8、INT4、Fixed SD4、LS-SD4與ternary必須以完整activation policy為父節點獨立比較，不能把兩邊的獨立winner事後拼接。

## 產物與可追溯性

- 完整JSON：artifacts/reports/activation-smoke-v2.json
- JSON SHA-256：3c9301adaa1937f50bdd0c059f7d8000ea4699ce229e98c8ddb387651340c1f1
- CSV摘要：artifacts/reports/activation-smoke-v2-summary.csv
- CSV SHA-256：e5d76871d3119e681cdcb7707dc2c4a6a24ad3e35c54815c666c6c72a2f51ab6
- machine-readable實驗契約：configs/experiments/activation-smoke-v2.yaml

runner以每格原子寫入並支援resume；另以5格A8暫存報告驗證resume全部skip，結果5 passed、0 failed。

## 限制與未解風險

- upstream finalist仍pending／interrupted，finalization_ready仍是false。
- calibration每task只有2張、probe每task只有1張，不具統計代表性。
- smoke使用min／max初始化；MSE representative-batch初始化尚未比較。
- 尚未跑完整validation mAP、QAT、weight量化、integer export或bit-true end-to-end parity。
- 尚無FPGA／ASIC板上latency、power、resource數據。
- 因此不宣布activation winner、不判斷0.04 mAP gate、不宣稱硬體speedup。
