# 論文改善為何未直接出現在本地 MASF：ball／bat 與尺寸分析

## 結論

目前結果只支持「這個局部微調版本的 MASF 沒有取得足夠額外收益」，不支持「MASF 永遠無效」，也不能僅因完整流程優於原 B100，就把改善歸功 MASF。使用者所指的特定論文尚未在本次指定，不假設本地 Full35 MASF 就是該論文完全相同的模組／配置。

本研究依 research 技能查核一手來源，由主代理完成，未使用子代理。只讀資料及 CPU 尺寸統計，未啟動 GPU、訓練或新推論。

## 原始研究與本地方法並非同一件事

FPN 的多尺度設計是以 top-down 與 lateral connections，建立各尺度的高階語意特徵；其結果不是「任何額外多尺度卷積都必然有同樣收益」的保證。[FPN 原論文](https://arxiv.org/abs/1612.03144)

本地 MASF 則是同一個特徵解析度上的 DW3／DW5＋1×1 混合。增加鄰域資訊，不增加輸入解析度；P2 版本沒有新增 stride4 預測 head，最細 Detect 仍是 stride8。這是實作的結構事實。是否因已有 Neck 融合而功能重複、或局部背景干擾小球，是可檢驗假設，尚未證實。

搜尋命中的 EFPN arXiv:2003.07021 已被作者撤回，頁面明示重要主張缺少充分實驗，本報告不把其效果作為方案依據；也不以此概括其他論文的可信度。[撤回狀態](https://arxiv.org/abs/2003.07021)

## 已證實的本地條件與可能限制

1. 新方向 P3 MASF 是既有模型插入後的局部適應，主要更新 P3 head／MASF，其他原有參數與 BN 統計固定；不是全網長訓的同義詞。P2 更新所有原 Detect head 加 MASF，但原 backbone／Neck 仍固定。本次 5 epoch／續訓到 E8–E10 是否足夠，不能只憑總 epoch 數判定。
2. 原生 one-to-one 對 MASF 的直接梯度被 detach 隔開，已有實際校準證據；one-to-many 仍有梯度。bridge 已接通該路徑，但精度提升很小，所以這不是已被證明的唯一原因。
3. 新方向 P2／P3 權重用 COCO 訓練；在 BBAT5 只做直接驗證，沒有使用 BBAT5 train 做這些 MASF 候選的專項適應。因此這是跨資料集泛化結果，不是完成棒球專用訓練的效果。可能的領域差異不能冒稱已量化的原因。
4. shared P3 MASF 會影響下游 P4／P5；Detect-only 版本隔離了這條影響。但現在不能僅由一個總 AP 判定收益或損失發生在哪個尺寸／錯誤類型。

以上限制不等於證明「再多訓練就一定有效」。目前只足以拒絕直接將未通過的候選升格，不足以宣布模組在所有合理訓練下都無效。

## BBAT5 真實尺寸統計

資料為 canonical `bbat5-v1/pose` 完整 validation：683 張、932 個框。逐張讀原圖尺寸與原 labels，使用原圖 box 面積 `<1024 px²` 統計；另以等比例最長邊縮放至 640 計算短邊。只統計，不修改 labels 或 split。

| 類別 | 總框數 | 原圖框面積 <32² | 比例 | 640 輸入等比例縮放後短邊 <8 px |
| --- | ---: | ---: | ---: | ---: |
| ball | 393 | 291 | 74.05% | 12（3.05%） |
| bat | 539 | 72 | 13.36% | 0 |

本次以縮放後面積 <1024 統計也得到相同數量，兩类短邊 <4 均為 0。上述是框尺寸分布，**不是 APsmall**，亦不是 COCO 官方 segmentation-area 統計。

COCO bbox 評估會依 `areaRng` 分 small／medium／large，small 上限為 32²、medium 上限為 96²；還有範圍外 GT／未匹配預測的 ignore 處理。不能僅刪除大框標註再直接算普通 AP，否則可能錯計 false positives。[COCO 官方 evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)

因此 ball 類別較多小框，但不全是小框；bat 多數不屬該框面積區間，還可能有細長形狀問題。不能把 ball／bat AP 直接命名為小物件 AP。短邊小於 stride 也不代表一定無法辨識，只是定位與特徵保留可能更敏感。

## 已有 BBAT5 指標是否改善

下表是已完成、同 epoch 候選減對應無 MASF control 的 **BBAT5 box AP50–95**，單位為百分點；不是 keypoint AP，也不是 APsmall。

| 方案 | ball ΔAP | bat ΔAP |
| --- | ---: | ---: |
| P3 shared E5 | +0.0174 | -0.1510 |
| P3 Detect-only E5 | -0.0724 | -0.0274 |
| P3 bridge E8 | +0.0105 | +0.0075 |
| P2 shared E5 | -0.0957 | +0.0606 |

bridge 對比 Head control 的結果略升，但微小且沒有多 seed 或統計檢定；P2 的 bat 小升不能抵銷 ball 小降並稱「小物件都提升」。原始數字見 `combine/artifacts/existing-masf-bbat-v1/summary.json`。bridge 對無 MASF control 是整體方案比較，不是只隔離 bridge 的因果比較。

## 最小下一步：先分錯誤，再決定是否訓練

若使用者後續允許新驗證，先固定目前 control／bridge／P2 權重和推論設定，保存完整預測，再以相同 evaluator 計算 ball-small、ball-non-small、bat-small、bat-non-small 的 AP／AR，並分 AP50／AP75。按尺寸核對漏檢與定位失敗，不從總 AP 猜測。

漏檢增加才優先檢查高解析特徵、監督覆蓋或 domain 適應；AP50 尚可而 AP75 較差時檢查定位；大量背景誤檢則先看分類與上下文。這些是診斷方向，不是在沒有結果前直接換模組或推進新訓練。

像素誤差的純幾何例子：相同 8×8 框水平偏移 2 px，IoU=48/80=0.6；32×32 框同樣偏移 2 px，IoU=960/1088≈0.882。這說明小框對定位偏差敏感，但不是本次已觀察到模型偏移 2 px 的證據。

目前仍未計算正式尺寸分組 AP，也沒有新 P2／P3 Pose keypoint head；不把已有類別 AP 冒充這些結果。維持停止，不啟動後續工作。
