# V5 mAP50 總門檻與 backbone early bit 邊界報告

日期：2026-09-03。

## 結論

使用者已把正式精度要求明確修正為：**activation替換與weight量化合併後，八項`mAP50`相對accepted Full35的最差絕對下降不得超過`0.015`**。例如基準`0.900`時，候選至少要`0.885`；這是1.5個mAP百分點，不是相對誤差1.5%，也不是舊報告的`mAP50–95`。

同一批raw validation metrics經SHA-256驗證後重新判定，qSiLU＋A8／`backbone_early` W8仍為`green`：最差總下降是COCO box `-0.011408`，尚有`0.003592`總裕度；W8本身相對matched qSiLU＋A8的最差增量下降只有COCO person `-0.000781`。因此W8通過不是因放寬門檻，而是在新門檻下仍成立。

W7–W4再以完整COCO val與固定600張BBAT5 search-val逐格驗證後，只有W8通過。W7為`recover`，W6–W4為`reject`。所以目前最合理方向不是把整個`backbone_early`強壓到W4，而是先量測其餘區域W8，再對低敏感layer/group做mixed precision、Fixed SD4／LS-SD4／ternary；W7則保留給有限QAT recovery。

## 現行門檻

八項指標全部獨立過關，平均值不能掩蓋任何一項失敗：

1. COCO box mAP50。
2. COCO person box mAP50。
3. BBAT box mAP50。
4. BBAT pose mAP50。
5. BBAT ball box mAP50。
6. BBAT bat box mAP50。
7. BBAT ball pose mAP50。
8. BBAT bat pose mAP50。

判定規則：

- `total delta = candidate - accepted`，已包含activation function、activation checkpoint與A8，再加weight格式。
- 八項total delta都必須`>= -0.015`。
- W8另要求`candidate - matched activation parent >= -0.01`，用來隔離W8的額外影響。
- QAT必須有同recipe matched sham，任一指標的absolute drift不得超過`0.01`。
- 未通過硬門檻、但最差total delta仍`>= -0.04`者才進`recover`；低於`-0.04`直接`reject`。
- Pareto先套硬門檻，才在「最大化最差任務mAP50」與「最小化packed weight bytes」間選accuracy／balanced／hardware角色。

舊[`V4 W8報告`](../../reports/2026-09-03-v4-qsilu-backbone-early-w8-search-validation.md)仍保留原始`mAP50–95/-0.04`判定，不能被原地改寫。本輪[`mAP50 re-gate`](../../../artifacts/reports/v5-qsilu-backbone-early-w8-map50-regate-v1.json)直接驗證舊總報告與三個raw metrics檔的SHA-256，沒有重跑或挑選不同資料。

## W8 re-gate

| 指標 | accepted | matched qSiLU＋A8 | candidate＋W8 | W8增量 | 總下降 |
|---|---:|---:|---:|---:|---:|
| COCO box | 0.670719 | 0.659143 | 0.659311 | +0.000168 | **-0.011408** |
| COCO person | 0.839410 | 0.834097 | 0.833315 | **-0.000781** | -0.006094 |
| BBAT box | 0.976230 | 0.980255 | 0.980446 | +0.000190 | +0.004215 |
| BBAT pose | 0.976784 | 0.981293 | 0.981182 | -0.000111 | +0.004398 |
| BBAT ball box | 0.958383 | 0.967959 | 0.968939 | +0.000980 | +0.010556 |
| BBAT bat box | 0.994078 | 0.992552 | 0.991953 | -0.000599 | -0.002126 |
| BBAT ball pose | 0.959489 | 0.970034 | 0.969514 | -0.000520 | +0.010025 |
| BBAT bat pose | 0.994078 | 0.992552 | 0.992850 | +0.000299 | -0.001228 |

正delta不能解讀成量化必然提升；它只代表這次有限validation的觀察值。總門檻永遠取八項最差值。

## W8–W4 bit 邊界

所有低位元候選沿用同一accepted、同一matched qSiLU＋A8、同一calibration identity、相同evaluator及資料順序。本輪沒有訓練，也沒有formal validation。

| 格式 | Weight NRMSE | 最差mAP50項目 | 最差總下降 | 判定 |
|---|---:|---|---:|---|
| W8 grid | 0.005451 | COCO box | **-0.011408** | green |
| W7 grid | 0.010941 | COCO box | -0.020950 | recover |
| W6 grid | 0.021577 | COCO box | -0.044686 | reject |
| W5 grid | 0.044955 | BBAT ball box | -0.286769 | reject |
| W4 grid | 0.089494 | BBAT bat box | -0.971333 | reject |
| W4 exact | 0.088849 | BBAT bat box | -0.854778 | reject |

`optimal_scaled_codebook`讓W4的靜態NRMSE與實際最差下降都優於十點grid，證明exact baseline值得保留；但`-0.854778`仍遠離門檻，所以不能因重建誤差變好就promotion。這也直接證明static MSE／NRMSE不能取代task mAP。

W7相對matched的最差增量為COCO box `-0.009374`，但activation parent本身已相對accepted下降約`-0.011576`，合併後成為`-0.020950`。這正是activation與weight強耦合、total budget必須包含兩者的實際例子。

## 容量解讀

只量化`backbone_early`的1,218,240個weights，而其餘deployment weights仍以FP32計價，所以全模型weight storage的壓縮率只從W8的`1.0421×`微升至W4的`1.0495×`。為了多約0.7%的全模型weight容量收益，W4造成極大精度損失，不是合理交換。

因此下一步先找出其他region是否能以W8安全壓縮，再逐層找真正適合W7/W6/W5/W4或SD4的低敏感位置。isolated region delta不可相加；任何累積policy都必須重新跑八項mAP50。

## 後續實驗

1. qSiLU＋A8其餘九區各自做isolated W8 diagnostic與search validation。
2. 只用通過的區域建立逐步累積W8 policy，每加一區重新量測總下降。
3. 對臨界區做layer/group sensitivity；W4、Fixed SD4、Paper-TWN及後續LS-SD4只放在數值與task sensitivity都支持的位置。
4. 重新以mAP50總門檻評估uniform Hardswish、poly_shift與Q3 regional Hardswish，再做各自matched weight實驗，不能把qSiLU winner直接接上。
5. W7與mixed lower-bit若落在`recover`，才做fold-aware QAT。先5/10/15 epochs、matched AdamW sham與早期驗證；只有趨勢有效才延長20/60 epochs。MuSGD若測，另立paired sham，不能沿用AdamW optimizer state。
6. 鎖定最多三個Pareto finalists後才使用BBAT5 formal val、多seed、長訓練、packed export及硬體測量。

## 證據與限制

- 完整bit搜尋：[`v5-qsilu-backbone-early-bit-search-v1.json`](../../../artifacts/reports/v5-qsilu-backbone-early-bit-search-v1.json)，SHA-256 `fd4442f34df0077243aadd4b6d5855cbf37e1cb215a490e5b43c970191772d9b`。
- W8 mAP50 re-gate：[`v5-qsilu-backbone-early-w8-map50-regate-v1.json`](../../../artifacts/reports/v5-qsilu-backbone-early-w8-map50-regate-v1.json)，SHA-256 `67fce27506fac33e7655e791d815c04d7030ba213978a1ab27873369a396573e`。
- grid diagnostic：SHA-256 `00cfd6007a7a3ce57ddef11e3bb40ba336455787493184691a559767d9e29fd4`。
- exact W4 diagnostic：SHA-256 `a65cb8227c4ffd1e42dd4ba6b18da73a92ce1fbe58613b8c1282525212ccd0be`。
- BBAT5只使用canonical assignment與既有search-val；沒有重切、抽樣或修改資料。
- 本輪只證明一個activation parent的一個region；沒有QAT、formal、多seed、native integer kernel或硬體速度／能耗結果。
