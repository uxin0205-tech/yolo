# 方向1完整規格：固定預算的物件／關鍵點區域排序蒸餾

日期：2026-09-08。代號：`R2-REGION`。狀態：`proposed`，不是ready-to-run。它是第二輪訓練方法，不是MASF方向1，也不是Detect head教Pose head。本文補足並優先說明[第二輪摘要](<plan.md>)的區域KD部分；不改第一輪架構或硬體契約。

## 1. 目標與非目標

目標：在Detect80＋Pose2共用trunk的前提下，測試FP-QK joint teacher能否補回BinaryQK遺失的任務相關排序；若普通KD有效，再測同樣蒸餾配對預算下，物件／有效關鍵點配額能否額外保護小物件及Pose。

不主張：蒸餾能自動解決合併任務衝突；老師一定比學生全面更好；注意力大就代表因果重要；前景mask本身是新發明。若FP joint本身已有相同回歸，模仿它不等於拿回standalone能力。

固定不變：COCO80 Detect、BBAT5 Pose、P3/P4/P5、第一輪選定的MASF／RepConv／BinaryQK sites／固定PoT。person-only、P2 head、新基底、多teacher、額外box/KL/feature loss均不一起加入。

## 2. 現在、訓練期提案、部署圖

目前：

~~~text
圖片 → 共用Backbone／Neck ─┬→ Detect head → 原生Detect loss
                          └→ Pose head   → 原生Pose loss
~~~

提案：老師和學生都已是joint模型，兩份權重各自獨立：

~~~text
同一張完成augmentation的圖片
  │
  ├→ FP-QK joint老師（eval、凍結）
  │    共用trunk → 對應task head
  │       └→ 指定attention site的score ──────────────┐
  │                                                │
  └→ BinaryQK joint學生                             ├→ 排序蒸餾loss
       共用trunk ─┬→ Detect head → Detect loss       │
                  ├→ Pose head   → Pose loss        │
                  └→ 同一site的live score ───────────┘
                           ↑
既有boxes／有效keypoints → token支持區域與配對配額

老師不更新；只有學生按已核准scope更新。
~~~

部署：只有更新後的BinaryQK joint學生，仍是一份trunk＋兩個heads。teacher、區域選取、row標準化與KD loss全部移除；不得把訓練輔助運算留進export。

這是同task教師對學生的監督，不是Detect預測框當Pose真值。即使teacher有兩個head，training helper也只需對應task的輸出／共同score；首版不為省算而另外改forward圖。

## 3. 老師、學生與資料入口

| 項目 | 契約 |
|---|---|
| 老師R2-T | 第一輪同架構FP-QK control，已完成Detect＋Pose訓練；真正qᵀk/√d，不是名字叫Float的binary surrogate |
| 學生R2-P | 第一輪採納的未fuse、fixed-PoT BinaryQK joint模型；不是唯一INT8部署檔 |
| 一致項目 | 任務類別、heads、尺寸、site layout、bias／normalizer類型、MASF／RepConv結構、資料与evaluator |
| 可不同項目 | QK精度及其訓練後權重；這是teacher/student關係，不要求兩者所有權重相同 |
| 必備記錄 | 各自checkpoint hash、graph/config digest、來源／訓練血緣、stage／optimizer／BN狀態、seed、backend |

FP teacher是否存在、是否在想修復的錯誤上提供較好訊號尚未驗證。不能直接把現有Full35 Float改名成R2-T，也不能用同一學生複本假稱取得新能力。需要兩個standalone老師來補joint遺忘時，是另一個研究問題，不在本案。

- COCO80：`/home/uxin/yolo/coco2017.yaml`。
- BBAT5 Pose：`/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml`。
- Registry：`/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。
- 不重切／增減影像、不修改標註；BBAT5沒有test split。只使用正式train和已核准的既有calibration manifest，不自行另抽BBAT5 View。

COCO batch只用其Detect原生loss與box支持區域；BBAT5 batch只用其Pose原生loss與box／有效keypoints。不可因teacher也輸出另一head就補造另一task的GT或把未標物件當負例。兩task每個macro的批次數與權重沿用已核准joint配方，不在本實驗更動。

## 4. 先判斷是否適合做

1. 用同evaluator核對FP joint與Binary joint的整體和區域差異，區分「合併前後的回歸」與「二值化後的回歸」；不同訓練血緣的差值不能稱純QK因果效果。
2. teacher在待改善slice須有可驗證的較好訊號；高attention margin只是取樣可靠性代理，不是teacher正確的證明。
3. 確認Binary score的梯度能到達實際trainable student參數。
4. 先跑普通ranking KD的配對對照。若沒收益，本輪預算策略是先停止，不代表數學上證明區域KD永遠無效。

沒有合格teacher、gap／teacher訊號不支持，或score沒有可更新路徑，就保留proposed，不直接開區域創新實驗。

## 5. 蒸餾位置與最重要的梯度陷阱

首版只選第一輪診斷中最敏感的一個仍為binary的attention site：`model.10.m.0.attn`或`model.22.m.0.1.attn`，不是兩個同時開。需要解析新joint graph的實際owner，不硬套舊字串白名單。這些不是新加的P3 attention；GT需對齊所選site的實際H×W，不能直接套P3解析度。

以[本地attention.forward](<../../../yolo_attention_final/final/yolo_attention/attention.py>)為依據，首版KD tap定義為：

~~~text
Q/K → score → 既有progressive（若配置仍啟用）→ relative bias
                                                   │
                                                   ├→ live score → KD
                                                   └→ 原normalizer → P×V → 原輸出
~~~

teacher/student同一tap，progressive狀態需在比較中固定。其shape為[B,heads,N,N]，N=H×W；本地q/k map依view把空間攤平，座標mapping需驗證。

目前`self.last_scores = scores.detach()`只適合診斷。**學生KD不能讀這個buffer**：即使loss數字會變，也不代表它能更新學生。需設計training-only、短生命週期collector取得bias後、normalize前的live tensor；teacher可detach。每個microbatch結束後釋放，不把整個epoch的graph存在module屬性。

### Q/K凍結與KD有效性

- 固定Wq/Wk，不代表所有排序KD必然無效：若STE允許對X傳梯度，且上游X的產生器可訓練，仍可經X間接適應。
- 若sign／bit-true路徑切斷梯度，或所有score上游都凍結，只有後面的P/V／heads可訓練，這個score-ranking loss就不能靠下游參數改好。
- 現行hardware contract保護Q/K、gamma、fixed coefficients／PWL，原joint配置還有`qk_ste:false`。不得直接刪assert、解凍常數或重新校準來讓測試通過。
- 三個KD arms使用同一已核准training scope。可採固定Q/K＋有效STE＋上游適應；若第一輪另有合法可更新Q/K版本，則整組採其scope及校準規則，不在arms之間換scope。

實作前需保存逐參數requires_grad／optimizer membership、KD-only梯度、實際update與硬體常數hash；不能只看總trainable數或總loss下降。

## 6. 固定蒸餾預算如何分配

以下是**新訂首版提案值，未實測**；以控制搜索量為目的，不是最佳超參數。

| 設定 | 首版提案 |
|---|---|
| KD site | 一個，由第一輪既有診斷選定 |
| Pair上限B | 每圖、每個選定site共512個；在heads間均分，不是每head512 |
| COCO K-REGION query配額 | 物件box支持區域75%、box外候選25% |
| BBAT5 K-REGION query配額 | 一般物件box區域50%、有效keypoint鄰近區域25%、box外候選25% |
| 區域內分配 | per-image後per-instance均分，餘數按可重現seed／step輪替，避免永遠偏前幾個物件；不讓大box按面積吃掉大部分預算 |
| K-FG | COCO／BBAT5皆前景75%、box外25%；不分instance、不用keypoint特殊配額 |
| K-UNIFORM | 同一eligible pair pool內均勻選取，沒有GT區域配額 |

用完成同一augmentation的GT建立temporary loss mask，不改labels。keypoints以本地visibility≠0為有效，實際`kpt_shape=(2,3)`；不猜ball/bat兩點的解剖語意、不加人體骨長／左右對稱loss。

box支持以與token cell的正面積交集建候選；極小物件至少對齊一個有效cell，keypoint取最近有效cell及一圈8-neighbors，裁切至實際特徵圖。ROI重疊採固定去重規則；BBAT5 keypoint區域優先分桶，普通box桶使用餘下支持，避免同tuple重複計費。多個物件落同cell時去重並記錄alias率；這個方法不能創造所選site已丟失的空間解析度。

box外只是「非已知GT支持區」的取樣集合，不宣稱全是真負例；不對它加額外負分類標籤。COCO crowd／未完整標註風險另記，不藉此重標。

### 各arm共用eligible pair pool

1. teacher對每個head／query的非self keys排序；高分候選a來自前4名，b由其他非self keys以固定seed抽取，a≠b。這是teacher-score比較，不是「正GT key／負GT key」標記。
2. 所有KD arms以相同規則產生pool，再按各arm的query配額選tuple(h,i,a,b)。先依第7節teacher margin剔除tie／不可靠pair。
3. 不足的桶依「其他前景桶→box外桶→全pool」回填，必須去重；pool不足B就所有arms共用相同較小上限，不重複同pair假稱補滿512。
4. 保存attempted／eligible／selected數、各桶實際占比、empty-region／small-object alias率。若某影像各arm選到同一集合，要如實計入沒有區域處理差異的影像比例。

上限相同不代表所有影像實際有效pair數必然相同。初始化前建立相同有效pool可降低此差異，但仍需報實際數；不能只印B就宣稱完全同算力。

## 7. Loss、normalization與task整合

S_T、S_B皆為bias後／normalize前的score。為降低把整列score放大就降低排序loss的捷徑，首版对每一query row在有效key域做訓練輔助標準化：

~~~text
z = (S - mean_keys(S)) / max(std_keys(S), ε)
ΔT = zT[i,a] - zT[i,b]
y  = +1 if ΔT>0 else -1
Lpair = softplus( -y · (zB[i,a]-zB[i,b]) / τ )
~~~

新提案值：ε=1e-4、τ=1、teacher有效margin |ΔT|≥0.1、w=1不另加confidence權重。teacher的std<ε則該row不入pool。student標準化分母保留梯度，且其低方差／大梯度case必須通過安全probe；這不是放進部署路徑的dynamic scale。std定義採population std以免N很小出NaN。teacher由no_grad產生，y及選取索引不反傳。

上述score標準化是本次補齊的KD比較契約；K-UNIFORM、K-FG、K-REGION必須相同。標準化／softplus以FP32計算並保留student梯度，三臂採同樣AMP邊界。不能把它與舊未標準化KD結果當同一控制。

每圖對實際有效pairs平均得到ℓKD；沒有pair則ℓKD=0，原生negative／box／keypoint loss照常。不同圖、task、microbatch不可直接拼全部pairs一次平均，避免多GT或大batch隱性改權重。

~~~text
L_D' = L_D_native + μD · mean_images(ℓKD_D)
L_P' = L_P_native + μP · mean_images(ℓKD_P)
L_joint' = 既有macro對L_D'／L_P'的task加權與reference-batch規則
~~~

本地native loss為batch-sum介面；實作應以μt·Σ_images ℓKD加入該task的raw_total，再交既有MacroStepEngine，不要對native loss再除一次batch。KD項另記log，不冒充原生DFL或RLE分量。o2m/o2o原loss與progressive schedule保持不變；共享score的KD每圖只加一次，不因兩分支再算兩份。

μD／μP用同一train-only校準規則選一次：在KD確實可達的active shared參數上，以KD/native梯度norm比0.1為目標、0.05–0.15為初始安全區間。各KD arm可有不同數值μ，但校準資料量／程序／目標相同並鎖定；正式val不得調μ。native梯度接近0時跳過該校準樣本並記錄，不把μ推到極大。

首epoch把μ從0線性升至校準值，其後固定；所有KD arms一致。LR／optimizer／最大更新長度繼承第一輪採納的QAT recipe，而非本次新掃。若第一輪尚未產生可重播trace，就保持未定；20 epochs僅是第一輪提案上限，不是本案已成功的訓練長度。

## 8. 最少實驗順序與成功條件

| 階段 | Arm | 回答什麼 | 何時啟動 |
|---|---|---|---|
| 前置 | FP joint／Binary joint及KD-only梯度查核 | 老師是否有訊號、學生能否學 | 必須先完成；本次未跑 |
| A | K0：相同recovery、不加KD | 普通繼續訓練本身能補多少 | 首輪控制 |
| A | K-UNIFORM：普通ranking KD | 蒸餾對joint學生有沒有用 | 與K0同起點／更新長度 |
| B | K-REGION | 同配對預算下，區域分配是否額外有效 | K-UNIFORM先有訊號才做 |
| C | K-FG | 是否超過普通foreground KD | K-REGION有訊號才補 |
| D | 最关键control/candidate paired seeds1/2 | 收益是否可重現 | 只對初篩通過者 |

初篩最多四個training arms，但不是預設四個都跑；普通KD失敗可在兩臂後停。K0先鎖定更新次數／LR trace，後續arms從同一R2-P重播，不能從上一個winner接續。teacher亦固定同一R2-T，不隨student更新，不用EMA換掉研究問題。

K-UNIFORM對K0、K-REGION對K-UNIFORM、K-REGION對K-FG各自比較：COCO overall或既有joint score至少+0.001，八項各不低超過0.001；要聲稱任務保護，事前鎖定的弱項另需+0.002。這些是工程gate、非統計顯著性，也不保證跨seed通過。

八項為COCO overall/person、BBAT box/pose、ball box/pose、bat box/pose。另報COCO AP_S、sports-ball、baseball-bat、Pose尺寸slice與區域ranking reversal／tie率；internal/canonical COCO evaluator不可混表。

K-REGION只勝K0不足以證明區域機制；只勝uniform不足以證明超過既有foreground KD。即使ranking改善，AP未改善仍不採用。若teacher無優勢、梯度不通、任一保護指標超限或跨seed不穩定就停止；不堆第二種KD loss補故事。

## 9. 成本、工程工作與驗證

新增訓練工作：凍結teacher載入／forward、live-score collector、GT-to-token mapper、共用pair pool、ranking loss與原生macro整合。這些程式本次均未實作。

512只限制loss處理的pairs，不限制teacher完整forward或N² scores；row標準化／候選pool建立亦有成本。相比K0，KD通常更耗訓練時間和記憶體，不能宣稱同算力；三個KD arms間盡量固定teacher／pool／pair預算並報wall-time與peak memory。離線cache只有在相同影像及augmentation可重現時有效，不強行套用不同random crop的舊score。

部署預期沒有新增模組／每圖scale，但必須從同一student權重比較移除helper前後eval輸出、op catalog及實際target成本。若student權重有變，仍需依原契約產生部署副本與校準／PTQ驗證；不覆寫唯一權重。

必測清單：teacher eval/無梯度、student KD-only梯度與update、mu=0原圖等價、detach誤接能被測試抓到、row/tie/zero/NaN處理、GT augmentation與token mapping、tiny-object/empty-region回退、heads/pairs公平性、microbatch/full-batch等價、AMP overflow replay不重覆計數、resume RNG／μ／pool規則與trace一致、硬體常數hash不變、export無KD。

本次僅完成規格及文字查核，沒有測試成功數值、GPU工作、AP或latency結果。既有研究狀態不因文件完整就改成ready。

## 10. 創新邊界與待鎖定事項

[FGD](https://arxiv.org/abs/2111.11837)已有foreground/global distillation；[Hinton等人的蒸餾](https://arxiv.org/abs/1503.02531)提供teacher訊號的一般框架。本案可能的貢獻是：joint Detect/Pose情境下，針對BinaryQK、在固定pair預算內做per-instance／有效點支持分配，並以matched controls證明任務保護。尚未做完整新穎性檢索，不宣稱首次。

首版已提出B、配額、支持區域、ε／τ／margin與μ規則；尚待第一輪結果鎖定的是實際R2-P／R2-T、teacher優勢、單一site、可訓練scope、合法cache manifest、recipe／trace、弱項指標與裝置成本預算。這些缺口不能用虛構checkpoint／成功數據填滿。

返回[第二輪入口](<README.md>)；本次工作見[中文紀錄](<../../docs/worklogs/2026-09-08-region-ranking-full-spec.md>)。
