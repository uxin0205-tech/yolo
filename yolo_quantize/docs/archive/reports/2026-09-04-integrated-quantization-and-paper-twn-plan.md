# 2026-09-04 Full35量化整合流程與Paper-TWN逐區規劃（開跑前基線）

> 本文保留V19鎖定前的規劃語境，不回溯改寫。V19、V29、recovery gate修正與目前短QAT的實際狀態，以[執行報告](2026-09-04-v19-v29-progressive-quantization-execution.md)為準。

## 結論先說

目前**沒有任何區域已被證明可以安全換成現有Paper-TWN靜態proxy**。既有的靜態`safe`名稱只代表weight重建門檻，不代表task mAP；實際完整search顯示：

| 相同paths | 格式 | 最差total mAP50 | 最差total mAP50–95 | 判定 |
|---|---|---:|---:|---|
| 3個Pose predictor paths | Paper-TWN | `-0.064370` | `-0.111516` | reject |
| 3個Pose predictor paths | exact W4 | `-0.008532` | `-0.012394` | green |
| 4 predictor＋3 tower paths | Paper-TWN | `-0.037325` | `-0.074978` | recover |
| 4 predictor＋3 tower paths | exact W4 | `-0.008532` | `-0.017466` | green |

這表示問題主要是Paper-TWN的三值表示能力，不是那些paths本身不能低位元化。舊safe route只有1,280個weights，從W8改成2-bit理論上只少約960 bytes；balanced route也只少約6,528 bytes，卻造成數個百分點的mAP下降，兩者都不是合理Pareto點。

因此後續不再一次替換整個Pose群組，而是依使用者指定順序：

```text
完成目前W8 QAT且鎖定parent
  → backbone逐path替換、選winner並固定
  → 在固定backbone上測neck、選winner並固定
  → 在固定backbone＋neck上測Detect/Pose head
  → W8/mixed、Fixed-vs-LS-SD4、ternary三條recovery lanes
  → 最多三個完整policy做activation coupling與formal
```

可供程式讀取的總流程是[`full35-integrated-quantization-roadmap-v1.yaml`](../../../configs/experiments/full35-integrated-quantization-roadmap-v1.yaml)，Paper-TWN動態矩陣是[`v28-paper-twn-progressive-region-plan-v1.yaml`](../../../configs/experiments/v28-paper-twn-progressive-region-plan-v1.yaml)。兩者是新增的orchestration文件；既有v5、V19、PTQ JSON、checkpoint與hash完全不改。

## 一、目前正在做什麼

V19 `poly_shift + LSQ+ A8 / all-W8` QAT仍在GPU上執行，沒有因本次規劃中止。matched sham已完成；兩個已發現的工程問題——fused head checkpoint contract與FP32 `0.2` blend ratio嚴格比較——都已red→green修正，QAT由正式epoch-0 `last.pt`精確續跑。

目前尚不能把epoch 0當最終結論。epoch 0最差total mAP50為`-0.017448`，只在COCO box超過`-0.015`硬門檻；最差mAP50–95為`-0.022242`，仍通過`-0.04`伴隨門檻。必須等15 epochs完成後再看是否恢復、best selector指向哪一輪，以及checkpoint hash是否完整。

父節點鎖定規則：

1. 若V19 QAT完成、雙族gate皆green且不被既有PTQ Pareto支配，就以其`best_joint`作後續逐區parent。
2. 若V19未通過，回到已green的九區W8、`backbone_attention_safe`維持FP32之policy。
3. 不會因Paper-TWN規劃重跑或中止目前訓練，也不會在parent未鎖定時偷跑下一個GPU矩陣。

## 二、Paper-TWN現在真正完成到哪裡

### 已完成

- 現有程式已把TWN v1/v2的`delta=0.7×mean(abs(W))`公式實作成靜態PTQ proxy：大於threshold者取`±alpha`，其餘為0；`alpha`是被選weights的平均絕對值。
- 專案現有proxy是layer-wise、2-bit、三個logical levels `{-alpha,0,+alpha}`；這不是完整Paper重現。原始TWN以FP32 shadow weights訓練，且論文演算法是逐filter求threshold／alpha。
- 三個activation parents、master／BN-folded deployment兩個views均完成靜態分析。
- `poly_shift + A8`下已跑兩組Paper-TWN完整COCO val與固定BBAT5 search-val，且有同paths exact W4 control。
- 現有QAT seam能做「保留Paper threshold、學layer-wise正尺度」的recovery，但這是專案learned-alpha extension，不能冒充原始filter-wise TWN結果。

### 尚未完成

- TWN v3使用的`0.75×filter mean(abs(W))`初始化，以及faithful filter-wise QAT尚未實作。
- exact-scaled ternary強PTQ baseline尚未產生完整148-layer CPU artifact。
- backbone、neck、Detect head尚未完成Paper-TWN單path search validation。
- 既有Pose結果是3或7 paths一起換，無法定位單一damage path。
- Channel-TWN、TTQ尚未各自實作／跑matched QAT。
- 沒有native ternary kernel、packing後latency、power或板上量測，因此只能報容量proxy，不能宣稱速度。

## 三、哪些部分值得測，哪些目前不值得

### 已排除的組合

- 舊`safe_candidates`三個Pose predictor paths：整組reject。
- 舊`balanced_candidates`四個Pose predictor＋三個Pose tower paths：整組recover且離門檻過遠。
- 全region直接Paper-TWN：現有全網deployment NRMSE約`0.678`，而且148層中沒有一層的Paper-TWN MSE優於matched W4，不值得直接花GPU做整區暴力測試。

### 第一優先：backbone單path

backbone先分`backbone_early`與`backbone_deep`；每個都先測一個path，通過才累加第二個。候選母集合優先取「Fixed-SD4相對exact W4在三parents×兩views都較佳」的36-layer manifest，因為它至少證明非均勻codebook有數值依據；這仍不保證ternary會通過。

| Region | 首輪path | Elements | Paper-TWN deployment NRMSE | 用途 |
|---|---|---:|---:|---|
| backbone early | `graph.model.2.m.0.m.0.cv2.conv` | 9,216 | `0.644851` | SD4-supported集合內第一名 |
| backbone early | `graph.model.2.m.0.m.0.cv1.conv` | 9,216 | `0.674137` | 第一格green後才累加 |
| backbone deep | `graph.model.8.m.0.m.1.cv1.conv` | 147,456 | `0.562025` | 有較實質容量收益 |
| backbone deep | `graph.model.8.m.0.m.0.cv2.conv` | 147,456 | `0.568146` | 第一格green後才累加 |

另保留一個依Paper自身靜態排序選出的sentinel，檢查「SD4-friendly母集合」是否造成selection bias；sentinel不因NRMSE較低自動晉級。

`backbone_attention_safe`先不測整區Paper-TWN，因為它連isolated W8都越過total gate。最多只保留`attn.proj.conv`單path sentinel，Binary Q/K完全不動。

### 第二優先：neck與MASF weights

只有backbone winner鎖定後才開始：

| Scope | 首輪候選 | 理由 |
|---|---|---|
| neck | `model.13.m.0.m.0.cv1.conv`、再`model.19.m.0.m.0.cv1.conv` | 兩者各147,456 weights，且屬SD4-supported母集合 |
| neck large sentinel | `model.16.cv1.conv` | 262,144 weights，容量收益較明顯 |
| MASF weight-only | `context.project.conv`→`dw5.conv`→`dw3.conv` | MASF整區W4曾green；只量weights，不刪或改MASF拓樸 |
| neck attention | `attn.qkv.v.conv` | 已在SD4-supported清單；Q/K維持Binary Q/K保護 |

這階段每加入一個path，都以「已固定backbone＋目前neck累積policy」重新跑完整search；不能把isolated delta相加。

### 第三優先：head

順序固定為Detect tower→Detect predictor→Pose tower→Pose predictor：

- Detect tower／predictor優先使用已跑過Fixed-SD4的相同paths。尤其`detect_head.one2one_cv3.2.2`的Fixed-SD4已green，適合作為Paper-TWN／exact ternary的公平同路徑壓力測試。
- Pose tower與predictor只做單path起跑，禁止直接重用舊3-path／7-path組合。
- predictor雖小但語意敏感；若容量節省只有數百bytes，即使green也只算diagnostic，不自動成為final policy。
- one-to-many、pose-flow與sigma仍保留訓練supervision，但不計入部署壓縮收益。

## 四、每一個path怎麼公平比較

同一個精確path集合先建立五格；其中v3 filter-wise格若CPU明顯被支配，可不進GPU：

1. locked parent不變，作matched baseline。
2. exact uniform W4。
3. `TWN-v2-0.70-layerwise-static-proxy`，保留舊結果可重現性。
4. `TWN-v3-0.75-filterwise`初始化／QAT分支，貼近新版論文粒度。
5. exact-scaled ternary PTQ，使用相同`{-s,0,+s}` codebook但全域MSE最佳scale。

這能回答三個不同問題：

- 現有0.70 layer-wise proxy比W4差，是因為2-bit容量不足，還是threshold、scale或粒度不適合該層？
- exact ternary能否在同樣2-bit下追回部分差距？
- filter-wise TWN、learned scale或TTQ是否值得花QAT成本？

CPU的NRMSE、cosine、zero ratio與code occupancy只用來排序；winner必須看完整COCO 5,000與不可變BBAT5 search-val 600的八項mAP50及八項mAP50–95。

## 五、階段gate與回退

最終硬門檻不變：activation替換與weight量化合計後，八項mAP50每項delta都必須`>= -0.015`，八項mAP50–95每項都必須`>= -0.04`。

為避免backbone／neck先把全部accuracy budget吃完：

- 進head前，暫定至少保留mAP50 `0.002`、mAP50–95 `0.005`的餘裕，也就是階段winner最差值需分別優於`-0.013`與`-0.035`。
- 若新path失敗，立即回到上一個green policy；不把失敗path帶進下一區。
- accuracy差在`0.002`內才以packed bytes較少者勝出；差距更大時選精度較高者。
- 若某階段沒有ternary cell通過，合理答案就是該階段保留W8或FP32，而不是強迫一定要有三元層。

## 六、何時需要重新訓練

- 靜態TWN proxy／exact ternary PTQ若直接green，不需要為了形式而QAT；但不能把它宣稱成原始TWN訓練結果。
- recover候選只有在最差mAP50仍`>= -0.04`且容量收益有意義時，才進15-epoch短QAT。
- 第一輪最多8個短候選（6個主候選＋2個預註冊sentinels）；每個採3 epochs FP32、6 epochs progressive ramp、6 epochs full ternary，不從零訓練。
- AdamW作matched主control，Detect logical batch 128／physical 16、Pose physical 16，loss權重1.0／0.25，沿用accepted augmentation，不加noise。
- Faithful filter-wise TWN、Channel-TWN與TTQ必須分開命名、分開matched sham。TTQ使用正／負兩個learned scales，適合固定TWN接近門檻但恢復不足的候選；不是PTQ替代名稱。
- 若一般progressive QAT仍不足，可另做INQ-style layer內逐步量化比例`20→40→60→70→80→85→90→95→97.5→100%`；這只是借用INQ排程，外層backbone→neck→head順序仍是本專案設計。
- MuSGD若測，也必須另立paired sham，不能混入AdamW歸因。
- 只有短QAT顯示可恢復者才延長20／60 epochs；reject cell不靠盲目長訓練硬救。

## 七、完整整合後的實驗量

不是一次丟出全部組合：

| 階段 | GPU search上限 | 進入條件 |
|---|---:|---|
| 目前V19 W8 QAT | 已在執行 | 不終止 |
| backbone Paper/exact ternary | 12格 | V19完成並鎖parent |
| neck Paper/exact ternary | 12格 | backbone winner已鎖定 |
| head Paper/exact ternary | 16格 | backbone＋neck winner已鎖定 |
| recovery QAT | W8/mixed、SD4/LS-SD4與ternary分開；ternary最多8格＋matched sham | 各lane的PTQ/QAT前置gate通過 |
| activation coupling | 最多3個weight policies | ternary／W8／SD4 Pareto已形成 |
| formal／multi-seed | 最多3個finalists | 所有search gates通過 |

每階段使用successive racing：先單path，通過才累加；第一個失敗就停該chain。這樣能遵守「骨幹確定，再調neck，再調head」，又不會把數十個明顯無望cell送進完整GPU validation。

既有結果不會被新ternary lane覆蓋。V19 W8/mixed-bit是uniform recovery lane；Fixed-SD4 Detect predictor/head與LS-SD4是dyadic lane；Paper/static exact/TTQ是ternary lane。三條使用相同parent與metric contract，各自保留accuracy、balanced、hardware角色後才做跨lane Pareto。weight policy確定後才重新驗證qSiLU、Hardswish、Q3 regional Hardswish與A7/A6；A-SD4維持獨立研究分支。

## 八、目前建議

1. 不把舊Paper-TWN safe/balanced route放回主線；它們是重要反例。
2. 等目前V19 W8 QAT自然完成，先決定後續locked parent。
3. CPU先補TWN-v3 filter-wise與exact-scaled ternary，並對148 paths產生0.70 proxy／0.75 filter-wise／exact ternary／exact W4同路徑表。
4. GPU下一輪只做backbone單path；沒有green就保留W8，仍照順序進neck，不強迫backbone一定三元化。
5. 最可能有實際價值的不是數百weights的Pose predictor，而是能通過gate的backbone-deep、neck或Detect tower中型path；但這只是優先順序，尚不是精度結論。

## 證據與限制

- Paper static profile SHA-256：`306deaa51d29f16c3ed49b1a442f513fb09813a0966ba1960544816cfde48f24`。
- 舊Paper routing manifest SHA-256：`05540a7fcc1f5c5adbc87b65a6c49d90446162d526cda5ac634746fe9885f145`。
- V24 dual regate SHA-256：`bdd9cd2169e4f1827fdcf71ee317ed5c63cfe5dfd24baba46cb111fa193feade`。
- V19 plan SHA-256：`2745da54f419b529233da5d149c617981d9ad9ffe55b6b5291d84676e47e10a8`。
- 原始文獻查核見[`Paper-TWN／TTQ／INQ與exact ternary`](../../research/2026-09-04-paper-twn-primary-literature.md)。論文沒有提供YOLO backbone／neck／head逐區winner，因此本計畫的placement仍必須實測。
- 本報告沒有啟動新GPU工作、沒有碰formal val、沒有改BBAT5 split，也沒有改寫任何既有結果。
