# 三類主線之外：值得優先檢查的優化方向

日期：2026-09-08。範圍：COCO80 Detect＋BBAT5 Pose；person-only依使用者最新指示暫緩。本次只讀本地文字與一手來源，未執行GPU／模型forward／訓練／新AP評估。

主結論：先查影響訓練與評估的具體問題，再決定額外模組。最有直接本地依據的是 **Q/K凍結規則、fixed-scale多餘reduction、crowd ignore語意**；它們分別影響可學習範圍、執行成本與監督品質。Q/K可訓練範圍已收進[總計畫](<../../proposals/integrated-roadmap/plan.md>)的S7；fixed-scale冗餘是部署前的獨立parity／效能候選，不另開精度訓練大實驗。

## 1. 優先清單

| 優先 | 候選 | 目前證據 | 最小處理／對照 | 推論成本 | 何時停止 |
|---|---|---|---|---|---|
| A | Q/K trainability與STE有效更新 | 本地stage policy顯式hardware_frozen q/k/gamma | 先列逐參數grad/update；必要時一個scope對照，保持相同binary算式 | 需重fold/校準；不必增加per-image scale | 沒有可更新梯度，先修契約；有梯度但AP無收益不擴scope |
| A | Fixed scale inference early return | 現行coefficient先算dynamic再取fixed | fixed已ready且未calibrating時直接回傳；逐元素parity | 可省白做的abs/mean；速度需實測 | 輸出不等價就不採用；不要稱為精度回補 |
| A | 用既有predictions做錯誤分解 | final中保存prediction JSON；單看mAP不足以定位 | CPU分析漏檢、定位、重複、背景誤報與尺度 | 無 | 無顯著特定錯誤集中，就不新增對應模組 |
| B | COCO crowd ignore監督 | 現有converter跳過iscrowd，txt不表達ignore區域 | 先核對loss；若確有錯誤負監督，再做原loss vs ignore-aware loss | 訓練期處理，推論0 | crowd slice改善但一般AP/其他任務退化就不採用 |
| B | Fused graph量化敏感層 | RepConv融合等價不能推導INT8等價 | fused control/winner同校準，定位少數outlier層 | 少數層較高精度有代價 | latency收益不足或誤差不集中，停止加混精度 |
| C | Frozen standalone teacher的定位/feature anchor | joint person與ball-pose相對standalone有回歸，但原因未定 | 只有確認遺忘/定位錯誤後做一個masked anchor arm | 訓練多teacher；推論0 | feature loss下降但AP未改善就停止 |
| C | 訓練期高解析度auxiliary（P2） | 小物體可能受限，尚無本輪error slice證明 | 先量原P3 tiny-target occupancy；再另立單一P2 aux | 推論可0，訓練成本高 | 小尺度不是主因則不開 |

以上是研究排序，不是待執行queue。person-only、Lite64、完整P2 head、刪P5、全面RepConv均未列本輪新工作。

## 2. Q/K 凍結是值得檢查的訓練限制

本地 [stage_policy.py](<../../../yolo_combine/src/yolo_combine/stage_policy.py>) 同時存在：

~~~text
_HARDWARE_FROZEN_PARTS:
  .attn.qkv.q.
  .attn.qkv.k.
  .attn.score.gamma

_is_trainable(policy, stage):
  if policy.hardware_frozen:
      return False
  ...
  if role == attention:
      return stage.tune_attention
~~~

因此開tune_attention不足以解凍這些參數。Q/K projection的本地類型是從fused QKV拆出的Conv＋BN，見[projection.py](<../../../yolo_attention_final/final/yolo_attention/projection.py>)。

對BinaryQK：

~~~text
Q = Wq X，K = Wk X
S_binary = c · sign(Q)^T sign(K)

若Wq/Wk凍結：
  可以改X及其他下游參數間接適應
  不能直接更新Wq/Wk來改Q/K方向

若新增KD但Wq/Wk仍凍結：
  teacher提供訊號，不會自動解除requires_grad/optimizer限制
~~~

這只證明可學習路徑受到限制，**尚未證明它造成了多少AP損失**。凍結可能是既有硬體/穩定性設計，不能在本次研究直接當bug改掉。

建議QAT前量：trainable name、optimizer membership、grad是否存在、STE saturation、更新前後Wq/Wk差，以及score ranking變化。若允許離線更新後重新fold/量化，新QAT scope才開放Q/K。部署若要求原投影常數不可改，則此候選不適用。

score.gamma還有另一個陷阱：fixed係數已ready後，forward取保存的coefficient，gamma可能不在有效計算路徑。只把gamma設trainable沒有用；固定PoT訓練必須明確定義train-only重校準或可微固定槽的更新方式。

## 3. Fixed scale白做運算：確定的程式冗餘，收益待量

目前[BinaryScore._coefficient](<../../../yolo_attention_final/final/yolo_attention/binary_basis.py>)在辨識scale mode前先呼叫dynamic coefficient，再於fixed模式丟棄。正確候選順序：

~~~text
若 fixed mode 且 calibration未啟用：
    回傳已保存的 c_fixed（未ready仍必須報錯）
否則：
    計算dynamic magnitude
    若為dynamic mode，直接回傳（保留原本不累加校準的行為）
    若為fixed mode且校準中，累加觀測
    回傳dynamic值
~~~

這不能直接套用到其他basis的特殊路徑。訓練、動態模式、校準啟用與未ready錯誤行為各有原契約。針對現有Hadamard兩basis／兩sites，可避免每site四次、合計八次magnitude reduction。這是程式算式次數，不是八個GPU kernel或固定時間收益。

驗證需要fixed-ready输出／state完全一致，calibration count/sum一致，dynamic路径未變；最後量完整部署graph而非只量mean。它可能減少成本，不會提高精度。

## 4. 先用既有predictions定位失誤

可讀取[現有Bit-True prediction JSON](<../../../yolo_combine/final/full35/outputs/validation/accepted-best-joint/bittrue/detect/predictions.json>)，搭配相同GT/evaluator，在CPU進行重算與分類。TIDE將detection錯誤拆成分類、定位、兩者同時、重複、背景及miss，支持「先分解、再改模型」的流程；它不證明特定模組會增準。[TIDE作者頁](https://dbolya.com/tide/)、[原論文](https://arxiv.org/abs/2008.08115)

本專案值得看的slice：person、sports-ball、baseball-bat；small/medium/large；crowd/dense；score和IoU排序；person-negative圖的FP/image。Pose關鍵點錯誤需另外以既有pose evaluator檢查，TIDE不能直接解釋keypoint AP。

只有postprocess後的top-k predictions時，看不到被截斷的raw candidates；因此能觀察「漏檢結果」，不能直接判定是backbone沒表示、assigner失敗或top-k丟棄。既有JSON不足時記錄缺的raw trace，等將來有GPU權限才捕捉，不臨時重新跑validation。

## 5. Crowd ignore可能修正不合理負監督

本地已有調查：COCO train person crowd區域5,212個、涉及約4.41%影像；converter對iscrowd直接continue，見[person結構研究的資料稽核](<2026-09-04-coco-person-only-structural-specialization.md>)。這項問題也適用COCO80，不需先做person-only。

COCO官方evaluator對crowd/ignore有特殊匹配規則；txt略去crowd不代表training loss同步忽略那些候選。[COCO API原始碼](https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/cocoeval.py)

待驗證處理：保留原JSON crowd metadata為training-only side information；經與影像一致的resize/flip之後，只對沒有匹配正常GT、且屬該類crowd區域的負分類loss做ignore。正常positives優先，不能整片取消所有類別／box loss。Crowd region不是一個可直接當person正例的大框。

這需修改loss/augment契約，屬獨立單因子；資料本體保持唯讀，不新增split。不用BBAT5沒有的crowd標註去推測或補標。若稽核發現目前loss已有正確ignore，則只補驗證，沒有必要再開training arm。

## 6. 有些看起來合理的方向暫時不值得做

**Shared BN重估／分task BN：** `_apply_bn_modes`已把shared BN設eval，僅可訓練task heads的BN為train。PyTorch BN在eval使用running estimates；故「COCO與BBAT輪流刷新shared running stats」不是現有程式事實。[本地stage policy](<../../../yolo_combine/src/yolo_combine/stage_policy.py>)、[PyTorch 2.11 BatchNorm2d](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BatchNorm2d.html)

Frozen stats仍可能與新representation不匹配，但這是不同假說，需要activation分布與固定校準subset證據。直接加兩套shared BN還會讓同一影像的雙head不能自然共用同一份normalized features，不排首輪。

**更多HOG/WST/P2 loss：** HOG作prediction target有MaskFeat一手先例，但其設定為masked representation learning，並特別強調local contrast normalization；本案是unmasked supervised companion，不能照搬其增益。[MaskFeat](https://arxiv.org/abs/2112.09133)

本地舊HOG方案還需要定義CE用的9-bin概率分布、empty cell、energy權重，以及GT box邊界。若raw histogram H先除同一contrast常數c，再做每cell L1歸一化，則(H/c)/sum(H/c)=H/sum(H)，c被完全消除。即使如此仍可測orientation target，但不能聲稱實現了MaskFeat同樣的contrast機理。先把這個假說做清楚，再決定是否值得P2或其他filter。

**全域scale codebook：** 本地global dynamic相對fixed PoT只有+0.000505537，支持先處理site/ranking與QAT scope；不能由此斷言per-token/group永遠無用，但不值得先大掃4/8/16個global scales。

**RepOptimizer/QARepVGG：** 只有RepConv已帶來可重複AP收益，且post-fuse PTQ真的顯示outlier問題，才考慮量化友善重參數化。QARepVGG提供此風險的原始研究，不是本機兩層必定掉點的證明。[QARepVGG](https://ojs.aaai.org/index.php/AAAI/article/view/29045)

## 7. 本次工作與限制

已完成：讀取本地source／正式文字artifacts、外部一手來源；分析新增方向並排優先。未完成且未執行：Q/K梯度probe、HOG occupancy、TIDE/AP重算、crowd loss實作、fixed-return patch及CPU數值parity、硬體profile。不存在本次新精度結果。

所有正式BBAT5訓練仍固定Canonical BBAT5 v1；person-only維持暫緩。文獻只是機理依據，現有matched消融才是本地收益的判定來源。

困難：部分CVF HTML首次開啟失敗，改用原論文arXiv／作者官方頁核對；未以二手摘要替代證據。其餘尚未解的實作／測量問題已在上文逐一列出。

返回[研究索引](<README.md>)或[三類優化總計畫](<../../proposals/integrated-roadmap/README.md>)。
