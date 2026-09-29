# 第二輪創新分析：不是新增模組，而是保住二值化後的任務資訊

日期：2026-09-08。研究範圍為現行YOLO26M／Full35脈絡，COCO80 Detect＋BBAT5 Pose；person-only不在本輪。以下公式是工程推導與候選規格，不是新AP或硬體測量結果。

## 1. 先說什麼已經有人做過

- ITQ早已研究用旋轉降低二值編碼誤差，因此「先旋轉再sign」不是新的基本概念。[ITQ作者原論文](https://slazebni.cs.illinois.edu/publications/cvpr11_small_code.pdf)
- QuaRot與SpinQuant已採共同Q/K正交變換及signed Hadamard；LLM低位元量化的效果不能直接換成本案BinaryQK效果。[QuaRot](https://arxiv.org/abs/2404.00456)、[SpinQuant](https://arxiv.org/abs/2405.16406)
- FGD已區分前景／背景並蒸餾關係；「GT mask＋KD」本身不是創新。[FGD](https://arxiv.org/abs/2111.11837)
- CrossKD把student中間特徵經teacher head產生蒸餾輸出，並非Detect/Pose互教的同義詞。LD的分布KL需要box bins；不能因loss仍叫dfl就認定YOLO26有這些bins。[一手證據與適配限制](<2026-09-08-round2-primary-evidence.md>)

本次提出的可能貢獻是**固定成本下的選擇準則與任務保護機制**，不是把上述方法重新命名。尚未做全面prior-art檢索，兩個提案都只能稱「創新候選」。

## 2. 本地起點與研究動機

現有[BinaryScore](<../../../yolo_attention_final/final/yolo_attention/binary_basis.py>)採identity／Hadamard雙basis，固定模式使用離線coefficient。第一輪已規劃site isolation、QAT、KD及scale；第二輪不重複它們的baseline。

歷史global dynamic比固定PoT只高約0.000506，支持先檢查ranking而不是先掃更多global scales，但不是ranking已被證明為唯一根因。[先前評估](<2026-09-08-additional-optimization-priorities.md>)

現行[joint loss](<../../../yolo_combine/src/yolo_combine/joint_loss.py>)已有task-weight與progressive o2m/o2o介面；[正式joint配置](<../../../yolo_combine/final/full35/configs/joint.yaml>)還包含固定task weight與`qk_ste: false`。第二輪必須承接第一輪核對過的可訓練scope，不能直接把現有production設定叫QAT。

本地子代理限定範圍查核補充（未載入模型）：

- [joint source](<../../../yolo_combine/src/yolo_combine/source.py>)實際建立Pose26的`kpt_shape=(2,3)`，不是人體17點；[正式Pose YAML](<../../../original/pose/derived/bbat5-v1/configs/pose.yaml>)只有ball/bat。兩點的端點名稱／順序未找到正式語意，不能自行加人體骨長或左右對稱loss。
- 本地Ultralytics loss（本機／歷史參照：`../../../yolo_combine/.venv/lib/python3.12/site-packages/ultralytics/utils/loss.py:1156`；未隨本次報告發布）已有E2ELoss與TAL：o2m topk=10，o2o topk=7且topk2=1；reg_max=1的DFL不啟用。Pose26已有flow/RLE，不能把RLE當新提案。
- Q/K projection為Conv+BN+Identity，沒有SiLU；每site推導4 heads，Q/K維度是`key_dim=32`，V的`head_dim=64`不同。本地attention（本機／歷史參照：`../../../yolo_combine/.venv/lib/python3.12/site-packages/ultralytics/nn/modules/block.py:1298`；未隨本次報告發布）、[fold實作](<../../../yolo_attention_final/src/yolo_attention/projection.py>)
- [hardware_contract.py](<../../../yolo_combine/src/yolo_combine/hardware_contract.py>)明確保護Q/K、gamma、fixed coefficients／PWL。**目前合法流程不是任意可refold／recalibrate**；R2-BASIS需另立可重建的模型／硬體版本契約，這次沒有變更或授權該契約。
- 限定來源搜尋沒有找到可執行的common-QK sign-D、learnable Hadamard或teacher KD；未掃所有歷史runs，所以不能宣稱從未做過。全套歷史去重留待正式立項。

## 3. R2-BASIS：保護兩個任務的固定基底選擇

### 原本 → 提案

以下以每個token的column vector q,k∈R^d表示，H為正規化Hadamard：

~~~text
原本：
q,k ─────→ sign → identity popcount ──× cI ─┐
  └──────→ H → sign → H popcount ─────× cH ─┴→ score

提案：離線固定 D=diag(±1)，Q/K 同用 D
q,k → D ─→ sign → identity popcount ──× cI' ─┐
       └─→ H → sign → HD popcount ────× cH' ─┴→ score

D／cI'／cH' 不依每張圖片改變；不增加第三個basis。
~~~

浮點參考 S_FP=qᵀk/√d。因DᵀD=I，(Dq)ᵀ(Dk)=qᵀk，**共同變換前後的FP dot product精確相同**。但一般有sign(H D q)ᵀsign(H D k)≠sign(Hq)ᵀsign(Hk)，因此二值近似有改變的空間；不保證一定改善。

非零分量下，sign(Dq)=D sign(q)，所以identity分支的sign dot不變。然而本地sign以`x>=0`回傳+1，**遇到0就不能使用這個恆等式**。尤其INT8會產生更多0；新方案必須測zero-rate、sign(0)、clipping及完整bit-true score，不能宣稱identity分支必然無損。

另有兩種不值得當新basis的變換：對sign之後的Q/K做共同channel permutation，其dot product不變；某些D會讓H D僅成為H的row permutation，此時Hadamard分支也不變。候選需排除這類代數重複及全域正負等價（有0時另查tie規則），不能把重複的8個模式當成8種有效方向。

### 真正待研究的部分：選擇目標

不是以COCO佔優的大量背景token直接平均選D，而是分別計算Detect、Pose的teacher ranking reversal rate e_t(D)，每個任務先per-image平均：

~~~text
D* = argmin_D max_t { e_t(D) / max(e_t(I), ε) }
限制：每任務 e_t(D) ≤ e_t(I) + δ；basis數／位寬／PoT槽位保持原契約。
~~~

ε用來處理基準錯誤為0；δ首案設0，若無可行非identity候選就保留I，不事後放寬。每個D按同一train-only程序重新估PoT係數，不能只給新basis更充分的校準。teacher margin極小的pair不應當可靠次序；tie與margin門檻由train資料事前固定。

這是待驗證的任務保護目標，不是論文已證明的演算法。必須勝過「同一硬體表示的隨機D」，有訊號後再比「同一候選池的平均ranking-error選D」，才能把收益歸因於任務保護而非旋轉本身。

### 硬體能不能省下額外動作？

若Q/K最後一段為可重建的affine projection，可令Wq'=D Wq、bq'=D bq，K同理，將D吸收到離線權重；若最後是Conv+BN，可在eval fold後做此變換。只允許在Q/K最終channel域操作，不能跨不交換的activation／位置變換亂搬。

這只證明有條件的代數可吸收性，不代表現有部署契約允許重寫Q/K、不代表INT8的-128取負可無飽和、更不是延遲已實測不變。若必須保留原投影常數，或exporter仍產生額外op，本方向先暫停；不偷偷加dense rotation或per-image routing。

## 4. R2-REGION：固定token-pair預算的區域排序保護

### 為什麼不是再多加一個全域KD？

設一層有N個tokens，一個小物件只對應m個query。均勻平均N²個attention關係時，該物件query列只佔m/N；若只看其內部關係則約(m/N)²。這是占比推導，**不是本案已測得的梯度占比**，但足以提出背景稀釋假說。

~~~text
既有GT boxes／有效keypoints ─→ 對齊實際token位置 ─→ 固定pair預算分配
                                                         │
同一FP teacher → score排序 ─┐                            │
                            ├→ pairwise ranking loss ←───┘
Binary student → score排序 ─┘
推論：刪去teacher／mask／ranking loss，原圖與固定scale維持不變
~~~

每個task、每個site使用相同固定pair上限B；先按影像／物件分配，再按GT box、Pose有效keypoint支持區域、背景三類分配。初始只設一個固定配額方案，不掃多個比例；空類別的餘額依預先規則重分配。小物件box沒有覆蓋token中心時，以連續cell覆蓋率／最近有效cell建立query候選，不能讓它直接消失。

這不建立新影像split、不更改labels；支持區域只是loss內從既有GT及同一augmentation座標產生的temporary mask。COCO沒有本案Pose keypoints就不補造；visibility只用本地既有有效點語意，不把ball/bat當成人體骨架。

對同一query i和兩個keys a,b，teacher決定y=sign(S_T[i,a]-S_T[i,b])；以

~~~text
L_rank = mean[ w · softplus(-y · (S_B[i,a]-S_B[i,b]) / τ) ]
~~~

訓練student，teacher stop-gradient。只採teacher差異大於train-only鎖定門檻的pair；w有上限且per-task／per-image正規化。不要強迫二值score重建不可表示的大幅FP margin，也不把attention權重當成因果解釋。τ、pair數、aux梯度比例要與均勻／前景對照按同一規則校準。

### 創新需要回答的問題

均勻KD vs 區域KD的差值只能支持「非均勻選擇有用」；還必須用相同B、teacher、loss及更新次數，比較普通foreground/background KD與本案object/keypoint-aware配置。只有後者額外保護ball／bat Pose且不傷COCO，才有超越現成foreground KD的本地證據。

固定B只能控制KD loss處理的pairs，**不等於teacher forward免費**；若原模型仍materialize完整N² score，pair取樣不會省掉那部分記憶體。若student Q/K沒有梯度／STE失效，換mask也補不回來；先完成第一輪的scope查核。

## 5. 為何暫不選其他看似新的方案

- 加P2／大attention／整套新Neck：本輪沒有error-slice支持，硬體與因果成本較大，不因方法多而優先。
- 普通CrossKD或LD：可作適配基準，但不是本案創新；LD的DFL bins還需額外訓練頭。先不增加第二套teacher/head。
- 改task weight、one-to-one loss或共享BN：先查現有實作；本地已有task加權／progressive雙分支，shared BN統計亦已固定。不能把已有功能再命名。
- 只改scale數量／正scale幅度：可改善幅度，但單一正倍率無法改同一列內原有ordering；第一輪scale分支已涵蓋，不重複當新貢獻。

## 6. 結論與限制

推薦把「任務保護的固定basis」列研究主題首選；但現行硬體凍結契約未改之前，實際較適合先評估「固定預算的區域排序蒸餾」，且先確認哪些student參數真的可更新。完整對照見[第二輪計畫](<../../proposals/round2-innovation/plan.md>)。第一輪不變，第二輪沒有必須跑滿的矩陣。

目前完成的是文獻邊界、本地文字查核和推導。沒有Q/K cache replay、梯度probe、CPU模型驗證、AP、latency或硬體綜合結果；不宣稱恢復幅度、不宣稱全球首次。官方線上YOLO26文件與本地固定8.4.90可能不同，正式本地evaluator／分支契約不可隨線上預設改動。
