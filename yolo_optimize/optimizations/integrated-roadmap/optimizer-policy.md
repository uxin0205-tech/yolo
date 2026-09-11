# 方向1：AdamW適應、MuSGD長訓候選與超參數

最新執行結果（2026-09-09）：Native／HOG／heads-only 短訓皆未接受新 recipe；heads scope 的 MuSGD train-only 16＋16＋16 校準已執行，Detect no_decay 更新比0.268低於原安全帶，因此停止此 recipe，不開20epoch長訓。正式handoff／resume與AP比較未執行，仍是以下的條件式規格。詳見[結果紀錄](<../../docs/worklogs/2026-09-09-bn-musgd-j2-results.md>)與[完整稽核](<round1-audit.md>)。下方2026-09-08段落是規格提出時的歷史狀態，不是現有queue。

更新：2026-09-08。配合 [方向1主計畫](<direction1-master-plan.md>)。Native5 在 E5 觸發 EMA Ball Box 安全線；LR×0.25（Neck 2.5e-6、heads 6.25e-6）在 E2 觸發 live safety。縮小 LR 只部分減輕退化，BN-only 也未通過，因此沒有已接受的新 optimizer recipe，不再追加猜測 LR。兩次保存與退出均正常，原 BEST 不替換。MuSGD／HOG 尚未正式啟用，完整結果見[對照報告](<native5-results.md>)。

## 1. 先選BEST，optimizer不是第一步

B0是固定J3 best_joint歷史錨點；PSEL需先比較BEST再定。使用者說的是模型／參數「變動較大」，不是batch size或BatchNorm。可靠的基準可以有精度缺口，不要求先從零練到完美；有基本設定／載入問題才先修。

保守預設：新處理先用已有來源的 AdamW、縮小 scope 並使用 1 epoch warmup；若有長訓需求才做 AdamW 對 MuSGD 的獨立配對。不能因 loss 瞬間下降較快就直接升格 MuSGD。所有新訓練（包含未來若觸發的 MuSGD）均使用 1 epoch warmup；歷史 J3 的 3 epochs 僅作 measured reference，不回寫成新設定。

## 2. 目前程式到底支援什麼

- final的`_joint_config_impl.py`允许AdamW/MuSGD；`stage_policy.py::build_joint_optimizer`兩種都有建構分支。**MuSGD不是僅placeholder，也不是被AdamW guard禁止。**
- 原trainer啟動只建立一次optimizer；stage切換更新LR／scope與scheduler，不換class。`challenger_optimizer`欄位不會自動觸發切換。因此AdamW→MuSGD需要受控handoff或新run介面，現在未實作。
- 本地實際套件是 `/home/uxin/yolo/yolo_combine/.venv/lib/python3.12/site-packages/ultralytics/`，METADATA=8.4.90。不要把另一個venv、最新main與final的依賴混用。
- 現有formal builder的MuSGD設定：`momentum=beta1=0.948`、`nesterov=true`、`muon=0.2`、`sgd=1.0`；`use_muon=decay`。這些混合權重不需加總為1，不是20%／80%。
- decay group是非bias且`ndim>1`的參數，走Muon＋SGD；no_decay（bias、BN等）走SGD。Q/K等hardware-frozen參數不納入optimizer；更換optimizer不會解開硬體凍結。
- MuSGD類別本身預設lr1e-3、momentum0、nesterovfalse、muon/sgd各0.5；**不等於formal builder的有效設定**。沒有`adjust_lr`API。
- 官方標準trainer的`optimizer=auto`是依iterations選擇，不是看到架構變動就切換；本地joint parser只接受明確AdamW/MuSGD，不能把標準trainer的auto門檻、head LR倍數或batch rescaling默默套入。

本地來源：[stage_policy.py](<../../../yolo_combine/final/full35/code/project/src/yolo_combine/stage_policy.py>)、[config parser](<../../../yolo_combine/final/full35/code/project/src/yolo_combine/_joint_config_impl.py>)。一手外部來源與版本限制見[研究紀錄](<../../docs/research/2026-09-08-adamw-musgd-stage-policy.md>)。

## 3. 為何不能把兩者LR直接抄成相同數字

AdamW對每個參數的更新包含一階／二階矩估計及decoupled weight decay：

```text
ΔW_AdamW ≈ -η × m_hat / (sqrt(v_hat)+eps) - η×λ×W
```

MuSGD對矩陣的實作是先做orthogonalized Muon更新，再做SGD momentum更新；概念式為：

```text
ΔW_MuSGD ≈ -η × [0.2 × U_Muon + 1.0 × U_SGD]
```

實際有順序：SGD decay使用前一個Muon步後的參數，不能把上式當完全等價實作。no_decay group只有SGD路徑。其方向、幅度與state都和AdamW不同；相同LR或相同weight_decay數字不代表同樣更新與正則化。

因此O比較是「optimizer＋其適配LR／state規則」的recipe比較，不能把結果歸因為純正交化。AdamW較適合大改動只是本案保守起點的工程推論，不是必然定理；小LR、等價初始化和限定解凍範圍仍是主要安全措施。

## 4. 共通訓練設定

| 欄位 | 設定／提案 |
| --- | --- |
| 資料／輸入 | 原COCO80＋canonical BBAT5 Pose；imgsz640；不改split/labels |
| Detect有效batch | logical128；先測 physical32×4、64×2、128×1，兩個 arm 都能 fit 後才選最快 physical batch；每 macro 兩個 logical Detect batches |
| Pose batch | 16；每macro一個Pose batch |
| loss normalization | reference64；Detect/Pose權重1.0/0.25；保留原macro normalization |
| AdamW | betas=(0.948,0.999)，eps=1e-8，weight_decay=2.7e-4，沿role×decay分組 |
| MuSGD候選 | momentum0.948、nesterovtrue、muon0.2、sgd1.0；decay group λ2.7e-4，no_decay λ0；beta2不適用 |
| gradient clip | AMP unscale後global norm10；每macro只做一次optimizer更新 |
| AMP／EMA | 沿固定parent策略；兩臂相同，更新計數與起始EMA一起記錄 |
| BatchNorm | shared running statistics固定；head沿原train策略。與使用者所說「變動較大」無關 |
| augmentation | mosaic0；Detect fliplr0.5／Pose0；其他沿resolved parent配置，不新增mixup/解析度grid |
| scheduler | cosine，所有新訓練 warmup 1 epoch、起始因子0.1；AdamW／小實驗末因子0.5；O 長訓兩臂末因子0.1為新提案 |
| seed | 先0，有效才paired1/2；各臂同資料亂數，新增module初始化獨立RNG |
| selection | 原Bit-True joint公式＋八項gate；Float獨立列；固定末輪是等steps主比較，best_joint另列 |

歷史J3 warmup=3，J0/J1/J2=1；原generic YAML的1不代表最後J3。MuSGD名義λ先沿用是控制自由度，並非與AdamW等正則化；要記錄decay update比重，出現主導更新就停止調查，不依val私下改λ。

## 5. 各方向的首版超參數

所有未列為可訓練的參數凍結。每組由同一parent出發；基準修復不是所有人都必跑。

| 分支 | epochs | optimizer | 可訓練範圍／峰值LR | warmup |
| --- | ---: | --- | --- | ---: |
| B-HEAD5 | 5 | AdamW | 證據指向的task head 5e-5 | 1 |
| B-NATIVE5 | 5 | AdamW | Neck1e-5；Detect/Pose heads各2.5e-5 | 1 |
| B-LATE10（條件式） | 10 | AdamW | 另開 layer9+ backbone1e-6；Neck1e-5；heads2.5e-5 | 1 |
| W-CTRL5 | 固定5 | AdamW | Neck1e-5；Detect/Pose heads各2.5e-5 | 1 |
| W-HOG10 | 最多10、patience4 | AdamW | Neck1e-5；Detect/Pose heads各2.5e-5；candidate HOG head3e-4 | 1 |
| G-CTRL10/G-PROJ10 | 各10 | AdamW | Neck1e-5；heads2.5e-5；只投影active shared部分 | 1 |
| R-CTRL5/R-175 | 各5 | AdamW | Neck含新Rep branch1e-5；heads2.5e-5 | 1 |
| BR-KEEP5/BR-OFF5 | 各5 | AdamW | Neck1e-5；heads2.5e-5；舊MASF參數凍結 | 1 |
| M-CTRL5/M-DET5 | 各5 | AdamW | Detect P3 predictors2.5e-5；candidate MASF context5e-5、gate1e-4 | 1 |
| Q-CTRL10/Q-CAND10 | 各10 | AdamW | Neck1e-5；heads2.5e-5；合法解凍時 Q/K5e-7 | 1 |
| O-A20 | 20 | 新 AdamW | 沿共同 handoff 的相同 scope；若 Neck/heads 則 1e-5/2.5e-5 | 1 |
| O-M20 | 20 | 新 MuSGD | 同 scope；Neck 探測 1e-3／heads 探測 2.5e-3，**正式 LR 待下面的 train-only 校準** | 1 |

Q候選仍需獨立STE／硬體契約，現在不可以把正式baseline qk_ste改true就跑。小M分支沒有自動解凍整個Neck；如果O從M的checkpoint做handoff，也必須沿原P3/MASF scope，不突然改為全Neck/heads。其他role的MuSGD LR同樣待校準，不憑上述範例外推。

HOG：raw pre-MASF P3→1×1 Conv(C3→9)，cell8、9 unsigned bins、同一增強後 RGB 產生 box-masked target，無能量 cell 忽略。μ不是固定 0.05；目標是加權 HOG 梯度約主梯度 5%（觀察帶 2%–10%），用 training trace 估計 μ0。epoch1 μ0 未啟用／做量測，epoch2 ramp，epoch3–8 hold，epoch9–10 off。首輪 `W-CTRL5` 固定訓練 5 epochs，`W-HOG10` 最多 10 epochs、patience 4；兩臂前 5 epochs 共用同一個 10 epochs horizon 的 LR trace，每 epoch 保存 checkpoint。若 HOG 提早於第 5 epoch 停止，只比較兩臂共同完成的 epoch，不把不同預算當成純 HOG 因果。已關閉的 aux 在 O 長訓兩臂都維持 off，不能只在 MuSGD 臂偷偷重開。

MASF搬位之前如果沒有合法no-MASF來源，必須先做兩臂bridge；新位置effective residual gate初始0。RepConv17是stride2，不加identity branch，新增1×1 branch BN gamma/beta=0並驗證等價。這些初始化處理不因optimizer改變而省略。

## 6. MuSGD LR：探測值不等於正式設定

Neck1e-3／heads2.5e-3僅為**新候選探測初值**，不是已知適合這份checkpoint，也不是按batch線性放大。正式O-M20還不能啟動，直到完成下列僅用training的檢查：

1. 從共同handoff取原有training loader接續的固定16個macro作短probe，不另建dataset subset，不用validation調LR。control也重播相同trace。probe後恢復完整weights、BN buffers、EMA、RNG、loader、criterion和optimizer起點，probe算力另記。
2. 對有非零且有限梯度的既有參數，量測`u=RMS(ΔW)/max(RMS(W),eps)`；新零初始化gate／BN branch不套此relative式，另看絕對更新和輸出差分。
3. 以同role AdamW的更新量中位數作參考：`η_M,r = η_probe,r × median(u_A,r)/median(u_M,r)`。這是幅度對齊的近似規則，不保證多步trajectory等價；只做一次修正，再一個相同長度確認窗口，不做val-driven sweep。
4. 校準role LR時分別檢查其decay/no_decay群組，不能讓大量矩陣參數掩蓋BN／bias的過小或過大更新。目標既有active群組的中位數比在0.5–2.0，無梯度／非有限／嚴重偏離時停下來報告，不無限放大LR。安全帶是新工程提案，不是文獻最佳值。
5. 保存role→baseLR map；scheduler與stage更新不得把它覆寫回AdamW的role LR。現有update_optimizer_stage會從stage覆寫LR，故需驗證adapter／manifest與resume一致。現在未實作此保護。

若MuSGD一個role LR無法同時兼顧matrix與scalar updates，先判這份簡單recipe不合適，保持AdamW；不默默引入group-specific LR、layer-wise optimizer或新MuAdamW混合來救同一arm。

## 7. AdamW→MuSGD的公平切換

固定一個完成AdamW適應／短程候選的handoff checkpoint。首版優先用固定末輪，避免事後挑切換epoch。從它建立：

```text
共同AdamW前綴（只跑一次）
       │ 同一live weights／EMA／BN／資料及criterion狀態
       ├─ O-A20：fresh AdamW + 新20ep scheduler
       └─ O-M20：fresh MuSGD + 新20ep scheduler
```

- 兩臂都重置 optimizer state 並重新 warmup 1 epoch，控制「state reset」這個差異；不能一臂保留 AdamW moments，另一臂全部歸零，卻說只有 optimizer 演算法不同。
- 不將AdamW的一／二階矩硬轉成MuSGD兩套momentum buffers。MuSGD buffers按其實作初始化；比的是受控新階段，不稱exact resume。
- live、EMA、BN及loader/RNG由同一handoff保留；criterion progressive／E2E schedule不因optimizer reset自動回到epoch0，兩臂須同有效schedule。新run的日誌epoch與原criterion有效epoch分開記錄。
- scheduler總長與warmup/末因子相同，baseLR是各optimizer預先固定的recipe；不在只有MuSGD的臂解凍更多層、改loss或增加epochs。
- 第一版後段各20ep，沒有收益就不追加40/80/300ep。若有明確持續改善而要更長，另登錄新的共同長度，AdamW控制也要同樣延長。
- 最終若O-M勝O-A且沒有視覺／八項保護指標問題，才能把「AdamW適應→MuSGD長訓」變成後續可沿用recipe。沒有實測前預設仍AdamW。
- 這對比較只回答「固定AdamW前綴後，切換是否比同邊界重啟AdamW更好」，不證明AdamW前綴本身必要，也不回答MuSGD從PSEL開始是否更好。若另研究這一點，才新增同總steps的MuSGD-only對照；不先把三臂變成必跑。
- 20ep是較長訓練候選的第一個比較窗口，不是宣稱已充分長訓；標準auto的iteration規則不能直接換算成這份joint macro的epoch數。

## 8. 執行前測試與目前狀態

必須測：trainable group無重複／遺漏、hardware-frozen hash不變、handoff live/EMA等價、step0 loss相符、AdamW/MuSGD state獨立且serializable、AMP overflow replay、criterion／loader／scheduler resume一致、兩task gradient與role LR確實更新、指標完整且不混evaluator。

以上需模型／梯度的測試本次全部未執行；目前只完成文件同步與執行規格整理，未改 final 或依賴，沒有新訓練結果。GPU 授權只代表可以進入下一個執行步驟，不代表 `training_ready`、BEST 選擇或任何實驗已完成。
