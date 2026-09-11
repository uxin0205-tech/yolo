# 方向1完整整合：先選BEST，再做訓練、MASF／HOG／BinaryQK優化

更新：2026-09-09。實際完成情況以[第一輪需求稽核](<round1-audit.md>)、[逐 epoch 結果](<results/README.md>)與[機器可讀狀態](<direction1-plan.json>)為準。六組短訓及必要診斷已完成或依 gate 停止，目前沒有通過驗收的新精度候選，仍選 J3 BEST；不將完整 queue 退出當成增準成功。

下文保存原計畫的決策與方法門檻，不代表所有條件式候選都已排程。原 BEST、資料、checkpoint 保留；第二輪與 person-only 不啟用。MuSGD 更新校準未過分組門檻，不開長訓；MASF bridge 未過，不直接 relocation；QK STE／KD 尚無合法新 challenger／teacher，不能繞過原 guard。

目前無 active GPU job。新增 GPU 工作仍採每次最多 600 秒 blocking wait，正常不读 log 或額外取樣；使用者取消舊配額 cutoff，採單一主代理。原量化專案明確延後，恢復需重新確認。

「變動較大」指模型／參數／訓練方式，不指 batch 或 BatchNorm。超參數見[optimizer policy](<optimizer-policy.md>)，詳細歷史見[工作紀錄](<../../docs/worklogs/README.md>)。

## 1. 第一件事：先選哪個BEST

接續來源固定為 `/home/uxin/yolo/yolo_combine/final/full35/`（下稱F）；用F內code/project與source_bundle，不原地改final。下表全是**既有seed0 Bit-True結果**，不是本次新validation。

| 候選 | epoch | joint score | person AP | ball box AP | ball pose AP | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| J3 best_joint | 58 | 0.711175 | 0.620381 | 0.507437 | 0.859909 | 原始分數參照B0；現行joint最高候選 |
| J3 best_pose | 59 | 0.711142 | 0.620513 | 0.508312 | 0.859571 | 首輪比較候選；不是每項Pose都最高 |
| J2 best_joint | 35 | 0.710038 | 0.618511 | 0.510866 | 0.860799 | 回退候選；若關注ball錯誤才優先追加重驗 |
| J3 last | 63 | 0.709974 | 0.620398 | 0.507738 | 0.861180 | stage終點；特定ball-pose取捨才追加 |
| Shared best_detect | 0 | 0.580499 | 0.626057 | 0.281089 | 0.716679 | Pose未適應、八項舊gate未過，不直接作task=both起點 |

來源：[候選CSV](<../../../yolo_combine/final/full35/analysis/data/coco-person-candidate-comparison.csv>)、[逐epoch八項CSV](<../../../yolo_combine/final/full35/analysis/data/validation-metrics.csv>)、[原分析](<../../../yolo_combine/final/full35/analysis/FINAL_ANALYSIS.md>)。J3 best_joint與J2 best_joint有獨立final revalidation；best_pose/last的候選數字來自完整逐epochvalidation，來源等級分開記錄。

**best不是「所有指標最強」的意思。** 原程式的best_detect看COCO overall/person平均，best_pose看BBAT box/pose平均；best_joint在八項gate通過後看下述加權總分。因此best_pose的單一keypoint AP可以低於best_joint。

挑選方式：

1. 先沿用舊完整結果篩選，不把所有候選都重新訓練。J3 best_joint 與 J3 best_pose 已完成完整同指令重驗；J2／last 依實際弱項再追加。
2. 固定同一F程式版本、資料、EMA/live、backend、imgsz、head分支、evaluator與原始預測匯出規則，核對兩task都能正常載入。不能用partial load後遺漏的新head當已訓練權重。
3. 同一組使用者案例並排看漏檢、誤檢、框與點位。沒有GT時只報人工對照，不造AP或抖動量測。資料集不抽樣、不新切split；案例是開發用途，不冒稱獨立test。
4. 保留原joint公式；差距小於0.001視作本輪工程上的接近區間，不代表統計等價。若沒有清楚細項／視覺收益，就保留現行best_joint；若選較低joint但應用表現較好的候選，要明記「應用取捨」，不能說它joint最高。
5. 結果已存為 `PSEL=J3 best_joint` 的 inference EMA；**可靠的比較基準可以不完美**，但本輪 E1 不能以 run-local best 取代。只有確認有必須先修的基礎問題才補訓，不要求先把所有部分 train 到理想精度。

推論／評分用對應 `weights/.../inference/*.pt`；訓練狀態來源用對應`full-resume/*.pt`，不是只因同名就混用EMA與live。新方法／新optimizer另外開run，原權重永遠保留。

## 2. 比較分數與parent不能混在一起

```text
B0    = 原始 J3 best_joint（固定歷史比較錨點，不覆寫）
PSEL  = BEST比較後選出的接續來源（J3 best_joint inference EMA）
PBASE = PSEL，或必要原生loss修復後的可比較基準
P     = 每一對新實驗共同起點（可為前一個已驗收winner）
C/T   = 同P、同預算的control／單一改動candidate
```

原分數保留，定義見[metrics.py](<../../../yolo_combine/final/full35/code/project/src/yolo_combine/metrics.py>)：

```text
joint = 0.2 × COCO overall box AP
      + 0.2 × COCO person box AP
      + 0.2 × BBAT box AP
      + 0.4 × BBAT pose AP
```

報告分開列：PSEL−B0（選checkpoint的差別）、PBASE−PSEL（原生修復收益）、T−C（方法比較）、T−B0（最終總收益）。不把checkpoint選擇、額外訓練、optimizer和新模組的收益全部算成同一種創新。依序winner的單項差值不能直接相加當最終整套收益。

## 3. 整個流程

```text
final 的 BEST 候選 → 同口徑評分／視覺比較 → 選 PSEL
                                               │
                        ┌──────────────────────┴─────────────────────┐
                        │ 無必須先修的問題                            │ 有明確問題
                        │                                            ↓
                        │                               原生loss、小範圍AdamW修復
                        └──────────────────────┬─────────────────────┘
                                               ↓
                                       固定比較起點 PBASE
                                               ↓
                         一次一個方向：短程AdamW matched對照
                  HOG / 梯度投影 / RepConv / MASF位置 / 合法QK challenger
                                               ↓
                                 方法有收益才保留，無收益就回退
                                               ↓
                   若確有長訓需求：固定架構／scope，在同handoff點分兩臂
                               ┌───────────────┴───────────────┐
                               ↓                               ↓
                         O-A：新AdamW                    O-M：新MuSGD
                             同資料、同剩餘epochs、同state-reset邊界
                               └───────────────┬───────────────┘
                                               ↓
                        勝出的recipe → 下一個必要方向（仍獨立比較）
                                               ↓
                        最終組合驗證 → 去aux／fuse → 部署複驗
```

若第一個必要工作就是原生長訓，O比較可以提前到PBASE之後；但不得與首次加HOG、搬MASF、解凍Q/K同時發生。MuSGD不是每個5epoch小實驗都要用，也不是長訓必定勝AdamW。

### 3.1 本次第一輪的實際執行順序

```text
J3 best_joint／J3 best_pose
              ↓ 同口徑重驗（已選 PSEL = J3 best_joint）
Detect batch：32×4、64×2、128×1；Pose：16
              ↓ 只保留兩個 arm 都能 fit 的最快 physical batch
同一 PSEL、同一 EMA、fresh optimizer、同一資料 trace
              ├─ W-CTRL5：原生 loss，固定 5 epochs
              └─ W-HOG10：raw pre-MASF P3 輔助 HOG，最多 10 epochs、patience 4
```

`W-CTRL5` 和 `W-HOG10` 前 5 個 epoch 使用同一個以 10 epochs 為 horizon 的 scheduler trace；每個 epoch 都保存 checkpoint。若 HOG 臂在第 5 個 epoch 前因 patience 提早停止，只比較兩臂共同完成的 epoch，不能把不同預算的結果宣稱為純 HOG 因果。batch 測速只用來選 physical batch，測速更新全部丟棄，正式兩臂沿用同一個選定 batch。

首輪 HOG 的可訓練範圍先固定為 Neck、Detect head、Pose head，以及候選專用的 raw P3 HOG head；backbone、attention、既有 MASF 凍結。兩臂都用 AdamW；新訓練 warmup 固定為 1 epoch。HOG 的 μ 由 training-only gradient calibration 估計，使輔助梯度約占主梯度 5%，不是把 loss 權重直接寫成 `λ=0.05`。

## 4. 方向1所有候選與必要實驗

以下是候選清單，不是全部必跑queue。預設先做BEST比較，再按錯誤證據選一組；診斷明確指向QK時可以先做Q分支，不強制先HOG。

| 類別／ID | 原本 → 候選 | 最小訓練量／前置 | 何時停止／追加 |
| --- | --- | --- | --- |
| 基準修復 B | 原圖／原loss → 只調必要head或Neck+heads | 條件式5或10ep；不是新方法消融 | 沒有必須修的問題就跳過；不自動重建J0 |
| HOG W | 原loss → raw pre-MASF P3 training-only HOG9 | W-CTRL5固定5ep；W-HOG10最多10ep、patience4；前5ep共用10ep horizon scheduler | 無收益停止；有效且要主張HOG特有收益才補W-LUMA10 |
| 衝突投影 G | task gradients相加 → 只修Pose對Detect的負向shared分量 | G-CTRL10/G-PROJ10各10ep，先量到同stage負cosine≥20%且correction median≥0.02 | 舊logs不是新parent根因；不與HOG首輪疊加，不冒稱只保護person |
| RepConv R | layer17 stride2 Conv → 可融合RepConv | R-CTRL5/R-175各5ep，初始等價 | layer17有效才考慮layer20；不全面替換，不聲稱fused MAC自動降低 |
| MASF M | shared P3 seam → Detect P3-only分支 | 沒有合法no-MASF parent時：bridge兩臂各5ep，再位置兩臂各5ep | bridge失敗停止；不能把已訓練shared MASF直接搬過去 |
| BinaryQK Q | 現有固定PoT binary → 一個site／梯度策略候選 | 先查舊消融、live gradient、硬體契約；合法後Q-CTRL10/Q-CAND10各10ep | Float/Bit-True不是純FP/binary對照；不繞過qk_ste guard；FP site須確認硬體可接受 |
| scale Q-SCALE | 現有16個固定slots → B4固定channel groups | 僅剩magnitude誤差且kernel有利時，先replay與parity | A8每token selector仍有每圖成本，只作上界；不先堆8個動態scale |
| optimizer O | AdamW後繼續AdamW → 同點換MuSGD | O-A20/O-M20各20ep；共同前綴只跑一次 | 無長訓需求就跳過；無增益不強制切換 |
| 部署效率 E | fixed mode仍做無用dynamic reduction → fixed early-return | 重新確認final實際路徑存在冗餘後才做，必須輸出等價 | 這是效率修正，不是補精度；目前未改程式 |

短程scope／LR和O切換契約以[optimizer-policy.md](<optimizer-policy.md>)為準；各方向舊機理報告仍可從[方向索引](<../README.md>)追溯。MuSGD比較是其完整optimizer recipe（含對應LR與decay語意），不是只有換名稱的純單參數實驗。

### 第一輪的兩個主要圖

現有圖，先保留為對照：

```text
layer16：raw P3 → shared MASF → p3_shared ─┬─> layer17 → P4 → layer20 → P5
                                         ├─> Detect([p3_shared,P4,P5])
                                         └─> Pose  ([p3_shared,P4,P5])
```

HOG訓練期分支與條件式Detect-only MASF圖：

```text
layer16：p3_raw ────────┬─> layer17 Conv* → P4 → layer20 Conv → P5
                       │
                       ├─> MASF → p3_det ─┐
layer19：p4_raw ──────────────────────────┤
layer22：p5_raw ──────────────────────────┤
                                         ↓
                          Detect([p3_det,p4_raw,p5_raw])

Pose([p3_raw,p4_raw,p5_raw])：不經Detect-only MASF
p3_raw → HOG head → HOG loss：只在HOG訓練候選啟用，部署移除
Conv*：只有RepConv單點候選通過才替換，部署再fuse回3×3
```

這是分別驗收後才可能組合的目標，不是第一個run同時裝入所有方法。HOG首測保留現有shared MASF，從raw P3拉輔助分支；不需要先搬MASF才能測HOG。MASF新殘差effective gate=0初始化，Pose/P4/P5隔離需驗證；若尚無no-MASF共同parent，bridge成本必須如表列四個runs。

## 5. 變動大時為何先考慮AdamW

新head、模組換位或shared features重新適應可能帶來不同梯度尺度；AdamW的逐參數適應性是先作穩定對照的理由，加上現有正式run已有AdamW來源。但**大改動不代表AdamW必勝**，真正控制破壞程度的是初始化是否等價、解凍範圍、LR、warmup與更新量。

本案大改動優先「局部scope＋保守LR＋AdamW適應」，不要同時全模型解凍。長訓再把MuSGD作單獨候選；其實作本身就是Muon與SGD混合，不是先跑Muon再跑SGD。官方與本地證據、state/weight-decay/LR差異詳見[優化器研究](<../../docs/research/2026-09-08-adamw-musgd-stage-policy.md>)及[超參數規格](<optimizer-policy.md>)。目前沒有本模型AdamW→MuSGD勝出的實測結果。

## 6. 驗證与停止條件

1. 所有成對比較同起點、資料順序、augmentation、有效batch、可訓練scope、更新次數、AMP/EMA、criterion progressive狀態及固定QK係數；不能換optimizer時一起重設loss schedule或解凍更多層。
2. 保留完整COCO val5000與BBAT5 val683；沿原evaluator主表、Bit-True選模、Float另表。視覺conf與AP匯出conf分開；不換成person-only。
3. 原八項gate照原schema報告，不修改歷史0.08門檻。另設新優化工程門檻：T相對C及該對parent每項不退超過0.001；joint或overall至少+0.001，或預登錄主要視覺錯誤有可信改善。整體相對B0的正負差值全部展示，不用新選parent掩蓋退化。
4. 5/10/20ep分別在0/5、0/5/10、0/5/10/15/20完整驗證。固定末輪是等steps主比較，best_joint是另列部署候選；兩臂都用同一選模規則與驗證次數。候選提早安全停止只標失敗，不冒稱等預算勝出。
5. NaN/Inf、資料／硬體契約變動立即停止；中期保護指標相對parent退超過0.005暫停調查，不自動加epochs。對新零初始化參數另看絕對更新，不能用除以零的relative-update值判爆炸。
6. seed0有效才補paired1/2，報每seed差值與變異。若僅單seed、小差距或人工案例，只標provisional，不聲稱顯著或普遍優越。
7. 最後逐步移除HOG／teacher、RepConv fuse、既定export／量化，重新看完整指標與固定案例。固定QK係數不在一般校準中暗改，變更另立硬體版本。真實latency/energy須目標設備實測。

## 7. 第二輪與還能優化什麼

- R2-REGION：固定token-pair預算的區域排序蒸餾；要先有合格teacher及student live gradient。Detect＋Pose融合不自動讓KD失效，也不自動保證有效。
- R2-BASIS：任務保護的固定二值基底；要另立硬體契約，和隨機／普通固定基底對照，不能只換名字說創新。
- 首輪優先排除可實測的更新過大／任務衝突、錯誤的BEST選擇與部署前後差異。一般AdamW→MuSGD、HOG或局部RepConv本身不作全球首次宣稱。

這些列[第二輪研究](<../round2-innovation/README.md>)，不與第一輪同時全部啟用。person-only繼續暫緩。

## 8. 執行結果與歷史來源

[第一輪需求稽核](<round1-audit.md>)逐項區分完成、候選停止、未建立的契約與外部延期。歷史 fresh-age native、EMA-age 修正、native5、HOG、RepConv、MASF、heads-only、BN／MuSGD／J2 的完整過程見[中文工作紀錄](<../../docs/worklogs/README.md>)，不再把舊「尚未開始」當成現況。

來源與資料保持唯讀：COCO80 `/home/uxin/yolo/coco2017.yaml`，BBAT5 Pose `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml`，registry `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。不重切、不抽樣、不改 labels；未刪除來源／checkpoint，未 commit／push。
