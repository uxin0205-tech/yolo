# YOLO Optimize 全階段研究總報告

> 更新日期：2026-09-12。這是全階段唯一詳細總報告，保留已完成實驗的原始數據與限制。本次為實體資料夾重整及報告補全，不是新一輪訓練。第 16–18 節補充最終決策、遷移與復現邊界。

統整基準日：2026-09-11（Asia/Taipei；封存／後續完成時間以 UTC manifest 為準）。本報告是截至本輪的權威導航；歷史文件的「進行中／尚未開始」只代表當時狀態，不覆蓋本報告。所有實驗原始檔保留，不刪除失敗紀錄、不覆寫權重。

## 1. 摘要與目前決策

研究目標是在硬體可實作的 BinaryQK 、 PWL Softmax 、低成本 activation 下，盡量保住 COCO80／person 與 BBAT5 ball／bat 的框及關鍵點精度。研究不是只追求單一 joint 平均分，也不是只做 person-only 。

截至統整前，融合前方向 1 、 P2 最後對照、 P3 bridge 重新融合、 Pose 恢復、 activation 、雙教師 KD 、 Pose-head KD 、關鍵點分支重組均已有實測。工程上有明確收穫：發現 BinaryQK 正式布林前向切斷 Q/K 梯度；建立精確前向的訓練 surrogate；隔離 MASF 對 P4/P5 的影響；修正 one2one 對 MASF 的直接監督路徑；驗證固定共享 state 可以精確保住 Detect 。

但不能聲稱所有精度問題已解決。融合前 COCO 從 B100 的 0.503589 恢復到約 0.5082，仍低於原 FP 0.518019；HOG 、 RepConv 、 MASF 的額外收益未通過既定方法門檻。重新融合的 COCO 優於舊 combine，但 BBAT 尤其 bat 尚有缺口。 qSiLU 是目前新主線使用的研究起點，不是全面勝過歷史 combine 。兩種 KD 都沒有產生新的合格 best_joint；分支重組也沒有保住 KD 的 Pose 收益。

目前保留 qSiLU E2 作為新主線預設，不覆寫舊 combine 的 best_joint，也不刪除 P3 bridge 與無 MASF 對照。下一個最低風險方向是同權重的 one2one／one2many 推論比較，先不新增訓練。後续結果另以附錄保存，不改寫本次統整前的數字。

## 2. 資料、指標與比較規則

| 項目 | 固定契約 |
| --- | --- |
| COCO80 | `/home/uxin/yolo/coco2017.yaml`；train 118,287／val 5,000 |
| person | COCO80 的 class 0；沒有建立 person-only 新資料版本 |
| BBAT5 | dataset ID `bbat5-v1`；train 5,964／val 683；沒有獨立 test |
| registry | `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml` |
| Pose YAML | `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml` |
| 二類 Detect YAML | 同 configs 內的 `detect.yaml`；與 Pose 共用 assignment／框標註 |
| Pose 類別／關鍵點 | ball=0 、 bat=1；kpt_shape=(2,3)，不是人體 17 點 |
| 常用正式驗證 | imgsz 640 、完整 val 、 FP32 、 Float-PWL／BitTrue-PWL 分開記錄 |
| 軟體 | Ultralytics 8.4.90 、 Python 3.12 、 Torch 2.11.0+cu128；完整版本見封存 freeze |
| GPU | RTX 5090 約 32 GB；未上 FPGA／ASIC，未做板端能耗驗證 |

所有表格若未另述，均為 AP50–95 、 0–1 尺度，不是 AP50，也不是百分比。例如 +0.005 是 +0.5 個百分點。 COCO sports ball／baseball bat AP 與 BBAT 二類 Pose 的 box AP 不可混比；COCO Detect 在 BBAT 上泛化時，曾明確映射 COCO class 32→ball 、 34→bat，不把 class 0 person 當 ball 。

歷史 source archive／legacy 567 張 validation 不作新入口。 runtime View 只是 canonical 的可重建 symlink／cache 容器，不是新 split 。資料與 labels 不重切、不抽樣、不修改。反覆在同一 val 上選模型屬開發選擇，沒有獨立 test 或多 seed 顯著性證明。

### 2.1 三種門檻不能混用

- **訓練安全線**：判斷是否保存後停止，不代表模型通過驗收。
- **階段／parent 相對門檻**：例如新 KD 相對 qSiLU 起點各 AP 下降不超過 0.001，或方向 1 相對配對 control overall 至少 +0.001 。
- **原獨立模型融合門檻**：COCO 降幅不超過 0.005，六項 BBAT 降幅不超過 0.02 。部分 activation 的 best_joint 只是 activation-relative selector，未全過這個原門檻。

best_pose 的評分規則也隨階段而不同：部分融合階段取 overall box／Pose 平均，Pose-focus 取六項 BBAT 平均；不能僅依檔名視為最高 keypoint AP 。精確定義以各 run resolved config 、 summary 、 selector state 為準。

## 3. 全部研究階段與狀態

| 階段 | 已做內容 | 結果／處置 |
| --- | --- | --- |
| 舊融合後方向 1 | 原 BEST 重驗、 EMA age 、 native 5 、 LR×0.25 、 BN 、後續各分支原始記錄 | 沒有取代舊 BEST 的足夠證據；保留早期 artifacts |
| 融合前 Full35-B100 | FP／A0／B100 基準、 QK 梯度恢復、 narrow／late 適應 | 部分 COCO 恢復，不代表回到 FP |
| HOG | raw P3 HOG9 、 train-only μ 校準、 native5／HOG10 上限 patience4 | HOG E4 停止，不採用 |
| RepConv | 只 layer17，初始化與部署折疊等價，5 輪配對 | 不達門檻，不擴 layer20 |
| P3 MASF | shared／Detect-only／head 延長／one2one bridge | AP 差很小；使用者選 P3 bridge 繼續研究 |
| P2 MASF | 不加 Detect head，5 輪配對 | 未達增準門檻，停止此位置方向 |
| 現成權重 BBAT 驗證 | 七組新 Detect 權重＋歷史 Pose MASF 推論消融 | 新位置候選無一致收益；歷史 Pose 消融不能代替重訓因果對照 |
| 重新 combine | Pose head 適應、完整 Pose 、 J0 、 balanced J1/J2/J3 | COCO 保護較好，但 BBAT 不達原最終門檻 |
| BBAT 恢復 | 新舊同口徑、 Pose head 9 輪至平台、 BN128 校準 | 框部分回升，不足全面補回 |
| Activation | 四臂 zero-shot 、 SiLU／qSiLU 各 10 輪、匯出重驗 | qSiLU E2 作 KD 學生；Hardswish／PolyShift 未擴訓 |
| 雙教師 KD | YOLO26L Detect＋獨立 BBAT Pose；K0／spatial 各 5 輪 | bat Pose 改善但 ball 框下降，未升版 |
| Pose-head KD | 固定共享／Detect，只更新完整 Pose head，5 輪 | E2 keypoints 小升、框下降；native 正式對照由使用者取消 |
| 推論重組 | 原框分類＋KD E2 keypoints，兩後端全量驗證 | 框精確保留，但 Pose 無增益，不升版 |
| 尚未實施 | 新共享 Conv 配對、區域／排序創新 KD 、 PTQ/QAT 、板端部署 | 保留為條件式方向，不冒充完成 |

完整 run 、 smoke 、安全停止、失敗啟動與 raw metrics 由封存 `run-index.json`、`metrics-long.csv`、`manifest.jsonl` 追溯；這些包含歷史／診斷，不全部視為正式獨立實驗。原始資料分布與全部超參數 JSON 一併保存。

## 4. 架構演進與硬體契約

### 4.1 原 B100／舊 shared MASF

```text
layer16：p3_raw → MASF → p3_shared ─┬─> layer17 → P4 → layer20 → P5
                                   └─> Detect P3
layer19：p4 ───────────────────────────> Detect P4
layer22：p5 ───────────────────────────> Detect P5

舊融合版：Pose 也共用被改動的 P3/P4/P5 。
```

在 P3/P4/P5 共用節點加入 MASF 並非數學上不合法，但它不是 P3-only：P4/P5 下採樣路徑也被改變，已訓練下游會遇到特徵分布改變，且不能隔離收益來源。

### 4.2 現在採用的 P3 bridge

```text
layer16：p3_raw ────────┬─> layer17 → P4 → layer20 → P5
                       ├─> MASF(α) → p3_det ─┐
                       └─> raw Pose P3       │
layer19：p4_raw ─────────────────────────────┤
layer22：p5_raw ─────────────────────────────┤
                                            ↓
                            Detect([p3_det,p4_raw,p5_raw])
                            Pose  ([p3_raw,p4_raw,p5_raw])
```

α=0 可以關掉同一 checkpoint 的 MASF 殘差貢獻，但不能等同無 MASF 從頭訓練；也不保證實作會跳過 context 運算。新模型的 Pose 不經過 MASF，所以單改 α 的六項 BBAT AP 精確不變是接線結果，不是證明 MASF 從來無用。

one2one 原生路徑 detach 已增強特徵，阻擋其 loss 直接更新 MASF；one2many 仍有梯度，不能稱 MASF 完全沒有梯度。 bridge 把 detach 移到 raw P3 之前：`detach(raw P3) → 同一 MASF → one2one loss`，只縮放回到 MASF 的梯度。λ=0.012076444778011642 來自 train-only 校準，不是推論 scale；新路徑不回傳 backbone，推論仍只計算一次 MASF 。

### 4.3 P2 與成本

P3 MASF 在 80×80×256，Conv MAC=H×W×(9C+25C+C²)=0.475136 GMAC，按 2 FLOPs/MAC 為 0.950272 GFLOPs；未融合 BN 參數 75,777 。 P2 160×160×256 同模組為 1.900544 GMAC，僅 MASF 成本為 4 倍，不是整網 4 倍。 1×1 channel mixing 約占 MASF Conv MAC 的 88.28%。 bridge 相對原 shared P3 不增加推論 Conv MAC；訓練多一次 MASF forward/backward 。

這些是明確算式估計，不包含記憶體流量、 BN 、 activation 、 XNOR／popcount／PWL 異質硬體成本，不能換算成已測板端速度。詳見[架構與計算式](<../../experiments/combine/pose-masf/ARCHITECTURE.md>)。

### 4.4 BinaryQK／PWL／activation

BinaryQK 使用固定 PoT scale，沒有新增每圖動態 8 選 1 scale selector；8-scale/codebook 為保留提案，不是現行已訓練方案。布林 XNOR／整數 reduction 切斷 autograd 的缺陷已定位；training-only surrogate 保持精確二值前向，backward 用 signed-dot 與 clipped STE 。它修的是訓練通道，不代表二值表示不再損失資訊。

PWL 範圍已確認 `[-10,0]`、 20 段。 knots／values／endpoint table 是固定 buffers，不是本輪可訓練 Softmax 參數；Float 和 BitTrue 都不等於全 FP-QK 。最後 normalization 還有 exact software reciprocal reference，不能稱全整數無除法 Softmax 。

qSiLU-PQ 使用 |x| 節點 0 、 1 、 2 、 4 、 8 與固定 dyadic 分段二次係數，無可訓練 activation 參數、無逐圖 scale；保留原硬體友善方向，但 PyTorch GPU 上反而較慢。 activation 必須由重建程式恢復，state dict 不包含它的類型。

## 5. 早期融合後：EMA 、 LR 與 BN 的教訓

舊 BEST BitTrue：COCO0.498022 、 person0.620381 、 BBAT 框 0.630036 、 Pose0.903717 、 ball 框 0.507437 、 ball Pose0.859909 、 bat 框 0.752634 、 bat Pose0.947526 。

相同 live 軌跡，fresh EMA joint0.707247，continued EMA joint0.711226，parent0.711175 。 continued 初始 age26597→27060，EMA 對起點約保留 95.41%的累積係數，因此分數接近 parent 並不代表 live 已改善。 native5 E5 ball 框 0.502376，比 parent 下降 0.005061 而安全停止；LR×0.25 到 E2 live ball 框 0.491465，相對原 LR0.480829 有所改善，仍不及 parent 。換回 head BN 後 0.495485，仍不足。

固定 state 稽核沒有發現 frozen scope 失效。 24 筆共享梯度樣本負 cosine 比例 62.5%，但中位 cosine 約−0.012162，負向投影幅度約 1.22%，不能僅看負 cosine 比例就宣稱強烈 task conflict 並加入投影。梯度日誌的 clip_grad_norm_ 回傳值為裁剪前 norm，不能把頻繁>10 解讀成裁剪沒生效。

結論是：EMA 、 BN 與 LR 都是可能影響因素，但沒有單一根因已完全解釋退化。細節、全 epoch 表與測試紀錄見[原生／LR／BN 報告](<../../proposals/integrated-roadmap/native5-results.md>)、[EMA 報告](<../../proposals/integrated-roadmap/ema-age-diagnostic-results.md>)。早期其他 run 不省略：全目錄 `experiments/artifacts/direction1-20260908/` 保存於封存，不把摘要未逐列列出的 smoke 當不存在。

## 6. 融合前方向 1 的完整主結論

### 補充：早期融合後的其他已做實驗

早期融合後方向 1 與下方融合前方向 1 是不同 parent，不能混比。[早期證據稽核](<../../proposals/integrated-roadmap/round1-audit.md>)已有 6 組訓練、 25 個完成 epoch 、 50 筆 EMA／live 八項 AP，完整[逐 epoch CSV](<../../proposals/integrated-roadmap/results/epoch-comparison.csv>)一併封存。

| 早期融合後項目 | 實際完成與限制 |
| --- | --- |
| BEST 篩選 | J3 joint／pose 與 J2 全量重驗；J2 ball 框 0.510866 較好，但 person 0.618511 、 bat Pose 0.942970 下降，仍保留 J3 |
| Batch 探測 | 32×4 、 64×2 可行，128×1 OOM；吞吐差小，保留 physical32 |
| HOG | E1–E4 因安全 gate 停止；這組不是 patience4 造成停止，與融合前 HOG 分開 |
| RepConv17 | 完成 5 輪與 init／fold／reload／snapshot 驗證，未接受 |
| MASF 關閉 bridge | alpha0 診斷與 BR-OFF 4 輪；未過 gate，不直接搬移 shared MASF |
| BinaryQK 單點 | site10／22 改 FP-dot 全量驗證均下降；不是純 FP 教師，非 KD 已完成 |
| 固定 scale 早返回 | 原 16 個 slots 不變；跳過無用 dynamic reduction，CPU 兩 backend 及完整 8 AP 相同；尚未裝入正式 export，不是已測加速 |
| B-HEAD | 完整 5 輪、子分支 moments／更新／凍結稽核，減輕退化但未採用 |
| MuSGD 前置 | 修正版 48 macro 、另保留初版 48 macro；Detect no_decay 更新比 0.2685 低於 0.5，recipe 拒絕，不啟動 20 輪 |
| 視覺 | 完整 683 張逐圖資料、 12 張 GT／BEST／native E5 HTML/SVG 已產生；不是使用者真實影片盲測 |

以下回到融合前獨立 Detect 研究。

| COCO AP50–95 | overall | person | sports ball | baseball bat |
| --- | ---: | ---: | ---: | ---: |
| FP | 0.518019276 | 0.630794912 | 0.526040758 | 0.483490044 |
| BinaryQK A0 | 0.506738574 | 0.626805274 | 0.513110259 | 0.495725734 |
| B100 BitTrue | 0.503589001 | 0.624111237 | 0.516234472 | 0.469482418 |
| 無 MASF head control E8 | 0.508267092 | 0.627699455 | 0.514005652 | 0.480127550 |
| P3 bridge E8 | 0.508211955 | 0.627664127 | 0.513148090 | 0.480664186 |

A0−FP overall −0.011280702；B100−A0 −0.003149573 。這是歷史模型鏈，不是只改一個開關的完整重訓消融。 bridge−B100 +0.004622954，包含多階段適應，不能全歸於 MASF 或 QK 修正。 bridge−無 MASF control −0.000055137，沒有足夠增準證據。

HOG 的原生 control 就是相同預算框架下的原生 Detect loss，不是額外一種 HOG 演算法。 HOG9 頭是 training-only 邊緣方向直方圖監督，部署移除。原定 native5 、 HOG 最多 10 、 patience4 、 warmup1；實際 HOG E4 停止，μ約 0.875 為 P3 梯度 5%校準。既有 train 診斷中 ball 13/59 、 bat1/38 無有效 P3 cell，說明小物件 cell-center 監督覆蓋有盲區，但不是已證明唯一退化原因。

RepConv 只試 layer17，先驗證等價初始化與 BN 折疊，不全面替換。最佳 E4 overall0.508097536／person0.627500653，未超過起點與配對門檻，E5 回落，所以不加 layer20 。

MASF control／shared／fork 各 5 輪，control／fork 再續 E6–E10，bridge 由同 fork E5 續訓。提高 P3 head LR 與接通直接任務梯度都未產生足夠 AP 增量。 P2 兩臂各 5 輪，E5 相對 control overall−0.00014887 、 person+0.00009926；BBAT ball 框−0.000957 、 bat 框+0.000606，方向混合；約 39%回合時間增加是該訓練觀察，不是推論延遲。

七組已訓 Detect 在 BBAT 的完整表見[MASF 結果](<../../experiments/combine/pose-masf/RESULTS.md>)。它們沒有 Pose head，因此沒有 P2／P3 新 keypoint 結果。舊獨立 Pose 的 MASF on−off ball 框+0.005220 、 bat 框+0.004674，bat keypoint−0.001746；這是舊 checkpoint 依賴性消融，不能套成新 bridge 的因果收益。

## 7. 重新 combine 與 BBAT 恢復

使用者最後選擇 P3 bridge 繼續，因此未採用 P2／無 MASF 的歷史暫定主線。先做 Pose head 適應 40 輪，最佳 Pose 約 0.852475；再完整 Pose 適應，初次 LR 觸發安全停止，降低 LR 後 42 輪平台，E30 Pose 約 0.897997 。這仍低於原獨立 Pose 約 0.9122，所以融合前 Pose 品質已是限制之一，不是只能怪融合。

重組共享 trunk 後 J0 完成 11 輪，E1 Pose 約 0.866087；balanced J1 完成 14 輪，J2 完成 23 輪，J3 完成 17 輪。 J2 best 平均 E22：COCO0.505620 、 person0.626900 、框 0.605189 、 Pose0.883299 。 J3 E12 如下，仍無原嚴格門檻的 best_joint 。

| BitTrue AP | 舊 combine | 新 bridge J3 E12 | head 恢復 E5 | qSiLU E2 |
| --- | ---: | ---: | ---: | ---: |
| COCO 框 | 0.498022 | 0.504242 | 0.504242 | 0.503885 |
| person 框 | 0.620381 | 0.626983 | 0.626983 | 0.625887 |
| BBAT 框 | 0.630036 | 0.603892 | 0.608773 | 0.618008 |
| BBAT Pose | 0.903717 | 0.886071 | 0.886747 | 0.891329 |
| ball 框 | 0.507437 | 0.486159 | 0.492201 | 0.505192 |
| ball Pose | 0.859909 | 0.855779 | 0.854418 | 0.859649 |
| bat 框 | 0.752634 | 0.721624 | 0.725345 | 0.730823 |
| bat Pose | 0.947526 | 0.916362 | 0.919075 | 0.923008 |

qSiLU 比舊 combine 保住更多 COCO／person，但 bat 框仍約低 0.021811 、 bat Pose 約低 0.024517，不能宣稱已追回。此表跨階段不控制所有變因，只呈現現有可用模型取捨。

新舊完整重驗確認不是誤用 split 或 PWL 範圍。新 Float／BitTrue 差約 0.00013 量級，遠小於當時 BBAT 缺口；但兩者都 BinaryQK，不能以此證明二值化沒有代價。舊模型同權重 MASF 開關的 bat 貢獻很小，不足解釋整個 bat 優勢，也不排除歷史訓練影響。

head-only 恢復固定 827 個非 Pose state，360 個 Pose state 改變，COCO 完全不變，ball／bat 框部分回升。 9 輪平台 E5 最佳，ball Pose 略降。完整 train5964 、 physical128 的 BN-only 校準僅小幅混合改善；它沒有反向，不能據此宣稱訓練 batch128 可行。詳見[恢復報告](<../../experiments/combine/bridge_v1/BBAT_RECOVERY_RESULTS.md>)。

## 8. Activation 結果

相同起點 SiLU／qSiLU 各 10 輪、 seed1，SiLU 最佳 E9 、 qSiLU 最佳 E2 。 qSiLU−SiLU：COCO−0.000423 、 person−0.000896 、 BBAT 框+0.005175 、 Pose+0.001885 、 ball 框+0.011293 、 ball Pose+0.004936 、 bat 框−0.000942 、 bat Pose−0.001166 。 joint 約+0.001525，不是每項都提高。

zero-shot qSiLU 八項下降均小於 0.015；Hardswish 最大約降 0.158，PolyShift ball 框約降 0.024，沒有擴大它們的短訓。 qSiLU physical32 backward OOM，兩臂共同改 16 後通過；峰值 allocated SiLU10.81GB／qSiLU19.15GB 。整個 queue 時間 SiLU 約 2h30m／qSiLU6h37m，包括訓練驗證存檔，不是純推論 latency 。

qSiLU 匯出已獨立重建並精確重現八項 BitTrue AP 。直接將學生換成 FP-QK teacher 候選後八項全降，因此未拿它直接蒸餾，後來改用各任務強教師。詳見[Activation 詳細結果](<../../experiments/activation/bridge_v1/RESULTS.md>)。

## 9. 方向 2：KD 實際做了什麼

### 9.1 雙教師空間蒸餾

COCO batch 用 YOLO26L Detect 教師，BBAT batch 用原獨立 BBAT Pose 教師，不混 class IDs 、不補造另一任務標註，不使用人體 17 點教師。學生 tap 為 layer16／19／22，`A(F)=L2_normalize(mean_channel(F²))`，三尺度空間距離；teacher eval/no_grad，不進 optimizer／EMA／export，沒有部署 projector 。這是普通 Attention Transfer 風格基線，不是已完成區域排序創新。

K0／KD 各 5 輪。 train-only μD=0.8214335621 、μP=1.7886982661，目標 Neck feature 梯度比 0.1，固定不逐圖變動。 MuSGD 本起點的 attention 更新中位數 0，無法有限校準，拒絕該配方而用 AdamW；不是 MuSGD 普遍無效。

KD 最佳 Pose E4：COCO0.502955 、 person0.625193 、 BBAT 框 0.613722 、 Pose0.894121 、 ball 框 0.495441 、 ball Pose0.859507 、 bat 框 0.732003 、 bat Pose0.928735 。相對 qSiLU 有 bat Pose+0.005727，但 ball 框−0.009751；六項 BBAT 平均亦未勝起點。 K0 也有退化，不能把全部變化歸於 KD 。沒有新合格 best_joint，不原樣延長。完整配方見[KD 計畫](<../../experiments/kd/dual_task_v1/PLAN.md>)。

### 9.2 Pose-head KD

由同一 qSiLU E2 重新開始，不接退化 KD E5 。只用 BBAT 訓練完整 Pose head，凍結 shared／Detect／MASF／BN state 。 12 個 feature taps 來自 cv2 、 cv4 、 one2one_cv2 、 one2one_cv4，各 3 尺度。μ=10.063553373151326 對應 head feature 梯度 5%；係數大是 loss 正規化尺度，不代表梯度很大。

正式 native 對照經使用者取消，保留已完成 smoke 與取消 run；正式 head KD 完成 5 輪／1865 macro，約 18m50s 。不能稱有完整 native 配對，也不能把這個 epoch 速度與原 joint 每輪 50 分鐘直接相比。

| BitTrue AP | qSiLU 起點 | head KD E2 | head KD E5 |
| --- | ---: | ---: | ---: |
| COCO 框 | 0.503885 | 0.503885 | 0.503885 |
| person 框 | 0.625887 | 0.625887 | 0.625887 |
| BBAT 框 | 0.618008 | 0.613915 | 0.611005 |
| BBAT Pose | 0.891329 | 0.892962 | 0.890193 |
| ball 框 | 0.505192 | 0.501879 | 0.495931 |
| ball Pose | 0.859649 | 0.861716 | 0.858932 |
| bat 框 | 0.730823 | 0.725951 | 0.726079 |
| bat Pose | 0.923008 | 0.924207 | 0.921454 |

固定非 Pose state 與 Detect 輸出精確不變，證明隔離有效；但 E2keypoint 小升伴隨框下降，E5 更差，未升版。來源與全 5 輪結果見[Pose 專項](<../../experiments/kd/pose_focus_v1/README.md>)。

## 10. 推論重組、效果與限制

只取原框分類＋KD E2 keypoints，保留原 shared 和 Detect，換 96 個 keypoint state，新增參數／算子為 0 。 CPU 兩後端原框／分數精確保留、 KD raw keypoints 精確接上；獨立匯出 roundtrip 與全量驗證均通過。

BitTrue 框與 COCO 精確回原模型，但 Pose0.890906 低於 0.891329，ball Pose 不變、 bat Pose0.922163 下降 0.000845 。因同一候選的框、分數、 keypoints 共用索引，AP 會受分數排序與選取影響，不能把兩份最高指標直接拼接；未量化每個因素占比。候選保留、不取代預設。

本機實測 end2end=True，NMS fast path 只按 conf 篩選，iou 不參與 suppression 。其後已完成同一原權重 one2many＋class-aware NMS 比較，結果見第 15 節；不是仍待執行的工作 。詳細結果、原始 JSON 與程式見[推論研究](<../../experiments/inference/pose_branch_v1/README.md>)。

## 11. 超參數總表與可重建邊界

| 階段 | 主要更新範圍／LR | 預算與 batch |
| --- | --- | --- |
| 早期 native | Neck1e-5 、 heads2.5e-5；LR 對照乘 0.25 | native5；physicalDetect32 、 logical128；Pose16 |
| A0 梯度恢復 | Q/K5e-7 、 head2.5e-5，其他固定 | 10 輪 horizon／warmup1；32×4；完整 COCO 每輪 925 更新 |
| HOG | 同 parent 原生範圍＋HOG 頭，μ校準約 5% | native5／HOG 最多 10 patience4，實際 E4 停 |
| MASF bridge 續訓 | P3head1e-5 、 context1e-5 、 alpha1e-4 | E6–E10；延續 optimizer/EMA/scaler，不重 warmup；32×4 |
| J3 | backbone3.8e-7 、 Neck1.9e-6 、 heads5e-6 、 MASF1e-6 、合法 attention5e-8 | 上限 20 patience5，完成 17；Poseweight0.215 經校準 |
| Pose 恢復 | 只 Posehead2e-5 | 上限 10 patience4，完成 9／E5 最佳；batch16 |
| Activation／雙教師 KD | backbone3.8e-7 、 Neck1.9e-6 、 MASF3.8e-6 、合法 attention5e-8 、 heads5e-6 | activation 各 10／KD 各 5；Detectphysical16 logical128；macro256Detect+16Pose |
| Pose-head KD | 只 Posehead1e-5；非 Pose 全部固定 | 5 輪、 patience0 、 batch16 、 373macro/epoch |

主要 AdamW 共同項為 betas(0.948,0.999)、 eps1e-8 、 WD0.00027 、 clip10 、 warmup1；不同 stage 是否衰減 BN/bias 、 scheduler horizon 、 BN affine 與 running 、 seed 、 augmentation 、 task normalizer，以保存的 resolved config 與程式為準，不把總表當一份可通用所有 stage 的 yaml 。

Physical batch128 目前只在無反向 BN 校準有證據。梯度累積 128 不等同 BN 一次看到 128，也不保證更快；epoch 時間取決於影像數、 task 比例、可訓範圍、 teacher forward 、完整驗證與存檔。 Pose-only 去除 COCO backward 且共享層無反傳，變快不代表偷減少完整 BBAT 資料。

## 12. Checkpoint 選擇、保存與恢復

| 用途 | 目前路徑（相對 optimize，除非絕對） | 狀態 |
| --- | --- | --- |
| 原舊 combine | `/home/uxin/yolo/yolo_combine/final/full35/weights/combined/inference/best_joint.pt` | 歷史 BBAT 強參考，保留 |
| 融合前無 MASF 對照 | `experiments/studies/pre-fusion-full35-b100/artifacts/direction1-candidate-verification-v1/control-e8-bittrue.pt` | 完整 COCO 重驗 |
| 融合前 P3bridge | 同上 `masf-e8-bittrue.pt` | 使用者選定研究路線 |
| 新 J3 恢復 E5 | `experiments/combine/bridge_v1/artifacts/fusion/j3-pose-head-recovery-v1/inference/best_pose.pt` | activation 父模型，非原 gate 全過 |
| 現行 qSiLU E2 | `experiments/activation/bridge_v1/artifacts/runs/qsilu_pq-short-e10-seed1-v1/inference/best_joint.pt` | 新研究預設；activation-relative |
| headKD E2 | `experiments/kd/pose_focus_v1/artifacts/runs/kd-e5-seed1-v1/inference/best_pose.pt` | 框退化，不升版 |
| 分支重組 | `experiments/inference/pose_branch_v1/artifacts/branch-v2/candidate.pt` | 負結果，不升版 |

qSiLU SHA256 `1bbbbe0f87cb853827af1bb625e7066178e3fb2c7e14c410d997efb24e94834a`；headKD E2 `b67a5a82c3b36ff1bc701630ab7b874b06485ca2c9c6fff42074f6f202ebe0e4`；P3bridge `73705178305e55e497ead1e1d8948114b67e1d17bf528861c9606e66cd372721`；完整其他 SHA 以封存清單為準。

inference 檔與訓練 snapshot 分開保管。同 run 的`checkpoints/`通常保存 optimizer／EMA／scaler／RNG 等，但「有 snapshot 」不等於任意時刻 exact resume 已驗證；融合前部分 runner 尚無完整中斷 resume CLI 。封存工具不反序列化 checkpoint，只驗證 bytes 完整；未逐一做所有歷史 checkpoint 載入測試。自訂模型要保留程式與 non-state 配置（qSiLU 、 BN eps 、 PWL backend 、 MASF 位置、 class mapping），不能只搬 pt 後用原生 YOLO 直接載入。

安全載入維持 weights_only=True；需 pickle 自訂類別者只能用已核對 SHA 與有限 safe_globals，不能改 weights_only=False 省略檢查。 qSiLU 重建入口`experiments/activation/bridge_v1/verify_selected.py::SelectedSource`，Pose 重組入口`experiments/inference/pose_branch_v1/probe.py`；正式驗證用相同 materialization 與 canonical 資料。

### 12.1 封存內容

獨立副本位於 `archives/research-through-20260911-v1/`，不搬走來源。保存 optimize 全部一般研究檔案（包括大型 pt 、日誌、圖片、 PDF 、 JSON 、 CSV 、程式、設定、歷史報告），排除 cache 、 runtime 影像／labels 、 symlink 與 archives 自身；runtime JSON/YAML/manifest 仍保留。另保存必要外部 final 權重、教師 Pose 、架構／attention／combine／activation 原始碼、 registry 及 pip freeze 。

`manifest.jsonl`記錄原絕對路徑、封存相對路徑、大小、 mtime 與 SHA；每份 source 和 copy 都重新雜湊比對，期間來源改動即失敗。`checkpoints.json`是所有納入 pt/pth/onnx/engine/safetensors 的清單；角色只是路徑提示，不假稱已能 resume 。`metrics-long.csv`保留來源 JSON pointer，所以 checkpoint 、 baseline 、 delta 、 EMA/live 不會在 CSV 中被無標記混成一筆；使用前須看 pointer 。`run-index.json`含 summary／stop 入口，不把缺少 status 欄位擅自判為成功。`audit-anomalies.json`記錄 JSON 解析／非有限數值。

這是同磁碟 snapshot，不是異機容災備份；環境 binary 與資料集影像仍是外部依賴，絕對路徑在另一台機器需明確修正並重驗。全檔 SHA 通過只證明保存完整，不證明每個歷史模型精度合格。封存後新增結果另存補充包，不偷偷改已封存版本。

## 13. 下一輪必要且有用的方向

1. **既有推論分支比較（已完成，見第 15 節）。** 同 qSiLU 權重比較 one2one 和 one2many+NMS，全 BBAT val 、相同 imgsz/conf；COCO head 保持原路徑。記錄六項 AP 與算子路徑，不拿 archive I/O 期間的速度當 benchmark 。若不能改善，不增加 ensemble 或 TTA 。
2. **框與 keypoint 一致的訓練訊號。** 本次重組反證了「可獨立拼接最佳 AP 」；下一個可研究的創新是假設以有效物件區域／候選品質為配額，讓 teacher 的框、排序和 keypoint 監督指向同一候選。先量 ball 覆蓋與對齊，不直接再堆多個 loss；不是已完成的創新成果。
3. **小範圍共享特徵適應。** 已獲先前授權，但仍需同 parent 對照。候選只解凍實際確認的 P3 輸出 Conv，head LR1e-5 、 shared 初探 2e-7 、 3 輪、 warmup1，其餘硬體常數／BN running／Detect 不變；先校準真實更新與安全線，再啟動。 shared 改動會影響 COCO，不宣稱只訓 Pose 就保證 person 不退。
4. **硬體 Softmax 剩餘成本。** 真正 reciprocal 近似／定點累加誤差是部署必要研究，不再把當前 PWL 稱完整無除法；先 BitTrue 差異與溢位界線，再決定是否需 QAT 。

8-scale/codebook 、區域 ranking 、 conflict-safe projection 、較大 backbone 、私有 Pose adapter 保留原提案，但需證據觸發；不全面啟動。 person-only 維持延期。 HOG 新覆蓋設計若重啟必須另立對照，不能以調參後結果覆蓋已失敗 HOG 版本。無增益就停止同配方，不用放寬 gate 製造成功。

## 14. 已知缺陷、修正與未驗證

正式 QK 測試曾測到理想可微 matmul 而非 production XNOR，已用真正 checkpoint seam 重現。舊 Pose smoke 在 zero_grad 後讀 grad 的附加斷言失敗，不能當 MASF 無梯度證據。新建舊模型驗證骨架漏設 BN eps，修成 1e-3 後原八項 AP 重現。 qSiLU OOM 經雙臂共同 physical16 修復。最新分支 probe v1 把 Float state 覆到 BitTrue 額外 endpoint_table 而失敗，改只載入 Pose head 後 v2 完整通過。所有失敗 log 保留。

總報告整理時發現歷史「未開始」敘述，已補最新狀態提示。 MASF 稽核實際位於上層 `/home/uxin/yolo/docs/worklogs/2026-09-10-masf-results-audit.md`，不是 optimize 本地 docs；原相對連結有效，不列為文件缺陷。未做真實使用者影片最終目視驗收、多 seed 、獨立 test 、 ONNX/TensorRT／板端等價或能耗測量，不能用 AP 小增保證實際畫面變好。

### 保存／清理決策

| ID | 精確目錄 | 類型與依賴 | 建議 |
| --- | --- | --- | --- |
| C01 | `/home/uxin/yolo/yolo_optimize/experiments/artifacts` | 早期 run 、 checkpoint 、失敗與 EMA 證據，約 39G | Keep |
| C02 | `/home/uxin/yolo/yolo_optimize/experiments/studies` | 融合前全部來源與配對，約 41G | Keep |
| C03 | `/home/uxin/yolo/yolo_optimize/experiments/combine` | 融合／恢復／事件，約 15G | Keep |
| C04 | `/home/uxin/yolo/yolo_optimize/experiments/activation` | 四臂／配對／選定權重，約 4.4G | Keep |
| C05 | `/home/uxin/yolo/yolo_optimize/experiments/kd` | 教師／校準／完整兩階段 KD，約 5.9G | Keep |
| C06 | `/home/uxin/yolo/yolo_optimize/experiments/inference` | 重組失敗與成功驗證，後續推論比較 | Keep |

大小是盤點時 du 近似值，含可重建 view；依賴不明也保留。沒有刪除建議、沒有進行清理，無需為刪除索取授權。此處描述 2026-09-11 封存當時狀態；之後報告已發布，本次再以新目錄更新。checkpoint 封存不納入 Git 。

## 15. 詳細原始報告入口

### 本輪保存驗收與新推論附錄

主封存實際完成 5,817 個檔案、 113,656,761,817 bytes（113.66 GB），其中 406 個 checkpoint／模型產物共 104,705,418,919 bytes 。全部來源／副本 SHA256 比對通過，再核對全部 5,817 個封存路徑與大小；關鍵 qSiLU／KD hash 與已驗證來源一致。原 JSON 解析與 AP 非有限數值異常為 0 。這不是逐一載入所有 checkpoint 的驗證。

可讀索引：[checkpoint CSV](<checkpoints.csv>)、[run CSV](<runs.csv>)、[當前模型精確數值 CSV](<current-model-comparison.csv>)、[保存檢查 JSON](<PRESERVATION.json>)。完整原始 metric／pointer 表見主封存 `metrics-long.csv`，不把不同 bank／baseline／delta 合併成獨立實驗。

主封存後追加 one2many＋NMS：Float／BitTrue 各完整 BBAT val683，無訓練、原權重 SHA 不變。 BitTrue bat 框 +0.012767584 、 bat Pose +0.032442655，但 ball 框 −0.010844021 、 ball Pose −0.016198590，故不全面切換。下一個候選是 ball 用 one2one 、 bat 用 one2many；尚未實作或驗證，且同時計算兩條 head 會增加成本，不能拼接兩份最好 AP 宣稱成功。見[新推論詳細報告](<../../experiments/inference/routing_v1/README.md>)。

最新報告、索引、新推論及本機實際 Ultralytics 原始碼另存 `archives/research-through-20260911-addendum-v1/`；第一包不修改。最終副本保存結果以各包 `summary.json` 為準。 PTQ／QAT 仍保留延期邊界，需另確認階段，不因本次統整擅自啟動。全部本輪 GPU 工作已結束，沒有背景訓練 queue 。

- [方向 1 主計畫與原始來源盤點](<../../proposals/integrated-roadmap/README.md>)
- [融合前方向 1 完整報告](<../direction1/README.md>)
- [MASF 七組 BBAT 與歷史 Pose 消融](<../../experiments/combine/pose-masf/RESULTS.md>)
- [融合與恢復結果](<../../experiments/combine/bridge_v1/BBAT_RECOVERY_RESULTS.md>)
- [使用者論文實驗 4 的本機閱讀紀錄](<../../docs/research/2026-09-10-combine-report-experiment4.md>)
- [Activation 結果](<../../experiments/activation/bridge_v1/RESULTS.md>)
- [雙教師 KD 計畫與設定](<../../experiments/kd/dual_task_v1/PLAN.md>)
- [Pose-head KD 完整結果 JSON](<../../experiments/kd/pose_focus_v1/artifacts/direct-kd-result-v1.json>)
- [推論分支實驗](<../../experiments/inference/pose_branch_v1/README.md>)
- [全部中文工作紀錄](<../../docs/worklogs/README.md>)
- [全部原始研究文獻／推導](<../../docs/research/README.md>)

總報告以既有實測與本機程式為依據，不重新發明或補造缺失的實驗。細節未在正文逐行重印者，仍完整保存於原始報告／程式／config 與封存副本中。


## 16. 最終選擇：哪些有用，哪些尚未證明

| 問題 | 證據與判斷 | 本輪處理 |
| --- | --- | --- |
| BinaryQK 約降 0.01 能否追回 | A0−FP 為 −0.011280702；精確二值前向的 surrogate 修復真實梯度通道，仍有表示能力差距 | 保留硬體固定尺度與梯度修正，不把全部多階段收益歸因於單一修正 |
| MASF 是否有提升 | P3 bridge 比 B100 高，但相對同預算無 MASF E8 的 overall 差為 −0.000055137 | 依使用者選擇保留 bridge；未證明獨立增準 |
| P2 是否更適合小物件 | 不增加 Detect head 的 P2 試驗收益混合，MASF 模組 MAC 為 P3 的四倍 | 不採用 P2，不為理論解析度優勢忽略實測 |
| HOG／RepConv 是否加入 | HOG 未過門檻；單層 RepConv17 未勝配對起點 | 不部署 HOG、不全面替換 Conv |
| 新融合是否全面優於舊版 | COCO／person 提升，bat box／Pose 仍落後 | 新 qSiLU 主線與舊 combine 都保存，分任務報告 |
| qSiLU 是否值得 | 相對 SiLU 有 ball 與平均收益，但 GPU 時間及記憶體較高 | 保留研究預設，板端加速尚未測量 |
| KD 是否有效 | keypoint 小升伴隨 box 下降；完整 native 對照被取消，不能宣稱完整因果比較 | 保留候選，不升版、不原樣延長 |
| one2many 是否採用 | bat Pose +0.032442655，ball Pose −0.016198590 | 不全面切換；class routing 尚待驗證 |

α=0 僅代表同一模型殘差貢獻歸零，不等價於無 MASF 重新訓練，也不必然省下 MASF 算子。推論部署若要省算量，必須確認導出圖真正移除該分支。

## 17. 本次資料夾實際整理與保存

九個研究相關根目錄移入 experiments，維護工具移至 tools，提案移至 proposals，三個日期型報告目錄改為 final／direction1／publication，合計 14 筆實體目錄搬移。没有建立舊根層 symlink；checkpoint 隨所屬階段搬移，原始內容不變。封存包保持原位且不更動。

為讓程式及設定能找到新位置，更新本機路徑、Markdown 相對連結、少數外部 YOLO_ROOT 解析及模型索引。JSON／CSV 的指標不重新計算，只有路徑文字更動；每個受修改原文保存在遷移快照。checkpoint 內嵌歷史 metadata 不改寫，故其 SHA 不變，但任意 exact resume 仍未驗證。

詳見[目錄地圖](<../../docs/WORKSPACE.md>)、[完整遷移 manifest](<../../docs/history/layout-v2-20260912/manifest.json>)、[操作說明](<../../docs/OPERATIONS.md>)及[模型清單](<../checkpoints/README.md>)。本次沒有刪除權重、失敗 run、log 或 cache，不宣稱減少儲存量。

## 18. 如何驗證本報告與下一步

先確認 current-model-comparison.csv 的 model、backend、metric 和 source，不能把 COCO80 的 sports ball AP 與 BBAT 二類框 AP 混作同一指標。比較 MASF 必須用相同 parent 和訓練預算的對照；跨階段表只能說明目前模型的取捨。

驗證順序為：來源 SHA → canonical 資料版本與 split → source 重建及 PWL／activation 契約 → CPU 基本行為 → 完整相同 evaluator 的 COCO／BBAT 評估 → 每類與小物件分析 → 真實影像／影片目視 → 部署等價與效能。已完成的量測重用原始結果；本次目錄整理只新增必要 CPU 驗證，不捏造新 AP。

目前最合理的後續仍是低成本推論 class routing 的配對驗證，其次才是框／分數／關鍵點一致的區域 KD。共享特徵若再解凍，需同時量 COCO，不可只看 Pose。這些是下一步建議，不在本次整理／發布中自動排入 GPU queue。

## 19. 全階段精度與部署成本

本次另完成 26 組已訓模型的 CPU／GPU 受控比較，未重新訓練；AP 沿用完整驗證，權重 SHA 不變。

[詳細九項比較、算式與所有限制](<../performance/README.md>)／[精確 CSV](<../performance/comparison.csv>)。

- [BinaryQK：精度、大小、算量、峰值記憶體、延遲與能耗](<../performance/stage-1.md>)
- [HOG：精度、大小、算量、峰值記憶體、延遲與能耗](<../performance/stage-2.md>)
- [RepConv：精度、大小、算量、峰值記憶體、延遲與能耗](<../performance/stage-3.md>)
- [MASF：精度、大小、算量、峰值記憶體、延遲與能耗](<../performance/stage-4.md>)
- [融合：精度、大小、算量、峰值記憶體、延遲與能耗](<../performance/stage-5.md>)
- [Activation：精度、大小、算量、峰值記憶體、延遲與能耗](<../performance/stage-6.md>)
- [KD：精度、大小、算量、峰值記憶體、延遲與能耗](<../performance/stage-7.md>)
- [推論：精度、大小、算量、峰值記憶體、延遲與能耗](<../performance/stage-8.md>)

目標硬體未指定／連接，因此 target latency 與 target energy/frame 仍未量測。GPU 能量為 NVML 整卡遙測，不是整機或 FPGA 能耗。MAC／FLOPs 為明確算子範圍的 subtotal；請勿刪去這些限制再引用數字。
