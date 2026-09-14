# MASF 完成後：Attention 梯度／定點 scale-bias 與 Rep17／20

> 最新規格變更：RepConv 改接已完成的 BinaryQK scale_bias E5，見 [新有效排程](<../post_binary_rep_v1/README.md>)。本頁 B5 直出 RepConv 是歷史方案，後續已取消；舊 Rep17 僅保留完整回合結果。scale/bias E5 結果仍有效。


## 現況與最小順序

MASF B5 已完成訓練、獨立驗證、alpha-off 與全部權重稽核；詳見[完整結果](<../pose_masf_training_v1/RESULTS.md>)。目前 B5 是本研究共同起點，並非自動取代正式模型。

本次不再跑未改架構加訓對照，不延長 MASF、不恢復舊 native_qk E3～E20。四個**有方法變更**的候選各做一次固定 5 epoch 適應，warmup 1；不是以反覆多訓湊改善。四者均由同一 B5 開始，不把前一組優缺點混入下一組。

```text
既有 native QK E2 → Pose MASF B5（已完成）→ 共同固定 parent
                                         ├─ 原生 QK＋PWL 參考：重用已完成驗證，不加訓
                                         ├─ ① BinaryQK＋可學固定 scale／定點相對 bias
                                         ├─ ② RepConv：只替換 layer17
                                         ├─ ③ RepConv：只替換 layer20
                                         └─ ④ RepConv：同時替換 layer17＋20
GPU 順序：① smoke → ① E0／5E／重驗／報告 → ② smoke／5E／報告 → ③ smoke／5E／報告 → ④ smoke／5E／報告
```

「拔掉 QK」依前文指**移除 BinaryQK，恢復原生 QᵀK/√d**，不是刪掉 Attention 的 Q/K 計算。此路線已訓練到 native QK E2，且現在的 B5 正是它的後續。它不是只做推論切換；不必為這項對照再重跑相同 native 訓練。舊原生 E2 與本次共同 B5 的分數會分開列，不混叫同一個 best。

## 1. MASF 已完成結果如何解讀

B5 Pose AP50–95 為88.9071%，原 E2 為88.6944%，增加0.2128百分點；ball Pose增加0.4127百分點，COCO50.2215%／person62.4625%不變。但B5把Pose α關成0後，Pose AP為88.9096%，並未下降；僅BBAT框從60.9473%降至60.9137%。因此無法把Pose改善歸因於MASF，缺A加訓對照的限制繼續保留。這次不再為歸因新增A組。

共同來源：`../pose_masf_training_v1/artifacts/b-e5-seed1-v1/epochs/e5/ema.pt`，SHA `1b46fca17f4ae414f622e39afb6d4bae8686ac3a760e23b65fd971500a39e2c7`。Detect MASF與已學Pose MASF都固定，alpha不再繼續訓練。

## 2. 舊 Q/K 為何沒有有效 task-loss 梯度

[舊 score 隔離證據](<../studies/pre-fusion-full35-b100/artifacts/qk-gradient-probe.json>)顯示兩個site都 `score_requires_grad=false`、Q與K梯度非零檢查失敗。原因是 sign後轉成bool、做相等比較與popcount，這條整數／布林路徑不提供對原Q/K的自動微分；單把sign設STE不會自動跨過後續bool/popcount。部分舊訓練policy還會固定qkv.q／qkv.k，所以解鎖一項不代表整条路徑就通。

```text
舊：task loss → PWL → score → bool/popcount ─×─> sign STE → Q/K Conv
新：task loss → PWL → score → dot surrogate backward → clipped sign STE → Q/K Conv
```

對 `Z=QsignᵀKsign`，令G=∂L/∂Z，訓練代理反向為：

```text
∂L/∂Qsign ≈ Ksign Gᵀ
∂L/∂Ksign ≈ Qsign G
∂L/∂Q ≈ (∂L/∂Qsign) ⊙ 1(|Q|≤1)
```

前向仍使用exact binary XNOR/popcount；上述是代理梯度，不是假稱離散sign的真導數。STE選擇可能使優化有效，也可能不穩定，不能從「非零」直接推出AP改善。[STE原論文](https://arxiv.org/abs/1903.05662)

先前修正後的真實GPU smoke（本機保存：`../attention_recovery_v1/artifacts/smoke-scale_bias/summary.json`；本次未上傳）已記錄Q/K、尺度與bias梯度及參數更新，故「目前所有新訓練都沒有Q/K梯度」不符合證據。新版本會再檢查兩個site的Q、K各自task gradient、實際更新、clipped-STE覆蓋率及正負號比例；不拿V的梯度混充Q/K，也不只看requires_grad。

## 3. 固定 scale／bias：硬體規格與推導

保留原有兩個binary basis，不再增加分支。每site4heads×2basis，兩site合計**16個尺度**；所有圖片共用這16個常數，沒有selector、逐圖片校準或平方根。

```text
Z_h = (m_h0 Z_binary + m_h1 Z_hadamard) / 1024
      + bx_h[Δx]/1024 + by_h[Δy]/1024
1 ≤ m_hj ≤ 1024；m為固定無號整數
bx、by：固定signed16-bit整數；相對位移查表
```

訓練使用float master＋STE前向量化，部署固定m/1024，乘固定整數再右移10位。這是dyadic定點常數，不是單一2的冪；若FPGA以shift-add展開固定乘數，實際加法器數量須依m的碼型與綜合結果，不宣稱「只一個shift」或完全沒有乘法。

對學習中的尺度c：`c_eff = c_clip + stop_gradient(round(1024*c_clip)/1024 − c_clip)`。報告逐回合輸出實際部署整數碼；master變了但碼沒變，必須記為「尚未改變硬體前向」，不能宣稱scale已有效調整。

bias使用既有decomposed相對位移表，不新增每head一個全域常數b，因為row-max／softmax中 `softmax(z+b)=softmax(z)`，這種常數可能完全無效。偏置必須隨key相對位置改變，才可能改變同一query對不同key的權重。

兩site最多各504個offset表值，合計1008×16bit＝2016bytes；16個尺度以uint16保存共32bytes，合計約2KiB。這是邏輯常數資料，不包含QKV權重、索引電路或當前PyTorch臨時展開的NxN張量。目標硬體應以Δx／Δy串流查小表，不存完整bias NxN；目前PyTorch仍會展開作參考運算，不等於已完成硬體部署。

### CPU前置抓到的真實風險

最初嘗試signed12-bit m/1024（約±2），但既有bias約−5到+6.5，部分相對位移全部落在clamp外，導致bias梯度為0。只修改新候選為signed16-bit m/1024（−32至32−1/1024），不改舊權重；原CPU失敗測試重跑後通過。這比盲目增加訓練回合更直接地處理無效梯度。

兩條QK路線都固定PWL `[-10,0]`、20段。原生QK的1/√d是常數，可考慮離線合併投影尺度，但目前仍是浮點乘加參考，不冒稱INT8。Binary score、固定係數乘加、bias加法與rowmax減法應先用INT32 reference檢查溢位，再決定板端縮位；PWL最後的正規化仍是軟體除法參考，不宣稱全Attention無除法。

## 4. 為何選 layer17／20 做 RepConv

```text
layer16 p3_raw ──┬──────────────> Detect／Pose P3（各自MASF，固定）
                └─ layer17 3×3 /2 ─> concat(layer13) → layer19 p4_raw
                                                     ├─> Detect／Pose P4
                                                     └─ layer20 3×3 /2
                                                        → concat(layer10) → layer22 p5_raw
                                                                               └─> Detect／Pose P5
```

它們是Neck中清楚獨立、可等價替換的stride2 3×3 Conv：layer17連結P3→P4，layer20連結P4→P5。分開替換容易定位是哪個尺度傳遞有作用，不必重寫C3k2、增加head或在P3高解析位置新增常駐分支。**兩層不直接增強P3 head輸入，不能因ball很小就保證會改善ball。**

| 位置（640輸入） | 原Conv | 新1×1訓練分支參數 | 額外訓練Conv MAC | fold後額外Conv MAC |
| --- | --- | ---: | ---: | ---: |
| layer17 | 256→256，80²→40²，3×3／2 | 66,048（含BN affine） | 0.1048576G | 0 |
| layer20 | 512→512，40²→20²，3×3／2 | 263,168（含BN affine） | 0.1048576G | 0 |

兩層原3×3 MAC均為0.9437184G；新增1×1是該層Conv乘加的1/9，而非整網增加11%。這是有限Conv運算集合的解析值，不是實測latency。

訓練接法為 `Act(BN3(Conv3(x)) + BN1(Conv1(x)))`。1×1支路BN gamma/beta以0初始化，先保持原輸出；stride2無法用同形狀identity shortcut，**此處不加identity分支**。部署時先各自fold Conv+BN，再將1×1 kernel補零到3×3中心：

```text
W3' = gamma3/sqrt(var3+eps) * W3
b3' = beta3 − gamma3*mean3/sqrt(var3+eps)
Wdeploy = W3' + pad_center(W1')
bdeploy = b3' + b1'
```

活化保持在合併後，不能把支路中新增非線性任意fold。結構重參數化允許訓練／推論圖不同，但RepVGG的ImageNet結果不是本YOLO26M一定改善的保證。[RepVGG原論文](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html)、[作者實作](https://github.com/DingXiaoH/RepVGG)

[舊融合前Rep17結果](<../../docs/worklogs/2026-09-09-prefusion-rep17-result-masf-off.md>)：同回合E4相對control僅+0.00516百分點overall、+0.01485百分點person，E5轉差，沒有達到原改善門檻。本次重驗是因模型已換為原生QK、Pose MASF B5的融合起點，且使用者新增layer20需求；不重跑舊control，不把舊結果藏起來。

依使用者最新要求，保留 Rep17、Rep20 兩個單點，另加 Rep17＋20 雙層組。雙層組同樣從固定 B5 獨立開始，不接單層組的已訓權重。新增訓練參數共 329,216，額外訓練 Conv MAC 為 0.2097152G；正確 fold 後額外 Conv MAC 為 0，不代表已實測延遲不變。

## 5. 訓練、驗證與停止

實際設定見[config.json](<config.json>)：每候選5 epoch、warmup1、AdamW beta=(.948,.999)、WD.00027、cosine終點.5、clip10；head LR5e-6、Attention／bias／Rep1e-5、scale2e-4。所有BN running固定，兩套MASF固定，只放開被修改的Attention或Rep層及兩個head。

COCO為epoch主時鐘，完整118287張。延續先前Attention恢復的macro口徑：Detect physical16×16 microbatches＝最多256張＋Pose physical16，每epoch463次更新，reference64、task weights Detect1／Pose.25。Pose沿canonical train循環，463次可涵蓋約1.24次資料；不重切或抽樣另建版本。這不是B組Pose-only的batch128配置，也不把跨任務圖片數加在一起稱batch128。

每回合先保存完整續訓state，再驗證COCO val5000／BBAT5 v1 val683；失敗從保存邊界只補驗。結果列COCO overall／person與BBAT overall／ball／bat框和關鍵點。E5另strict重載BitTrue／Float，Rep使用fold後部署圖。相對自己E0任一AP大跌5pp即保留結果後停查因；不擅自延長回合。

參考採用閘：全部8項AP相對共同B5不下降超過.001，且COCO至少+.001或Pose至少+.002。即使通過，只列候選，不自动替換正式best。沒有未改架構同預算加訓組，所以方法修改與適應訓練的收益仍不能完全分離。

## 6. 實作與產物

- [models.py](<models.py>)：共同來源、fixed scale/bias、原生QK、Rep fold。
- [CPU preflight](<artifacts/preflight-v1.json>)：三候選梯度與重建／fold等價通過；不是AP结果。
- [run_arm.py](<run_arm.py>)：真實雙任務smoke／5E／獨立驗證／報告。
- [run_queue.py](<run_queue.py>)：先等GPU空閒與共享lock，依序處理三候選；正常600秒檢查、無週期log讀取／訊息，完成自動接續。
- `artifacts/<arm>/smoke/summary.json`：真實task Q/K與scale/bias梯度、參數更新、STE覆蓋率、GPU峰值。
- `artifacts/<arm>/train/epochs/e*/`：完整AP、推論權重與实际部署整數常數。
- `artifacts/<arm>/train/checkpoints/`：不可覆寫的逐epoch完整續訓檔。
- `artifacts/<arm>/train/RESULTS.md`：每組中文結論。

ERROR會停止相依工作，保留已完成結果，等待主代理依必要最少資訊修正；STALLED只發事件，不殺外部程序。背景程式不是可自行思考的模型，不保證喚醒已結束對話的代理。所有資料固定canonical，無刪檔，無自動commit/push。


## 7. 最新追加：Rep17＋20 雙層組

[雙層排程計畫](<queue-rep-both-plan.json>)使用獨立接續程式 [queue_rep_both.py](<queue_rep_both.py>)，等待原三組全部完成後，再執行雙層 smoke → 5 epoch → 全量驗證與報告。原 queue 與正在訓練的檔案、設定、權重均不變，不重啟正常工作。等待原 queue 時以程序事件阻塞，不讀訓練 log、不查 GPU；取得 GPU 後維持 600 秒監測節奏。

[雙層 CPU 驗證](<artifacts/rep-both-preflight-v1.json>)已通過：初始輸出保持 B5、兩支路非零時仍可等價 fold、Float／BitTrue 重建，以及兩層的 optimizer 分組與凍結範圍。GPU smoke 會要求兩個新支路各自收到有限非零 task gradient，之後才正式訓練。CPU 通過不代表 GPU 精度已改善。

雙層獨立產物在 `artifacts/rep17_20/`；接續器狀態在 `artifacts/queue-rep-both-v1/state.json`。全部完成後新增 `RESULTS-4arms.md`，不覆寫原三組 `RESULTS.md`，也不自動替換正式模型。
