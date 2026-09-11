# 下一步決策：暫停 COCO 訓練，保護 Detect 的 Pose 專項適應

本次為分析與規劃，未啟動新 GPU 工作、未修改模型或匯出新權重。已完成 K0／雙教師 KD 各5 epoch；本方案取代直接延長該配方。使用者表示 person 暫時不訓練，不自動解讀成同意犧牲整個 COCO80。

## 決定

先只用 canonical BBAT5 Pose train 更新**完整 Pose head**（框、類別、關鍵點各分支及相應 BN），固定 backbone、Neck、Detect head、Detect-only MASF、共享 BN running／affine、Q/K、scale、PWL。不新增 backbone、Neck 或部署 head，不改資料 split。

從 activation 選定的 qSiLU E2 開始，而非 KD E5，也不直接選 KD E4：E4 有較好的整體 Pose，但 ball 框及 COCO 已退步。來源：

`activation/bridge_v1/artifacts/runs/qsilu_pq-short-e10-seed1-v1/inference/best_joint.pt`

SHA256：`1bbbbe0f87cb853827af1bb625e7066178e3fb2c7e14c410d997efb24e94834a`。

```text
BBAT5 影像 → Backbone／Neck（固定）→ P3/P4/P5
                                      ├→ Detect＋P3 bridge MASF（固定；訓練不跑）
                                      └→ Pose head（框＋類別＋關鍵點，可訓練）

COCO 不參加 optimizer 更新；最後保留完整 COCO／person 驗證。
```

若 shared 參數與 buffers、Detect／MASF、圖接線和推論設定全部不變，`Detect(x; θshared, θdetect)` 不依賴 θpose，所以更新 Pose head 不會改變 Detect 數學函數。只設 requires_grad=False 不夠，還必須保護 BN buffers、EMA 與匯出；不能未驗證就宣稱精確不變。

反之，「只餵 Pose 資料但更新共享 backbone／Neck」仍會改變 Detect 特徵，存在 COCO 遺忘風險；目前不採用。person 是 COCO80 的一個類別，不能把停止 person 訓練等同於獨立關閉一個 person head。

## 已有證據：不是只因 person gate 才失敗

以下都是 Bit-True AP50–95，E4 是現有 KD best_pose；不是三個模型重新評估的宣稱，直接讀已完成全量驗證 JSON。

| 指標 | qSiLU 起點 | KD E4 | E4−起點 |
| --- | ---: | ---: | ---: |
| COCO | 0.503885 | 0.502955 | -0.000930 |
| person | 0.625887 | 0.625193 | -0.000694 |
| BBAT Box | 0.618008 | 0.613722 | -0.004285 |
| BBAT Pose | 0.891329 | 0.894121 | +0.002792 |
| ball Box | 0.505192 | 0.495441 | -0.009751 |
| ball Pose | 0.859649 | 0.859507 | -0.000142 |
| bat Box | 0.730823 | 0.732003 | +0.001180 |
| bat Pose | 0.923008 | 0.928735 | +0.005727 |

起點六項 BBAT AP 平均0.754668，KD 最佳 E4 平均0.753921；連不計 COCO 的六項平均也未超過起點。E5 又使 COCO／person 下滑，不能以最後一輪取代最佳候選，也沒有足夠理由原樣加長。

已重播 E5 相對起點0.001 gate，確定失敗項包含 COCO、person、BBAT Box／Pose、ball Box／Pose。K0 原生加訓也有 BBAT 退步；因此不能將所有退步歸因為 KD。只有單 seed，不能將微小差異當統計顯著。

## 原因假說與區分方式

1. 共享特徵適應／任務競爭：預測固定非 Pose 狀態後 Detect 輸出不再漂移；但本輪沒有單教師消融，不能確定是 Detect KD 還是 Pose KD 導致共享層退步。
2. Pose head 的特徵適配／訓練統計：預測只調 head 的 native 對照可恢復部分 BBAT。之前 SiLU J3 有 head-only 小幅收益，但本次是另一個 qSiLU 起點，不能直接搬用舊結果。若也失敗，再看 head BN 或特徵瓶頸，不預設增加 epoch 有效。
3. 空間能量蒸餾偏向 bat 或大支持區域：預測 head 內更接近框／關鍵點任務的訊號，或後續有效物件區域加權，可改善 ball；目前只是與現有曲線相容的推測，不是已證明的偏差機制。

本次依 diagnosing-bugs 技能採既有全量曲線的 differential replay 定位症狀；使用者要求分析／規劃，故不進入新的 GPU 因果實驗或修復階段。

## 必要實驗，最多先兩臂

PH0：完整 Pose head 原生 BBAT5 訓練，不加 KD。

PH-KD：同一 qSiLU 起點、seed、資料順序、augmentation、epoch 數及 optimizer，Pose head 原生 loss＋**head 內**的單一蒸餾訊號。只需要原 BBAT5 Pose teacher，不載 YOLO26L。

兩臂都從起點重建，不從 PH0 的 best 再起跑 PH-KD。PH0 即使有收益，也保留 matched 對照才能判斷 KD 額外效果。若 PH-KD 真實梯度／安全前置不通過，不啟動空轉蒸餾；PH0 可獨立判讀。

**不能沿用現有 layer16／19／22 空間 KD。** 它位於固定 Neck，對 Pose head 參數 θpose 的偏導為0；有 loss 數字也不表示能教 head。PH-KD 擬先使用 Pose head 框／關鍵點分支的中間特徵蒸餾，保留 native 類別監督，不一次堆疊輸出 KL、box loss、QK ranking 等訊號。實作前需確認實際 Pose26 one-to-many／one-to-one 分支與 shape，選擇可對齊的 tap，確保部署使用分支有 native／KD 更新路徑。這些尚未實作或驗收，不直接複用 μP=1.7887。

## 訓練超參數提案

| 項目 | 下一輪設定 |
| --- | --- |
| optimizer | AdamW，新 optimizer state；不沿用 MuSGD 校準值 |
| LR | Pose head 1e-5；其餘0且不進 active 更新 |
| betas／weight decay | (0.948,0.999)／0.00027，BN與bias不衰減 |
| warmup | 1 epoch |
| 首輪預算 | 兩臂各5 epoch，patience0，完整跑完才作 matched 比較 |
| batch | physical16；先不加 accumulation，避免額外改變 head BN 統計 |
| scheduler／AMP／clip | cosine final0.5／AMP／10 |
| 原生 loss | 原 BBAT5 Pose26 loss，保留框、類別及關鍵點，不只訓 keypoints |
| KD 權重 | 在 BBAT5 train-only 校準，初始目標約 native head 梯度5%；查核後固定，非逐圖 scale |
| 資料 | canonical BBAT5 v1 train5964／val683，不重切、不增加過採樣策略 |

LR1e-5 是保守試驗起點，不是已證明最優值。保留原 trainer 的 loss 正規化，明記 effective batch／reference scale，不照搬 joint 的 Detect:Pose=1:0.25：純 Pose 無另一任務權重競爭，不把 Pose loss 無故縮成四分之一。

若5輪有明確收益且仍在改善，再考慮兩臂同預算延長至最多10輪、patience4；需記錄原5輪 cosine 已走完，延長是新適應階段，不冒稱預先規劃的10輪連續 schedule。無收益則不盲目加長或默默新增 Pose-only Neck／adapter。

## 驗證與採用門檻

1. 開跑前：CPU 重建 qSiLU、比對起點 hash／架構；非 Pose 所有 parameters／buffers 固定；KD-only 梯度確實到 head，teacher無梯度。真實 microbatch 更新後比較 live／EMA 的非 Pose state，前後 Detect 同圖輸出必須一致；必要時先修正 BN／EMA，而非放寬門檻。
2. 每輪：完整 BBAT5 val683，分開呈現 ball／bat 的 box AP、keypoint AP、precision／recall；另確認非 Pose state hash。AP 改善不能拿同一批 val 選好的圖片當獨立測試結果。
3. 完成：選定權重獨立重建、Float／Bit-True 完整 BBAT5＋COCO val5000，確認 teacher／training helper 不在 export；同口徑 COCO 應維持起點值，若非 Pose state 相同但 AP 不同先診斷 evaluator／graph／BN，不能歸因於正常訓練波動。
4. 暫定初篩：對起點及 PH0 分別比較；BBAT Box／Pose overall 均不退超過0.001、ball／bat各項不退超過0.001，至少一項原弱項提升0.002才值得下一步。仍另列最初獨立模型融合 gate，不掩蓋尚未補回的差距。單 seed 通過只代表候選，不宣稱穩定增益。
5. 固定同一批已存在的驗證影像 ID 與推論閾值，呈現漏檢、框位置、關鍵點偏移；不新增資料 split、不以新挑圖替代全量 AP。若希望放寬 ball／bat 取捨，需另明確指定，不只提升 bat 就判成功。

訓練停止条件：非 Pose state 改變／非有限 loss 或梯度立即安全停止；BBAT任一指標比起點下降超過0.02先保存再診斷。GPU 啟動後仍由 shell 600秒事件監測，正常不讀 log；此文件不代表新 queue 已啟動。

## 預期與限制

不再做 COCO backward，且共享 backbone 無需反傳，預期比原 joint 便宜；但仍須做 backbone forward、Pose teacher forward（KD臂）與驗證。Pose epoch 與 joint epoch 的樣本構成不同，必須記錄真實影像數、update數與 wall-time，不能僅用 epoch 快慢宣稱同算力收益。

凍結共享層能隔離 Detect，卻限制可學特徵：若真正瓶頸在融合後共享表示，head-only 可能無法追回。若兩臂失敗，下一步才在「接受共享層 Pose 專項微調導致 COCO 取捨」與「新增很小的 Pose 專屬 adapter、付出推論成本」之間提出新決策。本輪不提前擴張架構，不承諾增準。
