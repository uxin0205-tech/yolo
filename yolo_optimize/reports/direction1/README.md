# 方向 1：融合前研究結果與驗收

後續狀態覆蓋（2026-09-11）：使用者最後選 P3 bridge 接續，combine／activation／KD／推論均已實際執行；下方「未開始」與無 MASF 暫定接續只屬當時歷史。完整最終範圍與封存見[全階段總報告](<../final/README.md>)，不修改本報告原始測量。

> 2026-09-10 更新：最後 P2 MASF 配對已完成，各 5 epochs，全部未達增準門檻；最新數據見[MASF 結果稽核](<../../../docs/worklogs/2026-09-10-masf-results-audit.md>)。本報告下方保留追加 P2 前的歷史結果；原 P2 設計與 [-10,0] PWL 確認見[工作紀錄](<../../docs/worklogs/2026-09-10-p2-masf-last-trial.md>)。

## 結論與狀態

本輪排定的融合前 BinaryQK 梯度恢復、後段適應、HOG、單點 RepConv、MASF 位置與訓練梯度實驗均已完成或依既定門檻停止。兩個最終比較候選已完成獨立匯出、完整 COCO 5000 重驗與固定 8 張圖片推論。

**實驗階段完成，不等於精度目標完全達成。** 相對原 B100 有部分回升，但尚未回到 FP；MASF／HOG／RepConv 的額外增益未達門檻，沒有正式升格新的方法 winner。融合、activation 與方向 2 均未開始；整體長期目標仍未完成。

本資料夾集中結論與來源入口，不搬移或複製大型權重。原始 checkpoint、失敗事件、對照與融合後研究全部保留。檔案僅供本機研究，未提交或上傳。

## 最終數值

同口徑 COCO val internal AP50–95，0–1 尺度；兩候選都是完成 E6–E10 中 overall 最佳的 E8 EMA。

| 模型 | overall | person | sports ball | baseball bat |
| --- | ---: | ---: | ---: | ---: |
| 原 FP | 0.518019276 | 0.630794912 | 0.526040758 | 0.483490044 |
| 原 BinaryQK A0 | 0.506738574 | 0.626805274 | 0.513110259 | 0.495725734 |
| 原 B100 Bit-True | 0.503589001 | 0.624111237 | 0.516234472 | 0.469482418 |
| 無 MASF 對照 E8 | 0.508267092 | 0.627699455 | 0.514005652 | 0.480127550 |
| MASF 梯度橋接 E8 | 0.508211955 | 0.627664127 | 0.513148090 | 0.480664186 |

無 MASF 對照相對 B100：overall +0.004678091、person +0.003588218；MASF 候選相對 B100：overall +0.004622954、person +0.003552890。這包含不同訓練與起點路徑，不能全部歸功於 MASF、QK 或單一變更。

MASF 候選相對同 E8 無 MASF 對照：overall -0.000055137、person -0.000035328，差距不足以宣稱實質優劣。兩者對 FP 的 overall 缺口仍約 0.00975–0.00981。只跑一個配對資料序列，沒有多 seed 顯著性或獨立 test 證據。

## 各方法的判斷

| 方法 | 實際發現 | 本輪處理 |
| --- | --- | --- |
| BinaryQK | 實際 bool／整數 score 前向切斷 Q/K 投影梯度；訓練 surrogate 可保持精確二值前向並接回梯度 | 已驗證連通與分段訓練；不能宣稱整個 FP 缺口已補回 |
| 後段適應 | late E8 優於同回合 narrow，但增量不達 +0.001 門檻 | 作探索 parent，與正式方法 winner 區分 |
| HOG | E2–E4 overall／person 不如同回合原生對照；小球有 P3 cell-center 監督覆蓋盲區 | 依 patience 4 停止，不加入候選 |
| RepConv layer17 | 初始化與部署折疊驗證通過；E4 有極小增量、E5 回落 | 不扩到 layer20，不加入候選 |
| MASF shared／fork | shared 會改到 P4/P5；fork 可隔離 raw 特徵與對應 raw logits，但沒有足夠 AP 增量 | 不將移位本身稱為增準 |
| MASF head 適應 | head LR 從 2e-6 提高至 1e-5，續訓 E6–E10 仍無穩定額外收益 | 不無限延長相同配方 |
| MASF one-to-one 梯度橋接 | 直接監督可接通，真實 loss 校準中 3/8 batch 梯度方向相反；小係數橋接後 person 各 E6–E10 均略高於 native fork | 收益小於門檻，只保留研究候選 |

完整配對依據見 [MASF E6–E10 橋接結果](<../../experiments/studies/pre-fusion-full35-b100/artifacts/masf-task-monitor-recovery-v1.json>)、[HOG 結果](<../../docs/worklogs/2026-09-09-hog-result-ball-bat.md>)、[RepConv 結果](<../../docs/worklogs/2026-09-09-prefusion-rep17-result-masf-off.md>)、[QK 梯度恢復](<../../docs/worklogs/2026-09-09-attention-gradient-recovery.md>)。

## 架構：原本與候選

原 B100：

```text
layer16：p3_raw → MASF → p3_shared ─┬→ layer17 → P4 → layer20 → P5
                                   └─────────────────────────────┐
layer19：p4 ─────────────────────────────────────────────────────┤
layer22：p5 ─────────────────────────────────────────────────────┤
                                         Detect([P3, P4, P5]) ←─┘
```

本輪 MASF 候選推論：

```text
layer16：p3_raw ────────┬→ layer17 → P4 → layer20 → P5
                       └→ MASF → p3_det ─┐
layer19：p4_raw ──────────────────────────┤
layer22：p5_raw ──────────────────────────┤
                                         ↓
                         Detect([p3_det, p4_raw, p5_raw])
```

新增的訓練路徑為 `detach(p3_raw) → 同一 MASF → one-to-one loss`，只將回傳 MASF 的梯度乘固定 0.012076444778011642；不回傳 Backbone。推論仍只計算一次 MASF，不增加額外 context 分支。

## Softmax／硬體說明

目前 attention 的 Softmax 是 PWL：訓練 Float-PWL，驗證／匯出 Bit-True PWL。PWL knots／values／endpoint table 是固定 buffer，不是本輪微調的參數。這輪更新 P3 head／MASF，不更新 PWL 表格、分段或 attention bias／score。

Bit-True PWL 近似 exp 的 lookup／插值／飽和；最後 normalization 仍有 exact software reciprocal reference。不能把它稱為完整無除法整數 Softmax，更不能在未量測前宣稱 FPGA latency 或能耗改善。

## 超參數與驗證

最新 MASF 候選：AdamW betas=(0.948, 0.999)、eps=1e-8、weight decay=0.00027、clip norm=10；P3 head LR=1e-5、context=1e-5、alpha=1e-4，原 10-epoch cosine。初始 warmup 1 epoch，E5 續訓不重新 warmup；保留 optimizer、EMA 與 scaler。physical batch32 × accumulation4 = logical128，imgsz640，完整 train118287，每 epoch925次更新；BN統計全固定。詳見 [校準與訓練設定](<../../docs/worklogs/2026-09-10-masf-task-bridge-training.md>)。

兩候選各完成 COCO5000 的 Bit-True 匯出重驗，overall／person／ball／bat AP 差值皆為 0；CPU160 重載前向相同。每個候選另完成相同8張含 person 圖片推論（all80、conf0.25），不是 person-only 訓練。這8張取自 validation 固定清單，不是獨立 test。

完整來源、SHA256與驗證：[machine-readable summary](<../../experiments/studies/pre-fusion-full35-b100/artifacts/direction1-candidate-verification-v1/summary.json>)。

- 無 MASF 推論權重（本機／歷史參照：`../../experiments/studies/pre-fusion-full35-b100/artifacts/direction1-candidate-verification-v1/control-e8-bittrue.pt`；未隨本次報告發布）
- 保留 MASF 推論權重（本機／歷史參照：`../../experiments/studies/pre-fusion-full35-b100/artifacts/direction1-candidate-verification-v1/masf-e8-bittrue.pt`；未隨本次報告發布）
- 無 MASF 原始訓練快照（本機／歷史參照：`../../experiments/studies/pre-fusion-full35-b100/artifacts/masf-head-control-v1/epoch-08-resume.pt`；未隨本次報告發布）
- MASF 原始訓練快照（本機／歷史參照：`../../experiments/studies/pre-fusion-full35-b100/artifacts/masf-task-bridge-v1/epoch-08-resume.pt`；未隨本次報告發布）
- 無 MASF 8 張推論輸出（本機／歷史參照：`../../experiments/studies/pre-fusion-full35-b100/artifacts/inference/direction1-control-e8-first8`；未隨本次報告發布）
- MASF 8 張推論輸出（本機／歷史參照：`../../experiments/studies/pre-fusion-full35-b100/artifacts/inference/direction1-masf-e8-first8`；未隨本次報告發布）

執行推論須保留 study scripts 與來源模組，不能只搬走自訂類別 `.pt`。例如在專案根層執行：

```bash
/home/uxin/yolo/yolo_combine/.venv/bin/python studies/pre-fusion-full35-b100/scripts/infer.py \
  --weights studies/pre-fusion-full35-b100/artifacts/direction1-candidate-verification-v1/masf-e8-bittrue.pt \
  --source /path/to/image-or-video \
  --name new-user-inference
```

## 保留與未解事項

清理盤點：訓練 checkpoint、logs、失敗啟動事件、runtime資料View與所有對照均為可追溯依據，全部 Keep；沒有刪除候選或已刪除項目。未 commit／push。

圖片讀取工具遭 sandbox `bwrap` 失敗，尚未目視驗收，不宣稱視覺效果已改善。使用者實際影片、獨立 test、多 seed、硬體 latency／energy 尚未驗證。

下一步進入 combine 前需要確定是否保留 MASF。無 MASF 的 AP 略高但差異很小；保留 MASF 符合原設計方向，卻尚無足夠額外增益。兩者都保留，不擅自把設計元件移除或把候選升格。後續順序維持 combine → activation（QSILU 優先，其餘評估）→ 方向2，但尚未啟動。
# 2026-09-10 接續決策

P2 MASF 最後 5 epoch 配對已完成，未達門檻，融合主線不採用 MASF。選 control E2 EMA，完整 COCO overall 0.5083300413／person 0.6278584228。來源、完整 P2 表格與新共享 Pose 初始退化分析見[最新工作紀錄](<../../docs/worklogs/2026-09-10-p2-result-no-masf-fusion.md>)。無 MASF J0 已啟動；新增加的 Pose MASF 專項尚未訓練，不能由 COCO 結果直接判定其 Pose 效果。下方保留先前各階段紀錄。
