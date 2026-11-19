# 第 1 階段：Binary branches 初篩與恢復

2026-10-18 修訂。原 Q/K-only／`5e-7` 配方已移除，錯誤 run 與權重依使用者要求直接刪除，沒有封存。此文件只描述目前唯一的 v2 流程；舊結果與更正原因保留於工作紀錄及小型數據，不再作可執行配方。

## 動手前的 optimize 依據

事前來源、SHA、採用內容、差異理由與選型規則保存於 [reference-review-v2.json](../artifacts/stage1/reference-review-v2.json)。每個候選啟動前會核對這些來源的 SHA；來源變更須先重新確認適用性。

Optimize 的 [A0 恢復程式](../../yolo_optimize/experiments/studies/pre-fusion-full35-b100/scripts/recover_a0.py) 從已訓 A0 做 Q/K＋head 微調；初次二值化則依其上游 [H-SCR](../../yolo_attention/artifacts/runs/h-scr/training.yaml)、[W-DIR](../../yolo_attention/artifacts/runs/w-dir/training.yaml) 與 [scope 實作](../../yolo_attention/src/yolo_attention/integration.py)。既有 H-SCR internal AP50–95 0.495243，W-DIR 為 0.507457；這些是歷史結果，不是本輪預期保證。

本輪保留 optimize [qk_challenger](../../yolo_optimize/experiments/studies/pre-fusion-full35-b100/scripts/qk_challenger.py) 的精確 XNOR 前向與 signed-dot／clipped STE 反向，不引入舊 Q/K 梯度中斷問題。

## 共同 parent 與候選

- 原始 YOLO26m COCO80 Detect、E2E：`yolo_attention/weights/yolo26m.pt`；SHA-256 `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`。
- 兩處：`model.10.m.0.attn`、`model.22.m.0.1.attn`；每處 4 heads，key_dim 32、head_dim 64。
- B1：sign＋residual；B2：sign＋Hadamard；B3：sign＋residual＋Hadamard。B3 共用 sign，score 合併後只做一次 Softmax 與一次 P·V。
- Scale slots 為 16／16／24；bias none。以共同 parent 的原 128 張 train 影像校準固定 FP32 係數，三組新增分支均分 `1/bases_count`。本階段不學 scale；Q10／二次冪留待第 2 階段。
- 640 × 640 校準下兩處均為 20 × 20、400 tokens；驗證的 rect padding 可產生其他 H×W。token 不必是二次冪，Hadamard 只沿長度 32 的 key channel。
- COCO train2017 全 118,287、val2017 全 5,000，不變更清單／split／labels。BBAT5 v1、Pose、未指定階段及 optimize 不修改。

## 執行順序及預算

| 步驟 | 起點 | 可更新內容 | 預算 |
| --- | --- | --- | --- |
| B1／B2／B3 初篩 | 三組分別從同一原始 parent 初始化 | 兩處 Attention 全部權重；全模型 BN statistics | 各 10 epochs，無 early stop |
| 分支選型 | 三份初篩結果 | 不訓練 | 以相同完整官方 COCO bbox AP50–95 選定一組 |
| 選定分支恢復 | 該分支初篩 best | 全模型權重及 BN statistics，保留 native 永久凍結的 DFL（若有） | 最多 40 epochs，patience 5 |
| 第 1 階段交付 | 恢復 best，可為 epoch 0 | 不訓練 | 產生報告／handoff，停止等使用者檢視 |

最多 30 個初篩 epochs＋40 個恢復 epochs。不同分支不相互續訓。只對初篩 winner 做全模型恢復，沿用舊研究的先篩後恢復順序；未選候選未做完整恢復，不能宣稱其完整恢復後仍較差。舊 B1 權重已刪除，三組不繼承其額外 10 epochs。

各次正式訓練都先完整評估 epoch 0，再逐 epoch 評估 EMA，best 包含 epoch 0。同分保留較早 epoch；跨分支同分優先較少 bases，再依 B1／B2／B3 順序。若最高初篩官方 AP 仍低於本輪實測 parent `0.5246855327 × 90%`，停止並記錄診斷需求，不自動用 40 epochs 掩蓋問題。

## 共同訓練設定與差異

AdamW、LR `5e-5`、betas `(0.937,0.999)`、Conv weight decay `0.0005`，bias／BN affine 不 decay；physical batch 16、logical batch 64，固定累積 4 microbatches。保留 native E2ELoss 的 batch-sum，不再除以累積次數，尾批保留。每個 phase 使用自己的 criterion horizon：10／40，不能將恢復沿用 10 的 loss 排程。

3 epochs warmup，linear LR 至 0.01 倍，gradient clip 10。所有 group 都由 0 warmup；與舊 native bias 起始 LR 0.1 不同，避免在新增 Q/K 梯度路徑下使用相對於 `5e-5` 高 2,000 倍的起始 bias LR。固定 macro 累積也與舊 native warmup 動態累積不同，已明列，不宣稱逐步完全復刻。

AMP FP16 訓練，參數、binary score／Softmax 及評估維持 FP32。Loss scale 初始 1024；遇到 AMP gradient overflow，參照 optimize 還原 BN buffers、降低 scale 並重算同一 macro，最多 16 次，不跳過資料。非有限 loss、未恢復梯度、Q/K 零梯度、固定 tensor 漂移立即停止。

初篩採 Attention 權重更新及全模型 BN statistics，符合上游原生 trainer 的 BN 適應行為；Attention 以外的 affine 權重固定。EMA 同時跟進 trainable tensors 與 BN buffers，其他 tensor 逐位不變，避免凍結參數被平均捨入改動。恢復重新建立 optimizer、EMA 與 criterion。

Mosaic／mixup／cutmix／copy-paste 關閉，其餘原生 augmentation 保留。這沿用 optimize 恢復做法；舊 10 epoch screening 的 close_mosaic=10 亦全程無 mosaic，但舊 W-DIR 的前段 mosaic 與本輪不同。Seed 0，各候選同 epoch seed、完整資料與評估器。

與舊 H-SCR/W-DIR 的 dynamic scale、可學 gamma 不同，本輪固定係數以維持階段邊界。[128 張真實特徵診斷](../artifacts/stage1/score-diagnostics-v2.json) 將 fixed 與逐影像幅度參考分開量測 score、P、P·V，兩者 P·V 誤差相近，無非有限值。這只支持固定平均在此觀測下沒有額外明顯惡化，不能證明任務 AP 或訓練必定恢復。

## 評估、續訓與監測

E2E、FP32、640、batch 8、conf 0.001、iou 0.7、max_det 300，完整 val 5,000。輸出 internal 與官方 COCO bbox；overall／person／sports ball／baseball bat 各 AP50、AP50–95，共 8 項主要 AP。原始 parent 已用此 evaluator 重驗為 internal 0.518002／官方 0.524686，不新增 baseline 加訓。

Checkpoint v2 記錄 root parent、immediate parent 路徑與 SHA、phase、架構、二值／scale／normalization／activation 契約。Resume 另含 live、EMA、optimizer、AMP scaler、epoch、best 與 history，核對 SHA、recipe、manifest；只支援完整 epoch 邊界。中斷的 epoch 從上一完整 epoch 重跑。

Supervisor 每 600 秒寫入 GPU 與進度，5 秒輪詢程序完成。GPU 忙碌等待後重查，不終止他人工作；30 分鐘無進度則終止自己的 child。`STOP` 在 batch 邊界停止，評估先完成當次工作；`--continue-queue` 嚴格核對已完成結果後略過，未完成者由 trainer 核對 resume。File lock 防雙重排程。背景監測落盤，不宣稱聊天自動推播。

正式隊列最後保存 `comparison.json/csv`、`curves.svg`、`RESULTS.md`、`handoff.json`，並停止，不進入第 2 階段。以 `python -m binaryqk.report --plot <隊列目錄>` 可僅讀小型 JSON 重畫，不需 GPU 或 checkpoint。

## 硬體與清理邊界

本階段 binary dot 是 XNOR/popcount 功能 reference，尚非 packed kernel。Residual 每 token 仍有 abs、32 項累加、`/32`、乘減；Hadamard 沿 32 channels 做 5 層加減。兩處 binary score 成本為 `2 × heads × bases × N² × D`，P·V 只做一次；N 依實際 H×W。沒有 RTL／HLS 或板端測量，不能宣稱加速。

固定 FP32 scale、Softmax、SiLU、QKV／P·V 的後續硬體友善化仍依整合計畫；RepConv 最終必須同時繼承 qSiLU 與完整硬體契約。

本次錯誤 v1 產物直接刪除清單見 [cleanup-invalid-v1.json](../artifacts/stage1/cleanup-invalid-v1.json)，沒有封存。既有 26 份唯一正式權重與 32 份尚未完成替代的舊唯一 checkpoint 不屬於這批錯誤 run，維持原保護；optimize 外部依賴 CSV 與實際載入中的 `yolo_p2/ultralytics` 不能刪除。
