# 2026-10-19：第 2 階段 Q10／二次冪 scale 與 bias

最新接續：兩組已於 2026-10-19 04:58 完成，推薦 S1，官方 AP50–95 0.514152；第 3 階段未啟動。見 [完成及交付紀錄](2026-10-19-binaryqk-stage2-results.md)。下文保留實作、啟動及初期 ETA 的歷史過程；「尚待結果」已由完成紀錄接續。

## 授權與範圍

使用者確認 Q/K 二值化仍保留，並明確指定進入第 2 階段。繼承第 1 階段 B2 最佳權重；S1／S2 分別從共同 parent 開始，完成後停止，不自動開始第 3 階段。沿用 YOLO／yolo-training 技能，以本機版本與本機來源為準，沒有雲端上傳。

`yolo_binaryqk` 已在第 1 階段完成首次清理重建，不再次清空本輪有效程式與結果。`yolo_attention` 本次只維護共用尺度／bias 算子，既有 optimize 依賴不刪除；完整 Attention 遷移清理留待第 3 階段。

## 動手前的來源核對與決策

已唯讀檢查 optimize `attention_recovery_v1/modeling.py` 的 `StaticDyadicScore`、`post_masf_hardware_v1` 的 models／config／run_arm／正式 summary，以及上游 EMA、macro loss 和 cosine 排程。來源正式 scale_bias E5 internal AP 為 0.505248，其原工程門檻未通過；不把歷史結果當本輪保證。完整來源 SHA、採用項目與差異先寫入 [reference-review.json](../../yolo_binaryqk/artifacts/stage2/reference-review.json)，再開始實作。

兩組各 5 epochs；physical batch 16、logical batch 256、reference batch 64。採 Attention／Detect head 適應、固定所有 BN statistics、分組 LR（attention 1e-5、head 5e-6、scale 2e-4、bias 1e-5）、AdamW、1 epoch warmup 後 cosine。共同 parent 沒有 bias，因此兩組均以零表初始化；保持 native Softmax、SiLU 與 B2 sign＋Hadamard。S2 的量化前向為專案延伸，不能宣稱 optimize 已比較驗證。

## 實作與整數前向

`yolo_attention` 新增 `FixedScaleFormat`／`ScoreAffineConfig`／`QuantizedScoreAffine`，在共同 Q10 grid 計算 score。S1 用整數乘法，S2 用明確 left shift；兩者共用 query-minus-key 水平／垂直 signed16 bias，不隨 basis 增加表。每處 H／W 上限 32，最大 N=1024，兩 basis score 累加需 signed18，reference 使用 int32。推論快取部署常數；訓練前向保持整數結果，反向使用記錄於 manifest 的浮點代理。

`binaryqk` 沿用同一 Attention、訓練及背景監測 engine，增加 Stage 2 policy、嚴格 manifest／checkpoint loader、正式比較與無 GPU 重畫入口。manifest 保存實際 learned codes、bias、shape、位寬、rounding、activation call-sites 及 parent SHA。載入缺設定、候選混用或回退 normalization 均失敗；舊 Stage 1 loader 不接受新 checkpoint。

## result 交接與清理

依使用者本輪新增指示，換階段時由 `result/` 交接，下一階段在自己的目標資料夾工作。已確認 Stage 1 和 Stage 2 smoke 無活動程序，將選定的 `artifacts/stage1/formal-20261018-v2/recovery/` 實體搬到 [result/stage1](../../yolo_binaryqk/result/stage1/README.md)。14 個原始檔案搬移前後 SHA 全部相同，沒有複製大型檔案；舊入口僅保留指向同一實體的相容連結，避免歷史報告失效。

第 2 階段 reference-review 的 parent 已改為新路徑，checkpoint SHA 不變，十個 optimize／上游來源 SHA 重新核對通過。正式 run 從新路徑載入；smoke 保存執行當時的 review SHA，不改寫歷史證據、不續跑 smoke。搬移證據見 [stage1-relocation.json](../../yolo_binaryqk/result/stage1-relocation.json)。本次沒有再清空有效實作或未指定資料夾；沒有 archive／backup。

## 驗證結果

- Binary Q/K 全套 50 項通過；Attention 全套 168 項通過、1 項既有 GPU BDCN 測試略過。新增算子測試涵蓋兩種 scale、全部 exponent、負 counts、bias 方向、飽和、ties-to-even、最大 token／signed18 邊界、梯度及非法 shape。
- Ruff 通過 Binary Q/K 程式／測試與 Attention src／tests／scripts；兩套 CLI help 通過。
- S1／S2 GPU smoke 均完成，每組 512 張、兩個 optimizer steps、16 張非正式評估。十項 Q/K／scale／bias 參數均有實際更新，AMP retries 均 0，固定 BN／EMA、checkpoint 與 resume 驗證通過；各組 peak allocated memory 為 6240560128 bytes。
- 兩組首批影像一致；已視覺檢查 `S2/train-targets.jpg`，bbox 對齊影像。短測 AP 不能代表完整 COCO，result 的未計算官方 AP placeholder 不作研究結果。
- 搬到 result 後重跑受影響的 6 項 Stage 2 測試，全部通過（28.44 秒）；未因單純搬移重跑 GPU 短測。

## 正式背景工作

2026-10-19 00:35:26（Asia/Taipei）已啟動 [formal-20261019-v1](../../yolo_binaryqk/artifacts/stage2/formal-20261019-v1/README.md)，supervisor PID 740046，S1 child PID 740053。事前 GPU 無其他 compute process；目前使用 RTX 5090 device 0。每 600 秒監測，完成或程序失敗由 supervisor 接續處理；兩組完成後產生本階段報告並停止，不開始第 3 階段。

S1 已完成 epoch 0 的完整 val5000：官方 AP50–95 **0.509822**、internal **0.503239**，與 B2 parent 官方 0.509889 接近，通過事前門檻。這是 Q10／零 bias 的初始替換評估，尚非五輪訓練結果。正式 epoch 1 已有 optimizer 更新，十項必要梯度皆有限且非零；loss factor 0.25、warmup 分組 LR、AMP scaler 正常。

目前 GPU 程序約占 7080 MiB。長訓穩定後依使用者要求退出 working；由背景程式繼續監測，不留對話反覆等待。預估完成時間於取得正式吞吐量後記錄在 [啟動紀錄](../../yolo_binaryqk/artifacts/stage2/formal-launch.json)。

00:38 的正式進度已達 5120 張／20 steps，訓練用時約 70.5 秒、AMP retries 仍為 0，十項必要梯度正常。依初期完整資料吞吐量估算，兩組共十輪含評估約於 **2026-10-19 05:00～06:30（Asia/Taipei）** 完成；這是估計，其他 GPU 工作、I/O 或中斷會延長。背景 PID／狀態與實際結果以 run 檔案為準。驗證摘要見 [verification.json](../../yolo_binaryqk/artifacts/stage2/verification.json)；本次新增／修改的十份入口文件已檢查本機連結，沒有失效連結。

## 困難與解法

初次測試有一項無法 import 上游 cosine scheduler，原因是測試未加入 `yolo_combine/src`；已在該測試用 monkeypatch 增加來源路徑，再核對所有排程邊界一致。初次 Ruff 的 import 順序與 list 累加建議已修正。來源查找中的路徑差異均以本機實際檔案及 SHA 解決。搬移及 GPU 短測沒有其他異常。

## 未解事項與限制

正式 S1／S2 五輪結果、winner、最終部署常數與選定 parent 尚待完成，不預設 S2 勝出。只有一個 seed，未新增 control 加訓；本輪仍保留 native Softmax、SiLU、浮點 QKV／P·V／projection，未宣稱整網定點或實測硬體加速。共用算子必要新能力已加入，完整 Attention 清理／整併留待第 3 階段指定。第 2 階段完成並檢視後，切換階段前再把選定候選整理為 `result/stage2/`。
