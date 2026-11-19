# 2026-11-12：768×512 同配置訓練

## 任務與授權

使用者指定再試 **768×512**，並要求「使用一樣的訓練配置」及「train」。沿用本研究 H×W 約定，新增高 768、寬 512 的直向候選；不旋轉原始影像。此為 Stage9 補充單一候選，新增最多 **10E＋半 LR 10E＝20E**，連同已完成 100E，研究總上限 120E。完成後停止，不啟動其他階段。

## 事前 optimize 核對與相同配置

已先唯讀查閱 optimize 的 `experiments/pose_only_rep_v1/config.json`、`run_arm.py`、`RESULTS.md`，以及 activation 的 `bridge_v1/train.py`、qSiLU 10E 實際 `resolved-config.json`／`summary.json`、融合 `j3_stage.py` 與 `calibrate_task_weight.py`。前者 Rep17＋20 E5 的 COCO／person／BBAT box／Pose AP 為 50.5869／62.6986／60.9797／89.1041%，使用含 Pose-MASF 的另一血緣，只作方法參考。完整來源 SHA 與採用理由已在實作前保存於 [reference-review.json](../../yolo_actshape/portrait/reference-review.json)。

另外直接核對前次 512×768 的 adapt10／low-lr10 `config.json`。共同 parent SHA 保持 `2454548104ed64768eb9ecdb99d44804ee9d8e1e371e2a508cf052c0911fdd35`，使用原 qSiLU 訓練版 folded Rep17＋20 E1，沒有從已適應橫向的候選接續。

- 原 seed 20261003、AdamW betas=(0.948, 0.999)、weight decay=0.00027。
- 第一段 neck LR=1.9e-6，Detect／Pose head LR=5e-6；各段 10E，第二段從第一段 best 出發，全部 LR 減半。
- 各段 fresh optimizer／EMA，warmup 1E、cosine 終值係數 0.5；AMP、FP32 CIoU、clip norm=10、overflow retries 上限 16。
- 只訓練 neck 與雙 head，固定 backbone、Attention 與 BN running statistics。qSiLU／D1 整數 PWL exp＋LUT256 reciprocal／B2／S1／folded17+20 保持。
- Detect physical batch=8、累積 32；Pose physical batch=8、每 macro 2 批；reference batch=64。每輪完整 COCO train 118287 張、463 macro；Pose 沿用既有完整資料與循環規則。
- 相同 HSV／合法水平翻轉與等比例 LetterBox pad114，COCO／BBAT5 v1 split 與標註保持。形狀是 H×W，不透過只取 max 的 imgsz 介面。
- 每段沿用同一 train-only 梯度校準：3 批量測、2 批確認、最多兩次，target=0.5、band=[0.2, 0.8]。橫向的實際 weight 0.03／0.09 是量測結果，不硬套給不同幾何。
- 校準若明確 needs_review 且 parent 已還原，才按既有 11 月 10 日有界恢復政策補做一次 16 批量測／8 批確認，維持 target／band／量化步長；不以 validation 挑 weight，不增加 epoch。其他錯誤直接停止供診斷。

固定配置見 [protocol.json](../../yolo_actshape/portrait/protocol.json)。初輪與擴充來源鎖不改寫，舊 100E 結果及正式交接保持。

## 實作與驗證

新增 `stage9_portrait/` 作獨立接線與交付，重用既有 Stage9 的模型、訓練、loss、optimizer、scheduler、完整 evaluator 及 Stage9 extension supervisor。作用範圍限定在獨立 worker，離開 context 時還原；共用 Attention 沒有新副本。

CPU preflight 已通過：3 項排程／還原／拒絕不當恢復檢查，實際完整模型 768×512 前向、形狀 cache 還原、D1 N=384、正式資料 membership 與兩任務真實標註反算。P3／P4／P5 分別為 **96×64、48×32、24×16**，兩個 Attention sites 均 384 tokens，符合既有軸與定點範圍；驗證未使用 GPU。結果見 [preflight.json](../../yolo_actshape/artifacts/shape-portrait-20261112-v1/preflight.json)。

queue 依序完整 E0 → 42 項 GPU replay → 校準 → 2 macro smoke → 10E → 新 parent 校準 → 半 LR 10E → 最佳完整重載 → 獨立 CPU 交接。沿用每輪完整 COCO 5000／BBAT5 v1 683 張 evaluator。E0 的尺寸掉點不直接排除；訓練仍保留相對自身 parent 的災難停止門檻與原十項 AP 選型。

## 資源、啟動與預算

單張 RTX 5090；事前查詢無外部 CUDA 計算程序，依前次預估需要約 4～7 GiB、10～12 小時，啟動後由 smoke 與實際 epoch 速度修正。背景 supervisor 每 600 秒紀錄進度、每 30 秒檢查外部 GPU 工作；有衝突只暫停本研究 worker，保留完整 epoch，不操作外部工作。SIGSEGV 仍只允許已有完整 epoch 時的一次有界恢復。

啟動／查詢：`yolo_actshape/run_portrait.sh run`／`yolo_actshape/run_portrait.sh status`。run 為 `yolo_actshape/artifacts/shape-portrait-20261112-v1/`，停止檔為其中 `STOP`。整體進度仍由 `yolo_actshape/queue-status.json` 提供。新增成果放在 `result/stage9-portrait/`，完成後保留候選最佳並直接清理無用續訓檔；不覆蓋原 `handoff`。

## 困難與未解事項

目前無實作或 CPU 驗證失敗。直向與橫向雖像素數相同，但對以橫向影像為主的資料可能改變有效解析度與補邊，精度必須由訓練與正式 AP 決定。尚未取得本候選正式結果，不預設更好。過去 CUDA 原生崩潰根因未定，本次未宣稱驅動已修復。硬體目標未指定；MAC、GPU replay／時間不等於 RTL／板端效能。

## GPU queue 已啟動

2026-11-12 14:02 啟動，supervisor PID 186783；起始 GPU worker 186788 正在完整 E0，約 4.34 GiB 顯存，無外部 CUDA 計算程序。已逐檔核對 466 個來源 SHA、兩方向相同 training recipe，以及 21 個文件連結。

CPU 實際雙任務 Conv＋P.V 為 45.508165632 GMAC，與 512×768 相同；COCO 平均補邊比例由 19.23% 變為 41.43%，BBAT5 由 21.57% 變為 52.63%。框／可見關鍵點最大反算誤差小於 0.000062 pixel。這些是幾何與成本診斷，不是新候選 AP。

## 正式訓練確認與離開 working

14:06 查核已進入 adapt10 第 1 輪、20／463 macro，EMA updates=20；三組梯度均存在，AMP overflow retries=0。E0、42 項 GPU replay、原規則校準與 smoke 已通過；校準 Pose weight=0.08，沒有使用補充恢復。smoke 實際更新 415 個參數，peak reserved=4.439 GiB；正訓程序顯存約 5514 MiB。

初期扣除啟動後每 macro 約 3.693 秒，完整 E0 評估耗時 86.974 秒。依此估算完成時間約 2026-11-13 00:09，對外給 00:00～01:00 的初步區間；尚未完成第一個完整 epoch，資源讓出或中斷可能延後。supervisor 186783 與 worker 187026 正常，最近 monitor 約 97 秒前，符合每 600 秒規則。已穩定進入長訓，本次退出 working，由背景程序接續。機器可讀查核見 `training-start-verified.json`。

## 背景 queue 完成

2026-11-13T00:07:28.395452+08:00：20／20E 完成，採用門檻未通過，完整結果位於 `yolo_actshape/result/stage9-portrait/`。重載與可攜驗證通過；現行主線保持，無新階段。清理清單見該目錄 cleanup.json。

## 2026-11-17 完成核驗

已核對兩段各 10E、最佳 E5、來源／交接 SHA 及實際清理。十項 AP 均低於 512×768 最佳；原正式主線保持，本研究已停止。完整結果與驗證見 [完成紀錄](2026-11-17-actshape-portrait-results.md)。
