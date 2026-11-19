# Stage9 新增硬體尺寸與 GPU 分塊驗證

日期：2026-11-08。使用者明確授權：「你都可以一起做做看驗證看看，在 GPU 上試，都可以的。」沿用本階段自主 queue 與長訓穩定後退出 working 的要求。

## 變更內容與原因

在 `yolo_actshape/stage9_extension/` 新增獨立擴充 controller，將 256×512、384×512、512×768 各安排完整 E0、梯度校準、真實 macro smoke、10E 適應及半 LR 10E。額外上限 60E，加上初輪 40E，整體上限 100E。三組共同 parent 仍為使用者選定的 qSiLU 訓練版 Rep17＋20 E1，SHA `2454548104ed64768eb9ecdb99d44804ee9d8e1e371e2a508cf052c0911fdd35`。

新增 `extension/protocol.json` 與 `reference-review.json`，先記錄 optimize 核對及新方法差異，再實作。主要來源為 optimize 的 pose_only_rep_v1 config／run_arm／RESULTS、activation bridge 的訓練程式與 qsilu_pq-short-e10-seed1-v1 實際配置、結果，以及兩個 profiling 模組；完整 SHA 見 reference-review。沿用固定 BN、AdamW、完整 macro／loss scaling、warmup1／cosine .5、fresh EMA、FP32 CIoU、neck／雙 head 適應；不同尺寸與 tile replay 是本專案延伸，不是 optimize 已驗證方案。

部署保持 qSiLU、D1 整數 PWL20＋LUT256 reciprocal、B2 二值 QK、S1 Q10／signed16 bias 與 folded Rep17＋20。正式資料仍為原 COCO 與不可變 BBAT5 v1，完整 COCO val 5000／BBAT val 683；不重切、不改標註、不新增原尺寸 control 加訓。

七組 canvas 做 tile 8／16／32、row pitch 還原、Rep17／20 的實際 halo／stride／padding／qSiLU，以及 query 16／32／64 分塊的 GPU replay。query 分塊每列保留完整 keys，D1 probability 要逐位相同，P·V 與卷積使用明確誤差門檻；不補入未遮罩 token、不用浮點除法強制重新正規化。完整 QK 分數矩陣仍物化。replay 不替換正式任務模型，完整 AP 另跑。

計時為 FP32 batch1、both routing、warmup20／100 次同步 PyTorch 推論；單任務一張的 replay 只驗算子，不能作 AP。分析中的 tile 補齊容量、Conv＋PV MAC 與 GPU 時間分開，沒有 RTL／HLS 或板端實測。

## 排程、監測與交接

原程序與原 `source.json` 不改。原來源 SHA 為 `eaaadf0d076f58086e5da1e13ec767cbbbbc3180f47f23dafe0fe4204e01efcb`；擴充 controller 必須等原狀態 complete 且取得原 queue 的 exclusive flock 才啟動 GPU。單獨完成標記不代表鎖已釋放，避免 finalization／清理與新工作競爭。

擴充 worker 在獨立程序重用原訓練引擎，僅於明確的 context manager 內綁定擴充輸出路徑與來源驗證，結束或例外均還原。每 600 秒記錄 monitor，每 30 秒查資源衝突；只停止自己的 worker，完整 epoch／配置／來源 SHA 相符才能恢復。SIGSEGV 最多重試一次，不無限加訓。初輪失敗或失去持鎖程序則停止擴充並回報，不跳過未完成工作。

全部完成後合併初輪與擴充 AP、成本與 GPU replay；保留三組新尺寸最佳作硬體取捨證據，清掉不再使用的 optimizer resume／被取代權重。初輪小型結果保留為 lineage，不複製整套 archive。交接包先建立並通過獨立 CPU 重載再取代舊包；Stage8 最終模型保持。Stage9 全部結束後停止。

## 驗證方式與結果

CPU 10 項測試通過：矩形尾塊／NaN pitch padding、卷積 halo／stride2／dilation／groups／qSiLU、D1 非二次冪 keys 與 query 尾塊、真實 Detect／Pose 樣本 hooks、引擎綁定例外還原、初輪鎖競爭與孤兒程序、resume SHA／完整 epoch、逐 AP 選型門檻、100E 預算與初輪來源不變。

實際 CPU 小尺寸 64×96 完成 42 項 replay，模型 state 未改、D1 probability 逐位相同，無 CUDA context。三個正式候選尺寸的完整 CPU 前向與位寬查核沿用前次 `analysis/hardware-shapes-cpu.json`。這些不是新增 GPU AP 或 GPU 等價性結果；相關 GPU 工作由擴充 queue 接續執行。

擴充來源鎖含 450 項，測試時 SHA `d8212a644a473d3bdd0e9d1f15cd38c14e8601d48d3b62dfd379f1a8f3da168b`。語法檢查通過。原 GPU 仍只有既有訓練 worker；第一組 384×768 在第 6 輪，尚未啟動新增 GPU worker。

## 困難與解法

既有長訓已鎖定程式，不能直接把新候選加入原 protocol；以獨立來源鎖、固定訓練引擎介面與相同 flock 串接解決。原 controller 會寫「初輪完成」文件，擴充在取得鎖後同步改為「初輪完成、擴充接續」，最終再統一交付，避免誤報整階段已結束。

未指定目標硬體，無法把單一二次冪形狀認定為最優。此輪先量測軟體、容量與 tiling，保留精度／成本取捨。其他困難：無。

## 未解事項與時間估算

新增尺寸訓練速度、完整 AP 與 GPU replay 尚待排程執行。依第一組約 25 分鐘／epoch 與其餘尺寸 Conv＋PV 成本比例，整體粗估 11 月 10 日上午至晚間完成；此為未量測尺寸的粗估，GPU 資源等待、中斷或異常會順延。後續實測預算寫入各組 budget-estimate.json。固定輸入下獨立改某級 stride／特徵形狀未實作，整網量化、RTL／板端均未進行。

## 背景啟動確認

2026-11-08T18:44:46.008525+08:00：擴充 supervisor PID 3774242，狀態 waiting_initial_queue；初輪 supervisor PID 3766388、GPU worker PID 3767409 仍正常。實際檢查兩把 flock 均有程序持有，擴充已有首筆 monitor，新增 GPU job 為 0，GPU 只有原訓練程序（約 4498 MiB）。原第一組第 6 輪 step440/463。

CPU 測試原始日誌保留 tests.log；擴充預估中央值約 11 月 10 日 11:30，仍對外以當日上午至晚間粗估。長訓由背景 controller 監測，退出 working；新增 GPU AP 尚未完成，不預填結果。

## Queue 完成

Stage9 初輪與擴充全部完成，採用 **reference_rect640**；[結果](yolo_actshape/result/stage9/README.md) 與 [交接](yolo_actshape/handoff/README.md)。階段完成後停止。 完整 AP、GPU replay 與獨立 CPU 可攜驗證完成；清理見 extension/cleanup.json。限制：單 seed、未做 RTL／板端量測；各候選失敗／停止原因保存在 result.json。
