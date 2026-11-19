# 2026-08-29：收斂Full35 activation-aware量化實作計畫

## 任務與範圍

將使用者多輪確認的Full35量化決策收錄成逐步執行的整體計畫，特別補入Detect logical batch 128、Pose logical batch 16、physical microbatch fallback，以及activation queue完成後先分析、再依工作樹逐節點推進的gate。本次只修改`yolo_quantize`文件，沒有修改Full35、`yolo_activation`、dataset、checkpoint或訓練程序，也沒有啟動GPU工作。

## 變更內容與原因

1. 新增根層`IMPLEMENTATION_PLAN.md`，取代舊`quantize_spec.md`作為現行執行入口；舊規格保留為研究參考，避免直接改寫或遺失原始假設。
2. 將正式基線改為`yolo_combine/final/full35` Detect＋Pose joint模型，並明確保護Binary Q/K、MASF、attention PWL、TopK／decode。
3. 收錄activation–quantization強耦合矩陣：SiLU、`poly_quality`、`poly_shift`、`qsilu_pq`與controls分別建立matched baseline，再測LSQ+ A3–A8，最多保留三個完整activation policies進入weight矩陣。
4. 收錄INT8、INT4、Fixed SD4、LS-SD4、Paper-TWN、Channel-TWN與TTQ候選，以及Distribution-Sensitivity Format Router。
5. 將最終精度硬上限凍結為每個主要task absolute mAP下降`0.04`；W8A8工程gate維持較嚴格的`0.01`。
6. 收錄固定30% diagnostic train、完整validation、20→60→100／120 epochs與3 seeds流程。
7. 補入batch契約：Detect logical 128、每macro兩個calls；Pose logical 16、每macro一個call。physical microbatch可依OOM降低，但必須保持logical batch、optimizer step、loss normalization與task exposure；observer跨microbatches彙整後才更新。
8. 定義activation-ready gate：queue終態、必要uniform結果與completion markers齊全、GPU job停止、來源與結果hash snapshot完成後，先只進入P0 intake analysis；P0通過gate後才循工作樹進入P1，不一次啟動全部矩陣。
9. 新增子專案README與工作紀錄索引，提供資料規範、現行計畫與動態狀態入口。

## 驗證方式與結果

- 唯讀列出`yolo_quantize`現有檔案，確認修改前只有`quantize_spec.md`與PDF，沒有既有README、實作計畫或子專案工作紀錄可被覆蓋。
- 唯讀核對Full35 activation queue、completion markers、policy plan與2026-08-28工作紀錄。
- 最新既有queue state內容為`3 completed / 16 pending / 0 blocked`，下一個job是`short-recovery-uniform-poly-shift`；因此計畫正確標示為尚未activation-ready。
- 已完成的qSiLU zero-shot與short recovery均有正式gate；Hardswish short recovery有一項原activation gate未通過。這些只記為動態證據，未宣稱winner。
- 本次沒有執行訓練、validation、export或資料抽樣。

## 困難與解法

- 困難：舊`quantize_spec.md`仍指向不存在的Detect-only checkpoint／dataset，且把SiLU視為主activation，和目前Full35 joint模型及硬體activation策略衝突。
- 解法：不破壞舊文件，另建現行實作計畫並明訂文件優先順序。
- 困難：最初沙箱唯讀命令遇到`bwrap: loopback: Failed RTM_NEWADDR`。
- 解法：使用核准的沙箱外唯讀命令完成查核，沒有修改外部專案。

## 未解事項或風險

- `yolo_activation` queue仍未完成；其最新結果、source與checkpoint hashes必須在正式啟動時重新快照，不能使用本紀錄中的動態狀態代替。
- 量化package、adapter、tests、configs與run orchestration尚未實作；activation-ready後先只進入P0分析，P1以後依工作樹與當階gate逐步推進。
- 固定30% diagnostic view尚未建立；建立時必須沿用canonical assignments、完整validation與group-safe manifest。
- 尚無target FPGA／ASIC profile，因此第一版只能報hardware-oriented proxy與BitTrue evidence，不能宣稱真實latency、power或resource改善。

## 後續修正（同日）

上游主queue其後已進入終態，finalist仍pending／interrupted。依使用者授權，screening gate與finalization gate分離；本專案已完成provisional intake、Full35 adapter與無訓練A2 smoke，但仍未進正式訓練。最新執行狀態以IMPLEMENTATION_PLAN.md與2026-08-29-activation-coupled-smoke-matrix.md為準。
