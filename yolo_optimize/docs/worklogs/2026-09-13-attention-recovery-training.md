# 2026-09-13：啟動 scale／bias 與原生 QK＋PWL 三臂恢復

> 後續更正：使用者已取消 binary_control 正式對照；下列三臂內容是啟動時的歷史紀錄。現行兩組計畫見 [取消對照紀錄](<2026-09-13-cancel-binary-control.md>)。

## 變更內容與原因

依使用者明確授權試驗，新增獨立 modeling、CPU preflight、完整 E0、真實 GPU smoke、三臂訓練 adapter、串列 queue 與最終比較程式。三組同 parent／資料／最大預算，原正式權重保留。PWL 固定 [-10,0]20 段；不重切 COCO 或 BBAT5。

scale_bias 的16個係數真正接到前向與 task-loss 梯度，量化為 m/1024 固定常數，取代只改無效 gamma。原生 QK fused 投影按 head 交錯映回 Conv／BN，保留 V／pe／proj，移除二值與額外 bias。新訓練局部政策開放 Q/K，保護共享 BN／PWL／固定 MASF；不修改外部套件與舊 runner。

## 驗證方式與结果

CPU preflight 通過：兩個 Binary 組初始輸出與選定模型精確一致；原生 fused QKV 對 FP-QK 無 bias 參考在3e-5容差內；新尺度有梯度、eval 快取更新及 state dict 重載一致；Float→BitTrue materialization 保留正確架構及 PWL。

新 E0 全量 COCO5000／BBAT683 通過。scale_bias 八項 AP 精確重現現行模型；native_qk E0 COCO0.490888、person0.617605、BBAT框0.600297、Pose0.874286，是尚未訓練的起點，不是重訓成果。GPU smoke 與後續正式訓練狀態、精確數值在 experiments/attention_recovery_v1/artifacts/ 各自保存。

三臂參數：AdamW、warmup1、20E上限、patience5，Detect physical16 累積、每macro256 Detect＋16 Pose。attention LR1e-5，尺度LR2e-4，backbone3.8e-7、neck1.9e-6、heads5e-6、MASF0。完整配置在該實驗 README 與 resolved-config。磁碟初查仍有605GB可用。

## 困難與解法

CPU 梯度測試初次保留舊權重的 requires_grad=False，改成測試時顯式解凍後通過；沒有忽略無梯度。新 evaluate.py 與舊 activation 同名導致循環 import，改名 validate_recovery.py。既有 loader 要求 full35/configs 目錄、tensorboard 枚舉值、合法 stages 及 Float→BitTrue 雙驗證，依既有 adapter 方式符合契約，不放寬舊流程檢查。

舊 guard 會重新凍結 Q/K，因此新流程使用限定的固定 state guard；共享 BN、PWL、MASF 與 control scales 仍逐次核對。scale-only 先採 task-loss 校準，不把學生 FP score 擬合當作精度保證。

## GPU 前置驗證與正式啟動補記

三組真實 GPU smoke 均通過，每組實際更新兩個 macro，共 512 張 Detect 與 32 張 Pose 訓練輸入。峰值 allocated 顯存：native_qk 約 16.847 GiB、scale_bias 約 16.968 GiB、binary_control 約 16.888 GiB。兩個 Binary 組的兩處 Attention，Q 與 K 在每個 macro 均有非零梯度；scale_bias 的尺度參數有梯度並實際更新。native_qk 的 fused QKV 有梯度與權重更新。固定狀態 guard 與 EMA 檢查通過。

2026-09-13 15:36（Asia/Taipei）正式 queue 啟動 native_qk，依序安排 native_qk → scale_bias → binary_control；各組訓練後接續獨立驗證與 MASF 開關比較。前置測試不算正式訓練回合，也不作為精度提升證據。queue 的事件記錄位於 `experiments/attention_recovery_v1/artifacts/queue-v1/events.jsonl`；原始正式權重未覆寫。此階段無新增訓練問題；文件更新沿用已核准的 apply_patch 包裝器，以避開本機 sandbox helper 的 bwrap 錯誤。

## 未解事項與風險

正式精度恢復尚待訓練及獨立驗證；一個 seed 不能宣稱普遍增益。同上限預算不保證早停後實際回合完全一樣，最後需列回合與曲線。沒有新增 Git commit／push、沒有目標板驗證；原 best 不自動替換。60秒内部等待用於工具限制，600秒監測契約不增加正常 log／GPU 探查。
