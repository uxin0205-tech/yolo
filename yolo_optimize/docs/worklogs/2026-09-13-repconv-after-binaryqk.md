# 2026-09-13：RepConv 改接 BinaryQK 重訓 E5

## 使用者意圖、變更與原因

使用者詢問 scale/bias 與共同起點 B5 的設計差異，並明確要求 RepConv 在 MASF 處理完、BinaryQK 重訓後開始。原設計是 B5 分別出 scale_bias／Rep17／Rep20／雙層，與新需求不同。新增 `experiments/post_binary_rep_v1/`，固定上游 scale_bias E5 SHA `c8dcaf439b68c77d45a4f213972e57d51912028b558418511badee55fafbf681`，三組獨立共用此起點。舊資料夾、weights、checkpoint、驗證不覆寫。

原排程器 PID 2517588 與雙層接續器 PID 2578493 已依命令路徑核對後停止，不終止其訓練 child。原 Rep17 PID 2579897 由新 queue 以 PID＋start ticks 確認身分，等有完整 epoch checkpoint 與 metrics 後收束；這是規格變更，不是訓練 ERROR。未開始的舊 Rep20／雙層取消。舊排程狀態的歷史 JOB_STARTED 不代表仍有效，新增 SCHEDULE-SUPERSEDED.json 與 README 最新入口。

## 設計解釋

MASF B5 是原生 QK＋PWL。scale_bias 組先保留 B5 QKV／PE／proj 與兩個 head、MASF，替換成兩個 binary basis 的 Attention score。16 個共用 m/1024 scale、相對 x/y signed16 m/1024 bias，以 STE 量化與 task loss 調整；不是每張圖估計 scale。前向 XNOR/popcount，反向 dot surrogate＋clipped sign STE。訓練範圍是 Attention 與 head，不僅 scale/bias；MASF、其他共享參數、BN running 固定。

新 Rep 組保留 E5 的已訓 BinaryQK 與常數，固定 Attention／兩套 MASF，開指定 Rep 層與兩個 head，5 epoch、warmup 1、AdamW 沿用設定。原 E5 bat Pose 有 -0.5228 pp 退步；繼續作研究起點不等於採用。新報告同列 MASF B5、BinaryQK E5 與 Rep 結果，採用參考閘同時保護兩起點，不自動升版。沒有同預算未改架構加訓對照，無法完全分離結構收益與適應訓練收益。

## 驗證與結果

新三組 CPU preflight 通過：strict 載入 E5 state、常數逐項一致、初始輸出等價、非零 Rep 支路 fold 等價、正確 optimizer 分組、MASF／Attention 凍結、Float／BitTrue 實體化保留 E5 QK scale 與 x/y bias，並確認 PWL[-10,0]20段。沒有使用 GPU 做 CPU 前置。新程式 AST 通過，啟動前來源 SHA 全部核對。後續 GPU smoke 必須逐一確認所選 Rep 新支路的梯度與實際更新。

新 queue PID 2589417 已啟動並確認存活，初始事件 `WAITING_OLD_EPOCH_BOUNDARY`（2026-09-13T14:41:28Z）。先保護舊回合，之後使用共享 GPU lock 與空閒准入，依序 Rep17 → Rep20 → Rep17＋20；新 GPU 訓練尚未開始。GPU 正常執行每 600 秒監測；只有本次舊回合交接每 60 秒檢查存檔邊界，無週期讀訓練 log。

## 困難、解法與未解事項

困難：執行中舊 queue 沒有可熱更新的 parent，也沒有 trainer 內建回合暫停旗標。解法：停止已被取代的排程器、保留 child 至可恢復回合邊界，新增獨立來源／產物的 queue，避免直接改動正在使用的檔案。新發現的程式／GPU 錯誤：無。

尚待：舊回合收束事件、新起點 GPU smoke、正式 5E 與全量驗證。新 CPU 通過不代表精度提升；未刪除任何資料、未推送 Git、未追加原生對照。背景排程器不具模型推理能力，不能保證對話結束後自動喚醒主代理。
