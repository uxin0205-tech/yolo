# 2026-09-10：P3 bridge 融合重新啟動

## 授權與變更

使用者選定 P3 bridge MASF，解除本分支的等待決策狀態；後續順序為 combine → activation（QSILU 優先）→ 方向 2。舊結果、停用 queue 與原始權重均保留。本次僅主代理執行。

新增 `combine/bridge_v1/`，提供來源安全載入、共享圖驗證、J0 head 沿用、完整初始驗證、J1 真實更新 smoke、訓練配置及執行入口。PWL 保持 [-10,0]、20 段；MASF 位於 Detect P3 分支，不插回共享 trunk 或 Pose head。

α=0 的輸出與 Identity bypass 等價，但單純設零不省略 MASF 運算。同 checkpoint 開關測試不等於重新訓練無 MASF 模型；此架構的 Pose 推論不隨 α 開關改變。

## 驗證與結果

- 安全來源載入：固定 SHA256、有限類別白名單、`weights_only=True`；不使用任意 pickle 載入。
- CPU 共享圖、來源 Detect 前向等價、α=0／Identity 等價、Pose 不受開關影響及 Float／BitTrue materialization 全部通過，見 `combine/bridge_v1/artifacts/preflight-v1.json`。
- 舊 J0 的 568 個共享 trunk 狀態張量與本分支完全相同，因此嚴格載入其已訓練 Pose head，不重跑 J0；新 J1 重新建立 optimizer 與 20 epoch loss horizon，不聲稱 exact resume。
- 完整 COCO val 5000 張及 canonical BBAT5 Pose val 683 張驗證通過，見 `combine/bridge_v1/artifacts/initial-v1/summary.json`。

| 初始 BitTrue 指標 | α 開啟 |
| --- | ---: |
| COCO overall AP | 0.5082119552 |
| COCO person AP | 0.6276641271 |
| BBAT box AP | 0.5148423321 |
| BBAT pose AP | 0.8070733858 |
| ball box AP | 0.4368011844 |
| ball pose AP | 0.7925487131 |
| bat box AP | 0.5928834797 |
| bat pose AP | 0.8215980584 |

α 關閉減開啟：COCO overall +0.0000207564、person +0.0000642773，BBAT 六項 AP 全部為 0。此時尚未證明 MASF 增益。原独立 Pose AP 0.912160548 仍是比較基準，不降低為 J0 的 0.807073386。

真實 GPU smoke 完成 2 個混合 macro（Detect 512 張、Pose 32 張）；optimizer step 前確認 α 與 context 梯度有限且非零，α 最大變動 0.00000189617。固定參數、EMA 固定狀態、MASF BN buffers 及硬體契約通過；峰值已配置顯存 11,445,453,824 bytes。見 `combine/bridge_v1/artifacts/fusion/j1-smoke-v1/summary.json`。

## 正式 J1 設定與目前狀態

已於 2026-09-10 06:11:40 UTC 啟動 `j1-bridge-v1`。最多 20 epoch、patience 8、warmup 1；AdamW，betas 0.948/0.999、weight decay 0.00027、clip 10。Neck LR 7.5e-5、Detect／Pose head LR 2e-4、MASF（含 α）LR 1e-5；backbone／attention 固定，MASF BN 統計固定，未加入 α clamp。

640 輸入，Detect logical batch 128／physical 32，每 macro 兩個 Detect logical batch（共 256 張）加 Pose 16 張；使用完整 COCO train 與不可變 BBAT5 v1，沒有抽樣或重切。每 epoch 完整 Float／BitTrue 驗證。沿用八指標 checkpoint 選擇，另外 COCO overall 或 person 比本分支初始值下降超過 0.005 AP 時，先保存 epoch 再 safety stop 分析。

監測使用既有 `combine/monitor.py` 的 `child.wait(timeout=600)`，正常 timeout 不輸出；退出事件才處理。此程式不提供 heartbeat stalled 自動辨識，不能聲稱已有此功能。事件檔 `combine/artifacts/logs/bridge-j1-v1.events.jsonl`。尚未建立 activation／方向 2 自動 queue，必須先驗收融合。

## 困難、解法與風險

不受限 checkpoint 反序列化被安全限制拒絕，改採明確 SHA256 與有限白名單的安全載入。一般 sandbox 指令遭 bwrap 網路初始化錯誤，必要指令經個別權限審核執行。避免在 optimizer 清除梯度後誤判：smoke 用 step pre-hook 觀察實際梯度。

未解：J1 尚在訓練；目前共享 Pose 仍低於原独立模型，未宣稱融合成功或精度恢復。訓練後需比較同一 checkpoint 的 α 開／關、COCO overall／person 及 ball／bat box／pose，再決定適應延長與後續階段。沒有覆寫既有結果，沒有 commit 或 push。
