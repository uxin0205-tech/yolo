# MASF E10 驗收、梯度路徑診斷與等待限制

## 變更與原因

延續融合前 Full35-B100 方向 1，不處理融合後、Pose 或 person-only。使用者要求減少 50 秒分段喚醒，並繼續增準研究。本次保留所有訓練與既有結果，新增 `probe_masf_task_gradient.py`，只作 CPU 梯度連通及 E6–E10 配對結果驗證，沒有更新權重。

## 600 秒等待核對

現有 `run_scope_pair.wait_job` 和 `monitor.py` 均使用 `child.wait(timeout=600)`，正常逾時直接繼續，不讀 log、不查 GPU。此部分已存在，不需要重啟或另開監測器。

本機 CLI 有 `codex queue` 與 app-server proxy，但唯讀執行 `codex app-server daemon version` 顯示 control socket 不存在，未證實有可連接目前會話的事件喚醒服務。本次沒有啟動 daemon、其他代理或排入訊息。工具層的短等待仍未改成完整 600 秒；不能把底層 600 秒等待描述成模型完全不會被喚醒，也不能保證畫面持续 Working 或零 token。

## 完成結果

接回原 session 的完成事件：fork 在 UTC 2026-09-09 17:19:00 正常退出，queue 回報 ALL_DONE。control／fork 均完成 E6–E10，每 epoch 118,287 張、925 次 optimizer updates，第一個 macro 的資料雜湊一致。

以下皆為完整 COCO val 的 EMA AP50–95，0–1 尺度，使用相同 epoch 比較：

| Epoch | control overall | fork overall | control person | fork person |
| --- | --- | --- | --- | --- |
| 6 | 0.508039439 | 0.508163747 | 0.627519679 | 0.627608130 |
| 7 | 0.508223400 | 0.508161314 | 0.627565375 | 0.627538193 |
| 8 | 0.508267092 | 0.508160784 | 0.627699455 | 0.627542027 |
| 9 | 0.508172836 | 0.508131163 | 0.627648052 | 0.627565773 |
| 10 | 0.508172720 | 0.508088950 | 0.627605819 | 0.627703129 |

提高 head LR 並續訓沒有帶來穩定的 MASF 增益，沒有達到既定 +0.001 驗收門檻。E10 fork 的 bat 高於對照、ball 低於對照，兩者不取代 overall／person 決策。不能將本輪稱為成功，亦沒有理由只延長相同配方。

## 梯度診斷與推導

目前 fork：

```text
p3_raw → MASF → p3_det ─┬→ one-to-many → loss → 梯度回 MASF
                        └→ detach → one-to-one → loss ─×→ MASF
                                      └→ 部署推論
```

原生 Detect 的 detach 是刻意設計，不是本次證明的程式錯誤。它意味著部署分支沒有直接監督 MASF，但不能單憑此事就歸因 AP 下降。

下一個可檢驗的訓練專用路徑：

```text
p3_raw ─────────→ MASF → one-to-many
   └→ detach ──→ 同一個 MASF → one-to-one
                    ↑ 梯度可回 MASF，不回 Backbone

推論仍是：p3_raw → 單次 MASF → one-to-one
```

在訓練時對已 detach 的原始 P3 重新計算相同 MASF，可以保持 one-to-one 的前向值，同時把該分支梯度導向 MASF。此等價性依賴本輪固定 BN 統計、沒有隨機 context 操作；不可直接外推到任意模組。

## 驗證方式與結果

CPU 診斷使用完成的 fork E10 EMA，固定種子的合成 P3/P4/P5 特徵，未使用 GPU、未更新參數。結果 `artifacts/masf-task-gradient-probe-v1/summary.json` 為 passed：

- 原生 one-to-one 對 MASF 梯度不存在；one-to-many 對 MASF 梯度非零。
- 新路徑的 one-to-one 對 MASF 梯度非零，對 raw P3/P4/P5 梯度不存在。
- one-to-one boxes／scores 前向完全相同，所有 head state 和來源 checkpoint 雜湊均未變。
- 同步確認两組 E6–E10 完成、完整 train 數量及配對資料雜湊。

這是合成目標的連通性驗證，不是真實 Detect loss 的梯度校準、AP 提升證據或硬體驗收。

## 下一步與超參數邊界

先以固定 train-only prefix 驗證真實 one-to-one loss 對 MASF 的梯度量級、方向與單次更新穩定性。通過後，才考慮讓新路徑與既有 native fork 配對比較；優先重用相同 E5 起點、資料 seed 與已完成 native fork E6–E10 對照，不重跑正常對照組。

候選沿用 physical batch 32、累積 4 次為 logical 128、AdamW、P3 head LR 1e-5、context LR 1e-5、alpha LR 1e-4、全 BN 統計固定、alpha 限制 [-0.25, 0.25]；保留 optimizer／EMA 年齡與原始 1 epoch warmup，不在續訓時重啟 warmup。新增梯度權重應由真實 loss 校準後固定，尚未指定或開跑正式訓練。推論不增加額外 context 分支。

## 困難、處理及未解事項

等待介面尚無已驗證的同會話事件喚醒服務；如實保留此限制，不新增第二代理。其餘 CPU 驗證困難：無。MASF 增準、真實 loss 校準、候選全量訓練與推論驗收仍未完成。本輪 GPU queue 已結束，目前沒有新 GPU 工作啟動。
