# 2026-08-27 Full35 J3 challenger

## 變更內容與原因

Full35 v2 已完成 J0 8、J1 20 與 J2 25 epochs；J2 因 Bit-True joint score 連續
17 epochs 未超越既有最佳而正常 early stop。J2 最佳 joint score 是
`0.7100382159275817`，最後一個 epoch 是 `0.7100156511462141`，兩者只差
`0.0000225647813676`，且兩者八項 hard gate 均通過。

依使用者授權，新增獨立的 J3 challenger queue：

- 父 run：`full35-joint-adamw-v2-j0e8-j1e20-j2e80-seed0`。
- 第一次diagnostic failure：`full35-joint-adamw-v2-j3-challenger-seed0`。
- 正式retry challenger：`full35-joint-adamw-v2-j3-b32-challenger-seed0`。
- J3 最多 20 epochs、patience 5、full-backbone 低學習率微調。
- XNOR 仍為 `tiled_exact`、`qk_ste=false`，Bit-True 是正式 selection backend。
- 原 J2 run、四個 checkpoint、logs 與 validation 完全不覆寫。

J3 使用父 run 的 `last.pt` exact-resume，而不是直接使用 `best_joint.pt`。原因是只有
J2 `last.pt` 記錄 `stage_complete=true`；若拿 epoch 35 的 `best_joint.pt` exact-resume，
trainer 會正確地把 J2 視為尚未完成並繼續 J2。父 run 的 `best_joint.pt` 保留為
acceptance reference：只有 J3 產生更高的 Bit-True joint score且八項 gate 全通過時，
新 run 才會出現自己的 `best_joint.pt`。

新增腳本：

```text
scripts/run_full35_v2_j3_seed0.py
```

它會依序執行 formal preflight、等待 GPU 連續兩次符合空閒門檻、exact-resume J3，並且
只在 J3 確實超越父 run 時對候選 `best_joint.pt` 執行獨立 Float／Bit-True final
validation。若沒有超越，正式選擇仍指向父 run 的 `best_joint.pt`。

## J3 設定

| 參數群組 | learning rate |
| --- | ---: |
| backbone | `3.8e-6` |
| neck | `1.9e-5` |
| MASF | `3.8e-5` |
| attention可微部分 | `5e-7` |
| Detect head | `5e-5` |
| Pose head | `5e-5` |

共同設定保持 AdamW、Detect/Pose task weight `1.0/0.25`、Detect logical128
（J3改為physical32×4）、Pose physical16、shared BN running statistics frozen、AMP與每 epoch
Float／Bit-True雙驗證。J3 不開啟 Q/K STE；Q/K sign path與hardware contract不改變。

第一次啟動沿用J1/J2的Detect physical64，在J3第一個full-unfreeze forward穩定重現OOM：

```text
torch.OutOfMemoryError: Tried to allocate 200.00 MiB
GPU total 31.35 GiB; free 209.06 MiB
process 30.74 GiB; PyTorch allocated 29.70 GiB
```

當時沒有其他GPU compute process，因此排除外部占用。J3解凍完整backbone與attention
可微部分後，physical64已沒有安全餘裕。修正只改執行切片：logical Detect batch仍是128，
每個logical batch由physical32×4累積；每個macro仍是2×logical128，因此會依序執行8個
Detect microbatches再加1個Pose16，最後只做一次optimizer step。

## 驗證方式與結果

啟動前驗證：

- `joint.py preflight`：`ready=true`、`blockers=[]`。
- J2 `last.pt`：`stage=j2`、`next_epoch=25`、`stage_complete=true`。
- GPU：空閒，約 31.7 GiB free，無 compute process。
- 容量：磁碟約 1.2 TiB available；RAM約 119 GiB available。
- `python -m py_compile scripts/run_full35_v2_j3_seed0.py`：通過。
- `scripts/run_full35_v2_j3_seed0.py --plan`：正確指向獨立 run、父 `last.pt`與父
  `best_joint.pt`。
- 修正前回歸測試：`2 failed`，能抓到runtime microbatch seam不存在。
- 修正後J3 runtime batch／stage／startup／patience／config／resume tests：`15 passed`。
- 完整CPU test suite：`124 passed`。
- b32原始GPU repro：已連續完成至少14個J3 macro-steps；每步Detect batch sizes皆為
  `[32,32,32,32,32,32,32,32]`、Detect images=256、Pose batch=16，shared／Detect
  head／Pose head梯度皆存在，AMP overflow=0。
- b32執行中GPU約使用21.0 GiB、保留約11.1 GiB，已通過原physical64第一個forward
  即OOM的失敗點。

## 遇到的困難及解法

困難一：J3 必須從已完成 J2 銜接，但最佳權重 checkpoint 位於 J2 中途，不具
`stage_complete=true`。

解法：使用分數幾乎相同且帶完整 stage completion state 的 J2 `last.pt` exact-resume；
父 `best_joint.pt`永久保留為候選必須超越的基準，並使用不同 run directory，避免任何
證據覆寫。

困難二：J1/J2可用的physical64在J3完整解凍後，第一個forward即OOM。

解法：新增只允許J3 transition/resume使用的`--j3-detect-microbatch 32`。runtime batch
plan寫入resolved config、provenance與checkpoint loader state；若J3中斷後沒有帶相同
plan續跑會fail fast。第一次失敗的partial run與queue status永久保留，不升格為正式候選。

## 未解事項或風險

- J3 是 seed0 challenger；即使勝出，未跑 seed1 前仍屬單 seed 結論。
- `qk_ste=false`，所以 J3 只微調 attention 的 V、PE、projection、relative-bias等可微
  部分，不會宣稱已訓練 binary Q/K sign decision。
- J2 已呈現驗證平台期，J3 可能無法改善；此時應拒絕 challenger並繼續使用父
  `best_joint.pt`，不能用最後 epoch 或平均分數取代正式 selector。
- physical32累積保持logical128 optimizer exposure，但head BN與loss內部batch統計不會
  宣稱逐值等同physical128；J2比較基準本來也是microbatch執行策略。

## 目前執行狀態

- 第一次physical64 service：`yolo-full35-v2-j3-seed0.service`，已失敗停止，診斷產物保留。
- 正式b32 service：`yolo-full35-v2-j3-b32-seed0.service`，於2026-08-27 00:32 CST啟動。
- queue state：`variants/full35/artifacts/queue/full35-v2-j3-b32-seed0/queue-status.json`。
- 啟動確認時狀態為`training_j3`；J3最多20 epochs、patience5。
