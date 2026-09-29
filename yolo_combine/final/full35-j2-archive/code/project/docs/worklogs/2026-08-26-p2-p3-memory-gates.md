# 2026-08-26 Full35 P2/P3 顯存診斷與 logical128 fallback

## 目的

診斷 P1 完成後 P2 第一個 batch 的 cuSolver failure，確認 physical batch可行邊界，並在
不關閉 Pose26 RLE、Bit-True XNOR、AMP或任何模型分支的前提下恢復正式管線。

## 紅色 feedback loop

實際正式命令 `variants/full35/run.py pose --stage p2 ...` 在空卡、physical128的第一個
forward穩定觸發錯誤。完整 log先顯示：

```text
CUDA out of memory with batch=128. Reducing to batch=64 and retrying (1/3).
```

固定 Ultralytics 8.4.90原生 trainer會在第一個 epoch偷偷把batch減半並重建loader／
optimizer；retry後才在RLE `MultivariateNormal`顯示`CUSOLVER_STATUS_INTERNAL_ERROR`。
所以cuSolver是OOM後的次生錯誤，不是第一原因。

## 假設與逐項結果

1. **P2-b128真正OOM，retry後CUDA狀態造成次生錯誤：成立。**
   新增fail-fast後，P2-b128直接在head forward報OOM，不再出現降batch或cuSolver假象。
2. **P1 checkpoint的RLE flow/covariance損壞：排除。**
   全新程序載入P1 best，float32 RLE flow對1,024筆輸入成功且全finite。
3. **cuSolver環境持續異常：排除。**
   空卡全新程序連續100次CUDA Cholesky通過。
4. **特定首批資料讓RLE失敗：排除。**
   同canonical資料的P2-b64全新程序完成真實forward/backward及完整validation。

## 實測 memory gate

| stage | physical batch | accumulate | 結果 | 觀察 |
| --- | ---: | ---: | --- | --- |
| P1 head-only | 128 | 1 | 通過 | train約22.9 GiB；17 epochs完成 |
| P2 neck+head | 128 | 1 | 失敗 | process 30.89 GiB、free120 MiB，仍需400 MiB |
| P2 neck+head | 64 | 2 | 通過 | 真實forward/backward峰值約20.5 GiB |
| P3 full-unfreeze | 64 | 2 | 失敗 | process 30.97 GiB、free36 MiB，仍需50 MiB |
| P3 full-unfreeze | 32 | 4 | 通過 | 真實forward/backward峰值約18.7 GiB |

P2/P3都保留`nbs=128`。Ultralytics的accumulate分別為2與4，所以每次optimizer update
仍看到128張，且weight decay縮放仍是`wd * batch * accumulate / nbs = 2.7e-4`。這是
logical128，不宣稱與physical128逐值等價；BN與每次loss normalization仍按64／32。

## 程式與設定變更

1. formal Pose trainer在每個train batch前鎖定`physical_train_batch_size`，並把原生
   first-epoch OOM retry設為不可用；OOM現在保留原始exception並fail-fast。
2. stage profile鎖定P1=128、P2=64、P3=32；`nbs=128`不變。
3. run名稱明確包含accumulation：P2 `b64a2`、P3 `b32a4`。
4. pipeline、`joint.yaml`的未來P3 checkpoint、runbook、README與計畫同步更新。
5. 所有memory repro都在`artifacts/pose/diagnostic/`，有fraction marker且不可進formal gate。

## 驗證

- fail-fast測試先紅後綠：會把`_oom_retries`鎖到上限，且batch被改動時明確報錯。
- stage contract測試先紅後綠：P1/P2/P3 batch為128/64/32，accumulate為1/2/4。
- P2-b64與P3-b32均以canonical派生diagnostic view完成真實CUDA forward/backward；
  RLE保留、loss finite、validation完成。
- 完整repository CPU回歸：`113 passed`。

## 正式恢復狀態

2026-08-26 09:36 CST建立
`yolo-full35-seed0-pipeline-resume2.service`。它已跳過完成的diagnostic P1與formal P1，
並於09:38啟動`p0-full35-p2-b64a2-e22-seed0`。首批執行時GPU約22.3 GiB，沒有
auto-reduction或cuSolver錯誤；service為active。P2成功後會自動接續P3與其餘pipeline。

## 困難與解法

1. 困難：表面stack落在cuSolver，容易誤判為RLE或driver問題。
   解法：回看完整時間序列，發現其前一個事件是原生auto-reduce；再以fresh-process
   Cholesky、RLE-only、64/128 differential loop逐項排除。
2. 困難：使用者偏好所有stage physical128。
   解法：保留P1=128；只有實證失敗的P2/P3改用恰好整除nbs128的64/32，維持logical
   exposure與weight decay，不以任意batch或關閉模型分支換取可跑。

## 未解事項與風險

- JOINT shared trunk的Detect physical128尚未做GPU memory gate；若失敗，必須採同樣
  fail-fast與明確microbatch設計，不得silent reduction。
- physical64/32與physical128的BN統計、loss batch normalization不逐值等價，最終只能由
  standalone與JOINT同口徑validation及0.08 hard gate判斷。
- P2/P3正式長訓練尚未完成；memory diagnostic的AP不可當正式baseline。

除上述風險外，無其他未解困難。
