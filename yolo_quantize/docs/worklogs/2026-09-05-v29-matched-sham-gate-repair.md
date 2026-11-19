# 2026-09-05：V29 matched-sham gate修復與安全續跑

## 變更內容與原因

- V29第一個候選 `pose-tower-primary--exact_scaled_ternary` 的sham合法在epoch 9早停後，QAT前置檢查因缺少`best_joint.pt`而失敗；Queue把相同前置錯誤重試兩次後停止。
- 根因是同一個`gate.passed`同時要求最終deployment總下降門檻與matched-sham漂移門檻。sham是訓練控制組，應以相對locked parent的穩定性決定readiness；QAT候選的最終部署結果仍須通過包含activation替換的總下降門檻。
- 在`qat_metrics.py`明確拆出`deployment_passed`與`matched_sham_passed`，並新增獨立`best_matched_sham` selector；沒有降低mAP50 `0.015`或mAP50-95 `0.04`的最終gate。
- 在`qat_runtime.py`新增hash-pinned歷史sham reconciliation及`MatchedShamEvidenceError`。新run會直接保存`best_matched_sham`；既有run可重播逐epoch metrics並引用原本已保存、epoch一致的checkpoint，不覆寫原checkpoint或`qat-experiment.json`。
- 在`progressive_qat_queue.py`讓`retryable=False`的前置證據錯誤只執行一次，避免相同確定性錯誤浪費一次retry。
- 依使用者更正，執行順序維持先完成poly_shift V29七組paired sham/QAT，完成後才開始qSiLU完整權重替換主線。

## 驗證方式與結果

- 原始最小重播命令直接呼叫`Full35QATRuntime._assert_sham_ready()`，修正前穩定得到`TypeError: matched sham did not produce a gate-feasible best_joint checkpoint`。
- 三個CPU regression tests先得到三個預期紅燈，修正後為`3 passed`：分離sham/deployment gate、歷史sham reconciliation，以及非重試前置錯誤只呼叫一次。
- 相關測試：`24 passed in 19.29s`。
- 全專案CPU-only測試：`268 passed in 50.60s`。
- Ruff：所有本次修復檔案通過；`compileall`與`git diff --check`通過。
- 實際sham重播選定epoch 4，joint score `0.8300559015079926`，相對V19 locked parent最大絕對漂移`0.00669437885161106 < 0.01`。
- 重用的原checkpoint為`best_pose.pt`，SHA-256 `05f34ad1c6184226b81fb2d3f42858e3f5d989e777f8403ea2037be66d826202`；新證據`matched-sham-evidence.json` SHA-256為`966dc637db57ef69e935da4e9b073feed79367a0d92ec207e5c55edcdb4829df`。
- 原始失敗loop修正後成功回傳`evidence_kind=reconciled`。診斷、修復與重播全程未使用GPU，未更改BBAT5資料或split。

## 困難與解法

- 困難：sham epoch 0–8均未通過總體mAP50 gate，但相對matched parent的最大漂移皆小於`0.01`；原本沒有獨立的控制組checkpoint角色。
- 解法：將控制組穩定性與最終部署可接受性拆成兩個可稽核判定，僅以後者決定最終QAT候選是否晉級。
- 困難：既有sham未保存`best_matched_sham.pt`，重新訓練約需數小時。
- 解法：CPU重播十個epoch，並用安全`torch.load(weights_only=True)`確認原`best_pose.pt`的progress正是epoch 4，再產生獨立hash-pinned reconciliation；沒有複製或改寫大型checkpoint。
- 困難：`apply_patch`因環境`bwrap: loopback: Failed RTM_NEWADDR`失敗。
- 解法：已先依規範嘗試；之後使用限定workspace、唯一sentinel計數必須為1的寫入器，任何內容漂移即停止。

## 未解事項或風險

- 修復只允許穩定sham作為配對控制，不代表它符合最終部署門檻；第一個QAT及後續所有候選仍必須通過原`0.015` mAP50總下降gate。
- V29尚有第一個QAT與其餘六組paired jobs未完成，尚無short-QAT winner。
- qSiLU權重替換會在V29完成後另用自己的activation baseline、W8 parent與逐區sensitivity重跑；不直接搬用poly_shift結果。
- formal、長epoch、multi-seed及最終硬體優化仍未授權，不會在本Queue自動執行。
