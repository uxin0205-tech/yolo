# 融合前 Rep17 驗收與 B100 MASF 零殘差診斷

## 變更與原因

接回原 session 34916，收到 control、rep 各自 JOB_DONE／exit 0，以及 queue ALL_DONE。沒有重啟或重跑任何既有 job。兩組完整訓練各 5 個 epoch，每 epoch 都是 118287 張、925 次更新，並各驗證 EMA 與 live 的完整 COCO val。

| 權重／回合 | EMA overall AP50–95 | EMA person AP50–95 |
| --- | ---: | ---: |
| 起點 late E8 | 0.5081536354 | 0.6275205166 |
| control E4 | 0.5080459000 | 0.6273521098 |
| Rep17 E4（該組 overall 最佳） | 0.5080975361 | 0.6275006526 |
| control E5 | 0.5080466670 | 0.6273090847 |
| Rep17 E5 | 0.5078054401 | 0.6271856846 |

Rep E4 相對同回合 control 為 overall +0.0000516361、person +0.0001485429，未達 +0.001 主要改善門檻；overall 也未超過起點。E5 轉差，不延長、不擴大至 layer20、不升格權重。所有結果與 checkpoint 保留。此結論僅限目前配方，不能推論 RepConv 在其他訓練條件下一律無效。

## 下一階段與最小驗證

新增 `studies/pre-fusion-full35-b100/scripts/probe_b100_masf_off.py`，先完成原計畫要求的 shared MASF 依賴診斷，再決定 bridge 與位置實驗。

原 B100 路徑：

```text
layer16：p3_raw → MASF → p3_shared ─┬─> layer17 → P4 → layer20 → P5
                                   └─> Detect 的 P3 輸入
layer19：P4 ─────────────────────────> Detect 的 P4 輸入
layer22：P5 ─────────────────────────> Detect 的 P5 輸入
```

本次只在載入記憶體的 B100 Bit-True 模型設定 `model.16.p3_masf.alpha=0`，不改 context 權重、不搬位置、不輸出取代原始權重。先檢查逐 tensor 只有 alpha 改變、零殘差精確 identity、原位介入傳到 P3／P4／P5 的程度；正式驗證前移除只讀觀察 hook。之後跑完整 COCO val5000，主要看 overall／person，另外列 ball／bat。

對照重用已完成、同來源 SHA256 與同設定的 `baseline-bittrue-v2`，不重跑正常基準。瞬間關閉的 AP 不是重新訓練後無 MASF 的效果，也不是位置優劣的完整因果結論。

## 驗證狀態

Rep17 兩組 summary 均為 complete，來源 SHA、首批 trace、首批 loss 一致，既有逐回合固定 state／部署格式驗證已由訓練流程通過。新診斷先做 AST 檢查，再交給現有 600 秒 supervisor；實際結果待 job 完成後記錄。

## 困難、解法與風險

困難：前次結束模型回合後，shell 已完成但分析沒有自動接續。本次實際接回並讀取完成事件，接續分析，不將背景存活誤報為模型持續處理。

另有相似命名的融合後 Rep17／MASF 紀錄；核對其 joint 指標後確認屬舊研究，未混入本分支。其餘困難：無。新診斷 AP 與後續 bridge 尚未完成，不保證精度一定回升，也不聲稱硬體延遲收益。
