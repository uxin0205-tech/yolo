# 2026-09-09：RepConv 五 epoch 結果與 MASF 零殘差診斷接續

## RepConv 結束事件與驗收

`rep17-parent-ema-control` 完成 E1–E5，quiet supervisor 於 01:39:11（Asia/Taipei）回報 JOB_DONE、exit0、status complete。Blocking launcher session13168 收到 SUPERVISOR_DONE 才讀取 summary；正常期間沒有讀 log 或追加 GPU／指標取樣。單一工作完成不代表全流程 ALL_DONE。

| Epoch | RepConv EMA joint | 相對 native 同 epoch |
| --- | ---: | ---: |
| E1 | 0.7107231630 | −0.0005028413 |
| E2 | 0.7108172764 | −0.0002496546 |
| E3 | 0.7108208400 | −0.0001978796 |
| E4 | 0.7105404138 | +0.0004883535 |
| E5 | 0.7104340984 | +0.0003180428 |

最佳 E3 joint 仍低於原 BEST 0.7111747390，沒有 +0.001 的必要主要收益。E4／E5 比 native 較不退化，Ball Box 分別改善 0.00141305／0.00113186，但相對原 BEST 的 E5 Ball Box 仍低 0.00392926。只能說此配方後段緩解部分退化，不能說已恢復精度或全部指標優於原 BEST。本輪不升格權重、不追加 layer20，也不直接延長已轉差的候選。原 BEST 及所有候選 snapshots 均保留。

## 接續 MASF 的原因與範圍

依 master／trained-model-recovery 計畫，先執行共享 MASF effective gate=0 的免訓練診斷，量測既有依賴，再決定必要 BR-OFF5 bridge。不能直接把已訓練共享模組搬到 Detect-only 路徑，不能把瞬間關閉掉點當成新位置永遠不好。原 native5 可作相同 scope／recipe 的 BR-KEEP 比較來源，重用前核對 parent／trace／設定，而非為湊矩陣重跑。

新增 `masf_bridge.py`：只接受已核對的 `P3MASFFull35` scalar residual，將載入記憶體的 alpha 設為 0 並凍結。保留 context、state keys、原 source、QK 係數與 dataset，沒有使用 forward hook 偷改驗證結果。`run_recovery.py validate --candidate masf_off` 從原 best_joint 載入，provenance 記錄介入；quiet supervisor 新增 masf-diagnostic stage，正常只等待，退出才讀 status／summary。

## 必要驗證

CPU 實際原 BEST：原 alpha=0.11065910756587982；逐 tensor 核對唯一變動為 `graph.model.16.p3_masf.alpha`，零殘差輸出 torch.equal(input)。測試沒有改原 checkpoint，也不作 AP 結論。

下一個 run：`artifacts/direction1-20260908/masf-off-parent-diagnostic`，完整 COCO val5000／canonical BBAT5 val683、原 Float／BitTrue evaluator，不訓練、不抽樣、不重切資料。原 as-is 分數重用已完成同口徑原 BEST revalidation。取得結果後再判斷 bridge 的成本與可行性。

## 困難與未解事項

RepConv 只通過安全完成，沒有驗收收益；MASF 尚未取得新 AP。沒有硬體 latency／energy 實測，不聲稱部署加速數字。困難：無。所有產物保留，無刪除、commit 或 push。使用者允許加訓，但是否延長由驗證趨勢決定，不以訓練不足假設覆蓋已測得的退化。
