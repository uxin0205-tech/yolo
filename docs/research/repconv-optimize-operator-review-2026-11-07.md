# RepConv 與 optimize 算子查核（2026-11-07）

使用者要求先等等並重新查核，GPU 尚未啟動。先前準備的 SiLU／exact Softmax 方案撤下，不把它當成 RepConv 的必要流程。原生 RepConv 類別並不強制使用 SiLU；optimize 實際保留上游 qSiLU。

## 直接證據

- `yolo_optimize/experiments/src/yolo_optimize/repconv.py:20`：建立 Rep 時傳入 `act=copy.deepcopy(original.act)`，不採用預設 activation。
- `yolo_optimize/experiments/pose_only_rep_v1/preflight.py:52` 以及 `:73`：確認 parent 與三組 Rep 都是 qSiLU，且沒有 `nn.SiLU`。
- `yolo_optimize/experiments/pose_only_rep_v1/artifacts/preflight-v1.json`：三組均已通過，`native_silu_absent=true`、初始等價與非零分支 folding 等價。
- `pose_only_rep_v1/models.py` 的 build／folded：保留原 activation，3×3＋BN 與 1×1＋BN 相加後才做 activation；部署融合線性部分。
- `pose_masf_training_v1/training_b.py:62`、`attention_recovery_v1/modeling.py`、`activation/bridge_v1/verify_selected.py`：上游載入 float PWL；`achitechure_1/final/configs/attention/float-pwl-final.yaml` 固定 [-10,0]／20 段，並非 exact Softmax。

完整來源 SHA、實際 run 配置與 AP 位於 `yolo_achitechure/stage8/reference-review.json`。

## 分開理解兩種變更

Rep 訓練圖是 `(Conv3×3+BN + Conv1×1+BN) → 原 activation`；部署圖是 `Folded Conv3×3 → 相同 activation`。固定 BN 後，這個線性融合可以驗證數值等價。

SiLU 改成 qSiLU 則是另一個非線性近似，前向輸出與反向導數可能改變。SiLU 訓練的 AP 不能代替轉成 qSiLU 後的部署 AP。使用者允許訓練不受硬體限制，並不表示必須把已適應的 qSiLU 換回 SiLU。

| 項目 | optimize 已完成 Rep | 本專案 Stage8 v1 | 本輪已撤下草案 |
|---|---|---|---|
| Rep | 原生 RepConv | 自訂等價雙分支 | 原生 RepConv |
| 訓練 activation | qSiLU | qSiLU | SiLU |
| 訓練 normalization | Float PWL [-10,0]／20 | D1 整數前向＋Float PWL 反向代理 | exact Softmax |
| 部署 | qSiLU，Float／BitTrue PWL；分母仍浮點參考 | qSiLU，完整 D1 reciprocal | 原定 qSiLU／D1；未開始 GPU |

舊輪 Rep 的雙分支與折疊方式並沒有因使用 qSiLU 而錯誤。換成原生 RepConv 類別本身不應被宣稱會帶來精度提升。

## 接續建議與限制

保留已適應的 qSiLU；若要更貼近 optimize，可比較 Float PWL 訓練、最終轉完整 D1 部署的流程。保留 B2／S1，先重新固定配方與 parent，再驗證；不是直接回退 exact Softmax。最終仍按 qSiLU／D1 部署 AP 選型。

舊輪最佳 Rep17＋20 的 D1／Float PWL 診斷：COCO AP 0.513738473／0.513762903，person AP 0.642786257／0.642836211，Pose AP 0.901826341／0.901781850。這些是同一已訓權重的推論替換，差異很小，不能推斷換訓練算子一定改善。新輪沒有任何正式 AP 或訓練結果。

CPU 草案的三組初始化、非零 folding、兩種 dtype 與重載已通過，可攜包測試 1 passed；這只證明轉換程式的行為，不是 SiLU 較佳的證據。撤下草案的初始權重已刪除；舊輪最佳部署與小型數據仍保留，獨立 CPU 重載通過。

## 最新狀態補充

使用者隨後重新授權訓練與少量變體比較。已重新固定五組、完成CPU與短訓，背景queue已開始；上文「GPU未啟動」描述查核當時。最新方案與進度見 [11月7日工作紀錄](../worklogs/2026-11-07-repconv-native-training.md) 及 `yolo_achitechure/stage8/protocol.json`。原先全網SiLU/exact Softmax草案仍未採用，新SiLU候選只更動兩個Rep位置。
