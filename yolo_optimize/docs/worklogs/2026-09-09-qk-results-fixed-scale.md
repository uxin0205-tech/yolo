# 2026-09-09：QK 兩點診斷完成與固定 scale 等價修正

## 結果與決策

本次跨 job 接續 FP10 → FP22 → CPU 固定 scale 等價檢查 → 完整資料集等價驗證，沒有在單一工作結束時停止。兩個 QK 工作與最後 GPU 驗證均正常 exit 0。正常 GPU 執行期間只等待程序事件，shell 每次 wait 最多 600 秒，沒有讀進度 log 或查 GPU。

下表皆為 AP50–95，數字以 0–1 表示；FP10／FP22 是只換 score、保留 bias／PWL 的免訓練診斷，不是完整 FP teacher。

| 指標 | 原 BEST | FP10 | FP22 |
| --- | ---: | ---: | ---: |
| COCO Box | 0.498022 | 0.488301 | 0.484829 |
| COCO Person | 0.620381 | 0.617789 | 0.606442 |
| BBAT Box | 0.630036 | 0.619844 | 0.624777 |
| BBAT Pose | 0.903717 | 0.889259 | 0.899215 |
| Ball Box | 0.507437 | 0.489361 | 0.497142 |
| Ball Pose | 0.859909 | 0.831767 | 0.851881 |
| Bat Box | 0.752634 | 0.750326 | 0.752412 |
| Bat Pose | 0.947526 | 0.946751 | 0.946549 |
| Joint | 0.711175 | 0.700890 | 0.702896 |

兩項結果均不支持直接切換 score。可能原因是既有 QKV、bias、normalizer 與下游權重已適應二值 score 分布；這是合理解釋，尚不是排除其他因素後的唯一因果結論。它不能證明重新訓練 FP teacher 沒用，也不能量化原始 binary 化造成的全部損失。Detect／Pose 各有 159／45 次實際 hook 呼叫。

來源：`artifacts/direction1-20260908/qk-fp10-parent-diagnostic/summary.json` 與 `qk-fp22-parent-diagnostic/summary.json`。保留原 BEST，不啟用 baseline 禁止的 QK STE；沒有合格 challenger／teacher／梯度契約前不盲開 QAT／KD。既有 scale 消融不重跑；MASF bridge 未通過，因此沒有建立新 P-MASF，這兩項明確使用原 PSEL。

## 本輪整體判斷

| 工作 | 完成情況 | 決策 |
| --- | --- | --- |
| Native control | E1–E5，E5 安全停止 | 未接受新 recipe |
| Quarter LR | E1–E2，live 安全停止 | 不是單純降低 LR 就能修好 |
| HOG | E1–E4，安全停止；best joint 0.711278 | 相對原 BEST 增幅不足，不能稱為成功 |
| RepConv17 | 完成 5 epochs；best joint 0.710821 | 未接受，不追加 layer20 |
| MASF-off bridge | E1–E4，Ball Box 安全停止 | 保留 shared MASF，跳過 relocation |
| QK FP10／FP22 | 完整 COCO val5000＋BBAT val683 | 兩者下降，不直接替換 |
| 固定 scale 早退 | CPU 與完整 BitTrue validation 通過 | 接受本地等價實作，尚非硬體部署發布 |

目前仍選原 `final/full35/weights/combined/inference/best_joint.pt`，joint=0.7111747389752653。未解決精度回升目標，不把工程修正算成增準。長訓／MuSGD／QAT 皆為有前置條件的候選，不是無條件 queue；目前沒有足以合理延長失敗分支的趨勢，也沒有被新證據觸發的下一個 GPU job。已執行 queue 結束，不能宣稱仍在監測不存在的程序。

## 固定 scale：原本與修改後

封存來源 `source_bundle/code/yolo_attention/binary_basis.py` 的 `_coefficient` 原本是：

```text
每次 Q/K → abs/mean → dynamic coefficient
                            ↓
                   fixed 模式丟棄結果 → 回傳固定係數
```

本地 `src/yolo_optimize/fixed_scale.py` 採實例級 adapter，不修改封存來源：

```text
fixed 且非 calibration → 直接取既有固定係數
dynamic 或 calibration → 原方法（包含觀測與校準）
```

仍用原有 2 sites × 4 heads × 2 bases = 16 個固定係數。省略的是無用的動態 reduction，不是新增每圖 scale，也沒有更改 Hadamard、XNOR、PWL、BN 或 state keys。固定係數未準備好的錯誤仍保留。此 adapter 只在本地驗證入口安裝，未自動改變既有訓練／正式 export；後续模型重建仍須明確安裝，不能靠 checkpoint 自動携帶方法替換。

## 驗證與風險

- `tests/test_fixed_scale.py`：3 passed，0.59 秒；固定輸出逐值相等且禁止動態計算、dynamic 保持原行為、calibration 結果相等、缺係數與重複安裝拒絕。
- `scripts/verify_fixed_scale.py`：實際 Full35、CPU 160×160、Float／BitTrue、Detect／Pose，前後輸出與所有 state tensor 完全相等。產物 `artifacts/direction1-20260908/fixed-scale-equivalence/summary.json`。
- `scripts/validate_fixed_scale.py`：640×640，完整 COCO80 val5000＋canonical BBAT5 val683，4 個 task/site 安裝確認，8 項 BitTrue AP delta 全部精確為 0。產物 `artifacts/direction1-20260908/fixed-scale-full-validation/summary.json`。
- 原 BEST SHA256：`d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c`，與原選模值一致。

困難：監測連線資訊在上下文恢復時缺失；僅一次確認程序及終止事件，確定 FP10 已完成，沒有重複啟動。其他執行困難：無。

未驗證：目標硬體 latency／energy、INT8／PTQ export、完整資料集 Float adapter AP（已做 CPU Float 整圖等價與完整 BitTrue AP）。不宣稱 GPU 加速百分比，不宣稱精度恢復或新最佳模型。所有原權重、資料 assignment／labels、歷史 checkpoint 均保留；沒有刪除、commit、push 或啟動第二輪。
