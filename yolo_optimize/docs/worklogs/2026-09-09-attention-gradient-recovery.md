# 2026-09-09：Attention 梯度斷點與分段恢復

## 變更與原因

使用者指定 `yolo_attention_final/` 為優先調查來源，授權必要時回到融合前分段訓練，並要求同時看 overall／person AP 與推論。原 B100 研究及融合後研究保留；先不疊加 HOG／RepConv／MASF。

使用 diagnosing-bugs 技能建立實際來源的梯度失敗重現，再核對歷史 trainer／測試。`yolo_attention_final/src/yolo_attention/binary_basis.py` 的 fixed-PoT XNOR 將 sign 結果轉成 bool 比較與整數 reduction，因此即使 sign 有 clipped STE，score 仍 `requires_grad=False`。直接載入該專案 final checkpoint，seed0、N16 的 CPU probe 在 site10 以 AssertionError 重現。後續兩 site probe 也都確認無 Q/K score 梯度。

舊 `tests/test_pwl_final.py::test_float_pwl_surrogate_has_finite_nonzero_qk_and_bias_gradients` 使用可微 signed matmul，沒有走正式 XNOR 函式；因此該測試不能證明正式 score 路徑可微。搜尋現有 production source 未發現替換 XNOR backward 的 trainer patch。這是具體訓練路徑缺陷，但尚未證明它造成全部 AP 差距，不能說所有舊參數完全沒訓練。

新增獨立 challenger：保留精確 XNOR／popcount forward，training-only backward 採 signed-dot surrogate 與 clipped sign 梯度。不修改來源 baseline、固定係數或推論硬體契約。

## 同口徑基準結果

完整 COCO val5000、imgsz640、batch32、halfFalse、同 internal evaluator：

| 模型 | Overall AP | Person AP | Ball AP | Bat AP |
| --- | ---: | ---: | ---: | ---: |
| 官方 FP | 0.518019276 | 0.630794912 | 0.526040758 | 0.483490044 |
| A0：BinaryQK＋PWL，未加 MASF | 0.506738574 | 0.626805274 | 0.513110259 | 0.495725734 |
| B100：Bit-True | 0.503589001 | 0.624111237 | 0.516234472 | 0.469482418 |

A0−FP overall=-0.011280702、person=-0.003989638；B100−A0 overall=-0.003149573、person=-0.002694037。這是歷史模型鏈的比較，不是只改一項的重新訓練消融。

## 已驗證與待驗證

CPU challenger：兩 site 前向精確一致、Q/K 梯度非零且有限、surrogate backward 等於解析 signed-dot 梯度、160 整圖 eval／序列化／移除 adapter 一致，全部通過。

第一個 A0 QK GPU smoke 通過：真實 COCO 128 張、physical32×4、1 次 optimizer update、556 個固定 state 不變、QK gradient norm=151.823509。這不是完整 epoch 或增準結果。後續加入相同輸入 trace、逐 projection 梯度與 optimizer state 檢查，成對 smoke 通過才開始正式訓練。

新增 `infer.py` 與 `export_model.py`：推論不保留 surrogate；匯出使用 FP32 權重，避免隱式 half rounding 與訓練驗證口徑不符。尚須實際 smoke 匯出重載驗證與完整候選驗證，不因程式已寫完宣稱部署完成。

## 超參數與比較

成對 A0 原生 loss，唯一方法差異是 QK surrogate backward。Q/K LR5e-7、head LR2.5e-5、AdamW betas(.948,.999)、eps1e-8、wd0.00027、clip10、warmup1、cosine horizon10、patience4、AMP FP16、fresh optimizer／EMA。其餘參數與非 head BN statistics 固定。完整 train118287，每 epoch 925 個 macro（末批保留）；Bit-True EMA／live 的 overall、person、ball、bat 分別紀錄。保護項下降超過0.005即暫停；overall 與 person 不用加權平均掩蓋。

## 困難與解法

舊單元測試覆蓋的是 surrogate 的理想路徑，非正式 score seam；改用實際來源／checkpoint probe 與真實 GPU loss。舊 optimizer／EMA 不在 inference 檔內，因此新階段明確重設，不假裝 exact resume。無來源資料或權重變更。

## 未解事項／風險

## 正式啟動補記

成對 smoke 已通過，首 macro 的完整影像 bytes／標註／路徑 trace SHA256 均為 `1d20be5f68206800ffa5384b4aa4a6cc89fd3ba948ae1aa9a8482ab40568c199`。control 的 Q/K optimizer states=0，候選=12；候選每個 Q/K projection 參數的梯度均非零。smoke inference 匯出與 CPU160 重載一致已通過；這份 smoke 權重不作正式候選。

2026-09-09 12:07（Asia/Taipei）啟動 `a0-control-recovery-v1`，之後由既定 queue 接續 `a0-qk-recovery-v1`。queue 狀態在新研究 `artifacts/a0-pair-state.json`，每個工作有獨立 log／events／summary。每次阻塞最多600秒，正常不讀進度或額外查 GPU。退出後核對終態；安全退化與 runtime error 分別記錄，不能把安全停止說成程序崩潰。

已向使用者非阻塞詢問實際失敗影像／影片路徑，供最後同畫面推論比較；沒有該資料時只報開發集結果，不宣稱真實場景已改善。

還沒完成正式成對訓練或證明增準。Clipped STE 是有偏估計，可能仍有 saturation 或 ranking 問題；只有 AP 實測可以決定是否保留。新 trainer 保存快照但尚無 exact-resume CLI，不宣稱已支援任意中斷恢复。推論尚未做目標硬體 latency／energy 或全整數分母部署。
