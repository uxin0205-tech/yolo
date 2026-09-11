# 2026-09-09：E8 解凍結果、匯出推論與 E10 延長

## E8 實測結果

兩組正常完成至 E8，每 epoch 完整 train118287、925 optimizer steps，EMA／live 各自驗證完整 COCO5000。以下為 EMA BitTrue internal AP50–95：

| 權重 | overall | person |
| --- | ---: | ---: |
| A0 | 0.506738574 | 0.626805274 |
| narrow E6 | 0.507008041 | 0.626842439 |
| narrow E7 | 0.507161213 | 0.626840373 |
| narrow E8 | 0.507432701 | 0.626632484 |
| late E6 | 0.506190912 | 0.627150100 |
| late E7 | 0.507349061 | 0.626611905 |
| late E8 | 0.508153635 | 0.627520517 |

late E8 相對 A0 +0.001415061／+0.000715243，相對 narrow E8 +0.000720934／+0.000888033。相對 A0 已出現有意義的恢復訊號，但相對同預算 narrow 仍未達其中一項 +0.001 的預設門檻，尚不接受為正式 winner。對 FP overall 尚差 0.009865640；不能宣稱整個約 0.0113 缺口已補回。

late 每 epoch 含雙驗證約 594–597 秒，narrow 約 578–581 秒。本次只增加可訓練參數，沒有增加推論層或 selector；這些時間是本機訓練流程時間，不是 FPGA 或部署延遲。

## 匯出與推論驗證

新增 `verify_candidate_inference.py`，實際完成 late E8 EMA 匯出與驗證。推論權重：`studies/pre-fusion-full35-b100/artifacts/late-e8-inference-verification/late-e8-bittrue.pt`，SHA256 `b51c7504ffc7a4be298325e89f53c9856ff12d939f519cf6ff75b1fd142f3673`。

匯出使用 FP32 權重，移除訓練用 surrogate 類別與 criterion，保留原生 BinaryScore／BitTrue PWL。CPU160 匯出重載輸出完全一致；獨立載入後完整 COCO5000 重驗 overall=0.5081536353965246、person=0.6275205166051038，與 E8 訓練驗證的 AP 差值均為 0。這不是全整數 FPGA 實作或硬體效能驗證。

從原始 val 清單依序選前 8 張標註含 person 的圖片，不依模型效果挑樣；原 A0 與 late E8 都以 imgsz640、conf0.25、全80類別完成 `infer.py` 獨立推論。圖片與 confidence 標註分別位於 `artifacts/inference/a0-person-first8/` 及 `artifacts/inference/late-e8-person-first8/`。選圖清單與完整證據位於 `artifacts/late-e8-inference-verification/`。這些是既有 val 場景，不冒充独立測試集。

## 後續延長與變更

因 late E7→E8 兩項上升，narrow overall 也持續上升，保留兩組各自 E8 狀態，延長至原定 10-epoch horizon，暫不換 Backbone 拓撲、不擴大全 Backbone、不新增 MASF／HOG／RepConv。

擴充 `continue_a0.py` 完整 E8 邊界：late 恢復新增群已累積 2775 次更新的 Adam moments，既有 Q/K／head 與 EMA 為 7400 次；E5 新群 ramp 不重新啟動。新增 `run_scope_e10.py` 先實際一個 macro 驗證旧群7401／新增群2776、EMA7401、criterion updates8、成對資料 trace，再啟動正式 E9–E10。smoke 不作正式訓練來源，仍採明記限制的 epoch 邊界新 deterministic 資料流。

## 困難與解法

圖片檢視工具失敗：`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`。實際推論及圖片產生成功，但未直接目視圖片，因此不宣稱視覺品質已改善；保留對照產物供後續檢視。

late optimizer 群數較原 narrow 多，必須先重建相同群再載入狀態，不能把新群 moments 清零。新增明確 scope lineage 與 step 稽核，不把 E8 当成 E5 重新解凍。沒有修改原始模型、資料集或融合後分支。

## 未解事項與風險

E10 邊界 smoke 與正式 AP 尚待執行；尚無同預算對照通過驗收的 winner。實際使用者影片尚未提供，獨立場景、視覺評估及硬體延遲未驗證。反覆使用 val 選模仍有過擬合風險；最終候選不能僅憑微小單 seed 差異宣稱普遍有效。

## E10 續訓啟動補記

兩支變更程式 AST 檢查通過；2026-09-09 15:04（Asia/Taipei）两組 E8 邊界 smoke 均正常完成。queue 實際斷言通過：舊群 step7401、late 新群 step2776、EMA7401、criterion updates8、相同資料 trace，late moments 已恢復而非清零；證據是 `artifacts/scope-continuation-proof-v2.json`。此次 GPU 邊界驗證無新增困難。

15:04:27 啟動 `a0-scope-narrow-v2` E9–E10，接續 `a0-scope-late-v2`；monitor session17208，每次 shell wait 最多 600 秒。正式結果尚待完成事件，正常等待不讀 log 或額外查 GPU。
