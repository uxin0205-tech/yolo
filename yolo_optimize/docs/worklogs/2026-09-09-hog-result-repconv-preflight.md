# 2026-09-09：HOG 結束、安靜監測與 RepConv 單點前置驗證

## 最新授權與監測

使用者允許依證據自主決定加訓、調整範圍及擴展驗證；「原模型可能未訓練充分」保留為假設，不視為已證明。加訓須另列 baseline recovery 對照，不把普通額外更新收益算作 HOG／MASF／BinaryQK 的創新收益。原 BEST 不覆寫。

最新監測要求取代 400 秒取樣：正常狀態只作 blocking shell wait，每次最多 600 秒，不讀 log、不查 GPU 或進度。新增 `scripts/blocking_job_monitor.py`，先用 pidfd 固定訓練及舊 supervisor 身分，再只暫停舊 supervisor（不暫停訓練），用 kernel select 等待退出；退出後恢復原 parent 以回收 child、保存真正 exit code。實際 session 15895 無正常輸出，直到收到 JOB_EXIT 才讀結果。介面等待分段收取空輸出，不是額外訓練取樣。

此監測器只可靠偵測程序退出，不自行推定 STALLED；沒有 heartbeat／明確錯誤事件時，不能宣稱能識別仍存活的卡住程序。也不把 JOB_EXIT 直接冒充 ALL_DONE；必須讀 summary 分類。原 supervisor 已恢復並完成退出，沒有留下被暫停的訓練程序。

## HOG 結果與決策

`hog-parent-ema-band-v1` 自 2026-09-08 23:35:59 至 2026-09-09 00:20:37（Asia/Taipei），完成 E1–E4，exit code 0，summary=`paused_for_analysis`。E4 EMA Ball Box／Ball Pose 相對原 BEST 下降 0.0052214146／0.0054412480，依既定安全線保存後停止。不是程式崩潰；patience stale=3，尚未到 4，因此是精度停止線而非 patience 停止。沒有完成 10 epochs。

| 共同 epoch | HOG EMA joint | Native EMA joint | HOG − Native |
| --- | ---: | ---: | ---: |
| E1 | 0.7112260044 | 0.7112260044 | 0 |
| E2 | 0.7112779685 | 0.7110669310 | +0.0002110375 |
| E3 | 0.7104267398 | 0.7110187196 | −0.0005919797 |
| E4 | 0.7095609055 | 0.7100520603 | −0.0004911548 |

原 BEST joint=0.7111747390；HOG run-local best 是 E2，只高 0.0001032295，未達預登錄 +0.001 的主要收益門檻。E2 相對原 BEST 的 Bat Box 仍低 0.0007761。E4 COCO Box 高 0.0007471，卻犧牲 Ball Box／Pose，不能只報單一 COCO 改善。沒有使用者預登錄視覺案例證據支援另一路驗收；本輪不升格新 BEST。

來源 state hash、seed、batch／macro、scope、scheduler、EMA 一致；optimizer metadata 的差異只有 HOG 新增的 aux LR／groups，排除 aux 後 base optimizer 完全一致。E1 HOG 尚未啟用，八項 EMA AP 均與 native E1 相同。只比較共同 E1–E4，不把原生 E5 與 HOG E4 冒充等預算；不宣稱單 seed 差距統計顯著。

E2 ramp 時有微小改善，E3–E4 hold 後 ball 退化，只支持「這個 HOG recipe 沒有通過」；不能唯一歸因於 μ、宣稱所有 HOG 無效，或直接認定原模型已充分訓練。這條已轉差的曲線不直接延長，後續必要加訓從原 BEST 另開，不接退化 E4。

## 接續 RepConv layer17：原因、變更與驗證

已自主接續計畫中的單點工程檢查，而非結束整個方向。新增 `scripts/analyze_hog_and_repconv.py`，讀原 BEST，CPU 測實際 256→256、3×3、stride2 的 layer17；不抽樣資料集、不跑 GPU。新增 `src/yolo_optimize/repconv.py` 的轉移及 eval-only adapter，尚未接入正式 trainer。

依 `diagnosing-bugs` 重播真實單層輸出失敗，發現原 Conv BN eps=0.001、新 RepConv 預設=1e-5；只搬 state_dict 會漏掉這項非 tensor 設定，最大輸出差 0.05355549。曾試完整複製原 Conv，但也複製 activation，破壞「分支相加後一次 activation」語意；只在 CPU 前置腳本中發生，沒有啟動此版本訓練。最終做法是轉移權重與 BN eps／momentum，保留分支無 activation；原場景重跑 max_abs=0，torch.equal 通過。

直接替換 RepConv 仍被正式 graph materializer 拒絕：原 Conv keys 缺失，conv1／conv2 keys 是 unexpected。首次 probe 誤用舊 trunk materializer 的 AttributeError 是脚本錯誤，已改用正式 validator 的 `build_graph_validation_models`，不把錯誤入口算成候選問題。

解法只作用 eval 副本：將 RepConv kernel／bias 融合，還原官方 Conv 的 keys；bias 編入等效 BN。設定 running_mean=0、running_var=1、gamma=sqrt(1+eps)、beta=fused_bias，使 eval BN 數學上等於加 bias。不可拿這個 eval-only 副本接續訓練，live 分支保留。

驗證結果：

- `tests/test_repconv.py`：2 passed，0.64 秒，涵蓋非預設 BN、零分支精確相等、非零分支融合、strict reload、來源 state 不變、training-mode 拒絕。
- 原 Full35 CPU probe exit 0：零分支 exact；RepConv 原生 fuse max_abs=1.4305e-5；原格式 adapter max_abs=1.6212e-5，均通過預設 rtol=1e-4、atol=1e-5 的逐元素比較。不能把 max_abs 寫成小於 1e-5。
- adapter 後 Detect／Pose 全部 state 嚴格 materialization 通過，兩個 task 模型的 layer17 輸出皆通過同一容差。
- 尚未完成整圖 forward parity、正式 trainer／EMA／checkpoint factory 接線、完整保存恢復與 AP；不能把此結果稱為已可正式訓練或已提升精度。

CPU 分析產物：`artifacts/direction1-20260908/hog-results-repconv-preflight.json`（未加入 adapter）、`hog-results-repconv-preflight-v2.json`（adapter 驗證通過），兩者都保留。

## 下一步與未解事項

下一步是把 RepConv 的 live 分支與 eval-only 副本接入隔離 trainer／checkpoint 契約，完成整圖與保存恢復驗證後，才啟動 layer17 matched 對照；控制組沿原 BEST，不採 HOG E4。現有 launcher 只有 native／hog，沒有可自行接續的正式 RepConv queue，因此不假裝有下一個 GPU job 正在執行，也不空等不存在的工作。

MASF 的合法 no-MASF bridge 與 BinaryQK 的硬體／梯度契約仍是後續前置，不同時開啟。第二輪與 person-only 仍排除。沒有刪除、commit、push、修改 canonical dataset 或原 final。工具權限問題沿既有明示補丁流程處理；其他困難：無。
