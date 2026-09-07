# 2026-09-05：V30 qSiLU八格式CPU準備

## 變更內容與原因

- 保持正在執行的poly_shift V29 Queue不變；本次只完成qSiLU後續線的CPU程式、設定與CLI準備，沒有啟動qSiLU GPU工作。
- qSiLU逐path profile由原本五格式擴充為八格式：exact W7／W6／W5／W4、optimal Fixed-SD4、Paper-TWN v2 layer-wise、TWN-v3 filter-wise及exact scaled ternary。W8不重複列入八格式，因為它是先經paired QAT鎖定的共同parent anchor。
- progressive_preparation.py加入W7／W6／W5的exact uniform候選與可選八格式coverage；預設仍維持原V19五格式，避免改寫poly_shift既有artifact。
- progressive_queue.py與progressive_qat_queue.py加入W7／W6／W5的PTQ→短QAT格式轉譯；Fixed-SD4若進入recovery，短QAT對應LS-SD4，保留Fixed與learned兩種對照角色。
- qsilu_lane_queue.py將qSiLU專用逐區plan materialize為每個stage相同的八格式矩陣，並把預期CPU coverage鎖定為148×8＝1184格；仍依backbone→neck→head逐區累積，不把poly_shift winner直接搬到qSiLU。
- 新增yolo-quantize-qsilu-lane CLI entry，並同步README、實驗README與V30 YAML的機器format ID。
- poly_shift歷史V19的148×5／740格紀錄完全保留，沒有回寫或重新命名。

## 驗證方式與結果

- qSiLU與progressive相關CPU測試共31 passed in 11.48s。
- python -m yolo_quantize.qsilu_lane_queue --help成功；沒有執行Queue。
- QSiluLanePlan.from_yaml成功解析lane id v30-qsilu-complete-quantization-lane-v1與600秒poll contract。
- Ruff對src/yolo_quantize及tests全數通過；git diff --check通過。
- 全程沒有讀取V29完整console、沒有更改訓練process、沒有使用GPU、沒有改動BBAT5 split或資料。

## 困難與解法

- 困難：apply_patch再次因環境bwrap: loopback: Failed RTM_NEWADDR失敗。
- 解法：先保留失敗證據，再改用限定workspace且每個sentinel必須恰好出現一次的精確替換；任何內容漂移就停止。
- 困難：第一次fallback命令中的Markdown反引號被shell提前解讀，Python在執行前即SyntaxError；沒有來源檔被寫入。
- 解法：改用shell單引號完整保護、並在送出前檢查腳本不含單引號，再執行同一組唯一sentinel替換。
- 困難：規劃文件原先使用fixed_sd4_optimal、paper_twn_v2_layerwise等人類標籤，與runtime的fixed_sd4、paper_twn_v2不一致。
- 解法：YAML改採真正可執行format ID；optimal scale與layer-wise語義由profile／format payload保存。

## 未解事項或風險

- V30尚未啟動，必須等poly_shift V29狀態真正成為complete後才可取得GPU。
- qSiLU先要重跑全十區W8 PTQ與paired QAT；只有通過總mAP50每項下降不超過0.015及mAP50-95每項下降不超過0.04，才可鎖定W8 parent並生成1184格CPU profile。
- 八格式代表每區搜尋空間增加；promotion仍靠真實mAP，不以CPU NRMSE直接決定。可在region gate後剪枝，但不可漏掉W7／W6／W5或三元對照。
- formal、長epoch、multi-seed、MuSGD、export及硬體量測仍維持延後，不會由本Queue自動啟動。
