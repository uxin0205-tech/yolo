# 2026-08-27 COCO person AP候選與部署選擇

## 變更內容與原因

依使用者要求，最終分析不再只顯示COCO overall與單一person AP50-95，而是保留最後部署
選擇，並比較下列六個可用候選：

- 獨立Full35 Detect；
- shared `best_detect`；
- J2 `best_joint`；
- J3 `best_joint`、`best_pose`與`last`。

報告產生器新增person AP50-95、AP50、AP75、precision、recall、相對獨立Detect差值、
joint score、八項gate與checkpoint角色。另新增逐epoch person CSV、候選CSV、專用訓練
曲線與候選長條圖。`SUMMARY.json` schema升為2，明確記錄
`deployment_selection.state=pending_user_choice`；J3 `best_joint`只維持固定實驗selector
的目前預設，不代表使用者已鎖定部署權重。

更新只讀取既有正式Float／Bit-True artifacts，不重新驗證GPU、不重跑訓練、不更換或
刪除任何checkpoint。`final/full35`新增非破壞性的`--refresh-report-metadata`同步模式，
只更新分析、README、RELEASE_STATUS、程式／工作紀錄快照與SHA256 manifest。

## 驗證方式與結果

- `CUDA_VISIBLE_DEVICES=-1 ... py_compile`：report與final builder語法通過。
- 在`/tmp/full35-person-ap-report-check`以CPU隔離重建：64個正式validation epoch全部與
  `validation.csv`的person AP50-95逐值相符，8張圖完整產生。
- 新候選CSV共6列；逐epochCSV共64列。Standalone、J2/J3 best_joint使用獨立正式
  validation；best_detect、best_pose、last使用同資料／backend／evaluator的完整逐epoch
  validation，provenance已在報告分開標示。
- 新PNG格式檢查：逐epoch圖2090×1532、候選圖2180×1154，皆為8-bit RGBA PNG。
- 正式`reports/full35`與`final/full35`同步完成；同步器明確回報
  `checkpoint_files_changed=false`。最終manifest涵蓋407個受管檔案且`valid=true`；
  精確bytes與SHA256以`final/full35/MANIFEST.json`為準，避免在受管工作紀錄內形成
  manifest自我參照。
- `final/full35/verify.py`最終回傳`valid=true`、407個受管檔案；root與final內的
  `FINAL_ANALYSIS.md`、`SUMMARY.json`及person候選CSV逐SHA256一致。
- final的`best_detect`逐SHA256等於J2來源；`best_joint`、`best_pose`、`last`逐SHA256
  等於J3來源，證明本次報告同步沒有更換權重。
- 沒有啟動CUDA、GPU訓練或GPU validation。

## 遇到的困難及解法

內建`apply_patch`與`view_image`仍受
`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`影響。程式修改改由相同
`apply_patch`工具在已授權PTY中套用最小patch；沒有使用整檔shell覆寫。圖片無法透過
內建viewer顯示，但報告產生、PNG header／尺寸與機器可讀數值均已驗證。

## 未解事項或風險

- `best_detect`的person AP50-95接近獨立Detect，但global epoch0時Pose尚未適應完成，
  八項gate未通過；只能作detect-only選擇，不能當`task=both`正式答案。
- J3 `best_pose`相對`best_joint`的person AP50-95只高約0.000131，差距小於目前能由單一
  seed支持的穩健結論；不可宣稱統計顯著。
- 最後部署權重仍待使用者決定。若要在`best_joint`與`best_pose`的極小差距上作不可逆
  選擇，建議先以相同指令重驗兩者或執行seed1。
- 困難：如上；其餘無。
