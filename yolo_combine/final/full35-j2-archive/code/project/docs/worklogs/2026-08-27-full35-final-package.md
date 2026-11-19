# 2026-08-27 Full35 final 交付包

## 變更內容與原因

依使用者要求新增 `final/` 交付區，目前正式內容位於 `final/full35/`。交付包不是只放一個
`.pt`，而是同時保存：

- J2 accepted run 的 `best_detect`、`best_pose`、`best_joint`、`last`，各自包含
  full-resume 與 inference-only 版本；
- 獨立 Full35-A2 Detect Float／Bit-True 與 canonical Pose P3 回退權重；
- `src/yolo_combine`、tests、scripts、Full35 variant、設計、ADR、runbook與工作紀錄快照；
- 重建 Full35 graph 所需的 `achitechure_1`、`yolo_attention`、attention YAML與A2權重；
- accepted J2 的完整訓練 CSV／JSONL／PNG，以及 final Float／Bit-True validation 的
  metrics、曲線、預測圖與 COCO `predictions.json`；
- `RELEASE_STATUS.json`、八項 gate CSV、來源／queue snapshot、`MANIFEST.json` 與
  `CHECKSUMS.sha256`。

原始 Full35 source bundle 約721MB且混有Partial75、A0及多個B/C候選；final只裁出
Full35-A2必要子集。原始說明保留為`ORIGINAL_BUNDLE_README.md`，裁切版README明確標示
實際邊界，避免把不在包內的9個候選誤當成交付內容。

資料本體沒有複製。COCO與BBAT5只保存registry／YAML snapshot；正式BBAT5入口仍固定為
`/home/uxin/yolo/original/pose/derived/bbat5-v1/`，因此沒有建立另一份split或資料版本。

目前封裝的是已通過驗收的J2 best_joint。J3仍在獨立service執行，不能因存在checkpoint
就自動覆寫final；必須八項gate全通過且Bit-True joint score高於
`0.7100382159275817`才可人工升格。封裝時J3第一個validation的八項gate均通過，但
joint score為`0.7096216737087447`，尚未超過J2，因此維持J2。

## 驗證方式與結果

- `python -m py_compile scripts/build_full35_final.py`：通過。
- builder初始以staging建立後原子rename，沒有半套final；初始產物304個受管檔案、
  2,353,861,394 bytes。後續加入CPU smoke與中文紀錄後，以最新`MANIFEST.json`為準。
- `python final/full35/verify.py`：`valid=true`，304檔SHA256與size全通過。
- `python final/full35/run.py preflight`：`ready=true`、`blockers=[]`；final內相對
  source bundle、Pose P3與baseline路徑皆能解析。
- preflight後再次執行`verify.py`：仍通過，證明正常入口不污染manifest。
- CPU從final source bundle重建graph、factory report `complete=true`；strict載入
  `weights/combined/inference/best_joint.pt`的EMA共1,238 tensors。
- CPU輸入`[1,3,64,64]`、`task=both`：同一次shared model forward回傳
  `detect`與`pose`，模型參數26,529,701。
- canonical BBAT5 val單張影像以final CLI、CPU、imgsz640執行`task=both`：完成前處理、
  strict load、shared graph、雙head後處理與JSON輸出；Detect 1筆、Pose 1筆，Pose含2個
  keypoints。總wall time 28.56秒包含factory與checkpoint load，只作功能smoke，不作
  latency benchmark。
- 沒有執行GPU validation或訓練，沒有影響背景J3 service。

## 遇到的困難及解法

第一次在final內跑preflight後，Python自動產生43個`__pycache__`檔，使嚴格manifest把它們
判為額外檔案。解法是在final入口先設`sys.dont_write_bytecode=True`，verifier與builder
明確把可重建的bytecode cache排除於受管內容，並只刪除本次驗證新生成的cache。另一次
`py_compile`產生2個根層cache，也已清除。之後重跑preflight沒有再生成cache，manifest
仍通過。

另一個問題是原source README描述9個Full35／Partial75候選，與裁切後內容不符。解法是
保留原文作血緣證據，另以裁切版README及獨立source manifest作權威內容清單。

## 未解事項或風險

- J3尚未完成決選；目前accepted權重仍是J2。
- 目前只有seed0；未執行seed1前，跨seed穩定性仍屬provisional。
- final不包含COCO／BBAT5影像與labels；搬到另一台主機時必須提供相同canonical資料並
  修正本機路徑，不能改類別、split或visibility語意。
- 尚未進行FPGA、HLS、真實latency／energy驗收。
- Ultralytics／checkpoint的AGPL與非研究商用授權仍需另外決策。
- 困難：如上；其餘無。

## 清理紀錄

只移除本次final驗證自行產生的兩個`__pycache__`目錄（共45個`.pyc`）；它們皆可由
Python重建，沒有刪除任何模型、dataset、run、metric、prediction或使用者檔案。
