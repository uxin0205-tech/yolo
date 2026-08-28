# 2026-08-26：architecture_2 固定20%初篩與 QAT-lite

## 任務與範圍

依使用者決定，為 /home/uxin/yolo/yolo_achitechure/achitechure_2/ 建立
C0-Control、C1、C2、C3 的固定20% Float 初篩流程，並在 Q0／Q1 後增加低成本
QAT-lite 相容性檢查。本次只完成規格、YAML、manifests、程式與 CPU 驗證；沒有接受
yolo_combine winner、沒有啟動真實 Float／Pose／PTQ／QAT 訓練，也沒有使用 GPU。

## 變更內容與原因

1. 將唯一正式規格升為 2.1.0，SHA256 為
   2a2d1126f6f5c63e849db08db2314e26a1ebc03efb241e04f4cb7963f0252938。
2. 新增 float-screen-20.yaml：
   - C0–C3 共用固定 seed 0 manifests。
   - fraction=1.0，避免 Ultralytics 再取排序前段。
   - patience=0、不自動淘汰、不產生正式 C_best。
   - Pose 預設關閉，仍需使用者 opt-in。
3. 新增 quant-qat-lite.yaml 與量化 stage Q2L：
   - 從同候選 Q1 calibrated lineage 建立新 optimizer。
   - 固定200 optimizer steps、前50 steps更新 observer、每50 steps驗證。
   - lr0=parent_lr0×0.1、AMP關閉、最多3 epochs、不 early-stop。
   - 只報告 Q0−Q1、Q0−Q2L 與 Q2L−Q1，不自動接受量化，也不取代正式 Q2。
4. 新增 prepare-screening-data／validate-screening-data：
   - 預設 dry-run，只有 --execute 才寫入。
   - 目的地存在時 fail closed。
   - 只建立清單、README、來源／輸出 hash 與類別統計，不複製影像或標註。
5. 建立獨立 View：
   /home/uxin/yolo/yolo_achitechure/achitechure_2/artifacts/datasets/architecture-screen-20-v1/。
   - COCO train2017：118,287 張中固定隨機抽 23,657 張。
   - COCO train-only search-val：另取互斥的 5,000 張；官方 val2017 未使用。
   - BBAT5：從 canonical search-train 5,364 張依 .rf. 前 prefix 整組抽
     1,073 張／410 groups；沿用既有 search-val 600 張。
   - BBAT5 Pose／Detect 影像 assignment 相同，但 manifests 保留各自
     /pose/images/、/detect/images/ 路徑，以解析各自 labels。
   - canonical /home/uxin/yolo/original/pose/derived/bbat5-v1/、原始資料、
     影像與標註修改：無。
6. 新增 COCO／BBAT5 三份 screening dataset YAML、schemas、catalog、CLI、結果模板與中文文件。
7. QAT-lite 是本案 fake-quant simulation，不是 Ultralytics 官方 recipe，也不是 Bit-True 或真實
   INT8 deployment。

8. 更新 `.gitignore`：只讓 `architecture-screen-20-v1` 的 README、JSON 與清單可追蹤；任何 `images/`、`labels/`、影像副檔名與權重仍維持排除。

## 驗證方式與結果

- python -m achitechure_2 validate-screening-data
  - valid=true
  - COCO train/search-val overlap：0
  - BBAT5 train/search-val group leakage：0
  - Pose／Detect assignment equal：true
  - formal validation used：false
- 最終 screening-manifest.json SHA256：
  3259465c554ba89bd7e85504342c4118ca481dca7c0f51c4f4dad8973c1a1927。
- BBAT5 Pose／Detect selected labels：
  - ball instances：539
  - bat instances：907
  - missing labels：0
  - invalid rows：0
- COCO train／search-val 都包含完整80類；manifest 另記每類 images／instances。
  COCO YOLO 清單中沒有 label file 的 211／46 張保留為背景影像，未擅自刪除。
- python -m achitechure_2 config-check：valid=true，Ultralytics 8.4.90，
  六份 training templates、八份 dataset YAML 與 Q0/Q1/Q2L/Q2 契約全部通過。
- python -m pytest -q：63 passed、1 skipped；skip 是尚未提供正式
  ARCHITECHURE_2_HANDOFF 的 integration contract，沒有誤報為通過。
- python -m ruff check .：All checks passed。
- `python -m achitechure_2 validate-pose-data`：valid=true、source_untouched=true。
- `python -m achitechure_2 validate-github-dataset`：未通過；既有完整 publication snapshot 尚未物化，缺少 `github-dataset/publication-manifest.json`。這是獨立發布工作，不影響本次 screen View；未擅自執行 `export-github-dataset --execute`。
- 全部驗證以 CUDA_VISIBLE_DEVICES='' 執行，沒有存取或占用 GPU。

## 困難與解法

1. 本機 sandbox 建立 loopback 時回報 RTM_NEWADDR: Operation not permitted，使標準隔離層
   無法啟動。解法：經受控權限在相同 workspace 執行唯讀檢查與精確 patch，未存取其他 dirty
   專案內容。
2. 第一版 manifest reader 使用 Path.resolve()，會追蹤 canonical image symlink 到共同 raw
   路徑，使 Pose／Detect 失去各自 label View。解法：立即把該未提交 View 移到
   /tmp/architecture-screen-20-v1-flawed-20260826 隔離，改為只做 absolute normalization、
   不解析 symlink，新增 symlink regression test後重新生成。最終 Pose／Detect list SHA256不同，
   類別 assignment與統計相同。

## 未解事項或風險

- `artifacts/datasets/bbat5-v1/github-dataset/` 完整發布快照仍未物化；若要交付它，需另行執行具寫入效果且資料量較大的 export，再重新驗證。

- 尚未收到 yolo_combine 的不可變 Handoff Revision，因此不能建立或訓練真實 resolved
  C0–C3，也沒有任何架構精度結論。
- 使用者尚未授權目前時段的 GPU 執行；Float20、Q1 calibration、Q2L與正式Q2都未執行。
- Pose 是否加入 Float20／完整確認仍由使用者決定；只跑 Detect 不可宣稱完整融合 C_best。
- 20%結果可能受收斂速度與抽樣噪音影響，只能初篩；接近或仍改善的候選需要共同追加 seed，
  並由 C0與finalists使用完整資料確認。
- Q2L 只是200-step W8A8 fake-quant simulation；真正部署仍缺 backend convert、operator
  coverage、custom arithmetic、accumulator／requantization與目標硬體驗證。
