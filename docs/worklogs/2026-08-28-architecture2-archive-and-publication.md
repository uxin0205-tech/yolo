# 2026-08-28：architecture_2 不採用、清理與發布

## 任務與範圍

依使用者決定，將 `/home/uxin/yolo/yolo_achitechure/achitechure_2/` 的 C1～C3 架構簡化階段標成
不採用，停止 C2／C3 完整 Float、PTQ、QAT-lite 與 C_best 接續，整理可重建產物後以
`5090 Notuse 0828` 發布。這次不啟動新訓練、不使用 GPU，也不干預獨立的 `yolo_activation` 訓練。

BBAT5 v1 的影像、標註與 split 完全未修改；正式來源仍是
`/home/uxin/yolo/original/pose/derived/bbat5-v1/`。子專案的
`artifacts/datasets/bbat5-v1/github-dataset/` 仍只是已核准的 Portable GitHub Snapshot。

## 變更內容與原因

- 新增 architecture_2 ADR 0002，記錄 C1～C3 不採用、C2／C3 下游永久關閉，並說明上游
  Full35 C0 與 `yolo_activation` 不受影響。
- 將 `configs/runs/full35-c2-c3-auto-continuation.yaml` 改為
  `closed_rejected_by_user`；保留 2026-08-27 的 authorization 只作歷史紀錄，執行閘門一律拒絕。
- 同步 schema、config-check、pytest、README、RUNBOOK、TRAINING_GUIDE、configs／results／artifacts
  導覽，避免文件仍寫成「等待決策」或暗示可重新排 queue。
- 保留 `results/full35-j3-float20-seed0/` 的原始結果與 hash，不回寫其中的 `c_best=null`、
  `pending_user_decision` 或舊 spec hash。
- 清除經使用者核准且可重建／不再使用的產物：4 個 candidate checkpoints 目錄（20 files，約
  7.55 GiB）、4 個 inference 目錄（16 files，約 1.53 GiB）、4 個 candidate logs 目錄（20 files，
  約 27.7 MiB）、Float20 runtime cache（4 files，約 54 MiB）、舊 `pose_grouped` runtime symlink
  View（6,647 symlinks）、CPU validation 暫存（10 files，約 1.31 MiB）、pytest／ruff／Python cache、
  重複 queue log與空 lock；合計回收約 9.2 GiB。
- 修正 `_float20_preflight` 的測試 seam，讓單元測試注入隔離的 source-cache 路徑；正式執行仍檢查
  真實 COCO／BBAT5 source-adjacent cache。

## 結果判讀

C0 的 COCO mAP50-95、BBAT5 Pose box mAP50-95、keypoint mAP50-95、Macro F1 分別為
0.604258、0.832275、0.981708、0.959143。C2 雖是最佳簡化候選，但相對 C0 下降分別為
0.220553、0.176987、0.089216、0.119586，全部超過 0.008 gate；C1 與 C3 也未形成可採用結果。
因此沒有執行或宣稱 PTQ、QAT、部署 INT8 或 C_best。

## 驗證方式與結果

- `CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" /home/uxin/yolo/.venv/bin/python -m pytest -q`：
  95 passed、1 skipped；skip 是需手動設定 `ARCHITECHURE_2_HANDOFF=1` 才會重做的 2.35 GB
  Full35 materialization 整合測試。
- `python -m ruff check .`：通過。
- `python -m achitechure_2 config-check`：通過；Ultralytics 8.4.90、spec 2.3.0、C0～C3 與
  所有正式 YAML 契約一致，封存項目顯示不執行。
- `python -m achitechure_2 validate-github-dataset`：通過；6,647 張唯一影像、13,294 個 image paths、
  13,294 個 label paths、0 symlink、0 forbidden files、無 test split，tree SHA256 為
  `344ab62b9fb12b6f6a65c11abf96a987ec8415e53465bdc66bafcf336300cb4f`。
- 全程使用 CPU 驗證，未啟動 architecture_2 GPU process。

## 困難與解法

- 根層 local `main` 與 `origin/main` 分歧且包含大量其他專案 dirty changes。解法：不 reset、不 pull、
  不清理原工作樹；從最新 `origin/main` 建立 `/tmp` 暫時 worktree，只帶入本次授權範圍後提交。
- 單元測試原本直接讀取全域 COCO cache；另一個正在執行的 `yolo_activation` 工作使測試受外部狀態
  影響。解法：測試注入隔離 cache 路徑，production 預設檢查維持不變；現有全域 cache 未刪除，
  避免干預其他訓練。
- 內建 patch sandbox helper 因 loopback namespace 錯誤無法運作。解法：改用標準 unified patch，
  並清除所有 `.orig`／`.rej` 暫存。
- `gh` 目前未登入，因此無法建立或更新 GitHub Issue；Git 發布改由既有 SSH push remote 完成，
  並在交付說明揭露。

## 未解事項或風險

- 本機 COCO source-adjacent `train2017.cache` 屬於其他進行中工作，本次未刪除；若未來新建
  architecture_2 revision，正式 preflight 仍會 fail closed，必須先由該工作自行整理 runtime cache。
- 已刪除的 candidate checkpoint／inference 副本只能靠原 run 設定重建；因本階段已明確不採用，
  不再保留其大型二進位副本。正式 JSON／CSV／報告與 lineage 仍完整。
