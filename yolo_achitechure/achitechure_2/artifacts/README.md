# 產物契約

產生的 checkpoint、影像、標籤與 runs 不納入版本控制；只有可重建資料分割所需的 `datasets/pose_grouped/split-manifest.json` 與 `patch-manifest.json` 會提交。正式執行可能建立：

- `intake/accepted.json`：含兩份 checkpoint 雜湊與 fresh-process graph report 的上游驗收。
- `candidates/<id>/`：Detect checkpoint、完整 lineage、不可獨立載入的 graph snapshot 與 transfer report。
- `datasets/pose_grouped/`：影像 symlink、標籤副本、零洩漏 split manifest 與座標 patch manifest。
- `pose/candidates/<id>/`：只有使用者明確啟用後才可能產生的 Pose26 graft、reload 證據與 lineage。
- `runs/<run-id>/`：resolved args、optimizer groups、hashes、best/last、metrics、停止原因、profile 與 peak memory。
- `selection.json`：相對 C0 的決策、條件候選 trigger 與 C_best（可為 null）。
- `quant/<id>/`：Q0/Q1/Q2 checkpoint、lineage、metrics 與量化範圍；全部只宣稱 simulation。

不可把衍生資料、checkpoint、Ultralytics source 或 runs 複製到 `src/`。
