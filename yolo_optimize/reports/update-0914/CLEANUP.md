# 整理盤點與保留決定

本次只整理導覽與發布副本，不做檔案刪除。依 finish-work 規則，沒有另行核准的清理 ID 就不刪；目前沒有可確定無依賴且值得刪除的本輪實驗產物。

| ID | 精確位置 | 類型／數量 | 依賴與原因 | 恢復性／風險 | 決定 |
| --- | --- | --- | --- | --- | --- |
| K01 | /home/uxin/yolo/yolo_optimize/experiments/pose_masf_training_v1/ | 實驗樹、11 份 .pt | B5 與 alpha-off 結果來源 | 權重不在 Git；刪除風險高 | 保留 |
| K02 | /home/uxin/yolo/yolo_optimize/experiments/post_masf_hardware_v1/ | 實驗樹、12 份 .pt | BinaryQK E5 上游與舊 Rep17 回合來源 | 新三組需此 parent；不可刪 | 保留 |
| K03 | /home/uxin/yolo/yolo_optimize/experiments/post_binary_rep_v1/ | 實驗樹、30 份 .pt | 三組全部 checkpoint 與部署權重 | 完整結果依賴；不可刪 | 保留 |

以上 .pt 合計 10,354,210,367 bytes，53 份均完成 SHA-256；不是建議回收容量。queue state、事件、失敗／舊方案紀錄與 cache 全保留，本次沒有磁碟清理。

本次建立的專用 Git 工作樹 `/tmp/yolo-publish-0914.r98z7m` 只在推送完成且確認乾淨後，以 git worktree remove 收束；內容可從已推送 commit 恢復，不涉及本機實驗資料。其他既有 worktree 不處理。
