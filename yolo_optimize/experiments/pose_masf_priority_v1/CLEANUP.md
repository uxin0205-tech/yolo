# 產物保留／清理盤點

本次沒有建議刪除的候選，不需要額外清理決策。下列範圍全部保留；沒有執行刪除、覆寫來源或回收 checkpoint。程式、報告、CSV 與其他稽核 JSON 也全部保留。

| ID | 確切路徑 | 類型 | 本機項目數／bytes | 依賴／保留原因 | 可恢復性 | 刪除風險 | 建議 |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| K01 | `/home/uxin/yolo/yolo_optimize/experiments/pose_masf_priority_v1/artifacts/parent` | Keep／目錄 | 1／106,802,973 | 固定 E2 比較起點，source-pin 與所有比較均引用 | runtime 可重建；其他檔案需依來源與程式重現 | 刪除會降低本次可追溯性 | 保留 |
| K02 | `/home/uxin/yolo/yolo_optimize/experiments/pose_masf_priority_v1/artifacts/comparison-v1` | Keep／目錄 | 7／107,167,677 | 候選權重、三組原始驗證、全部報告指標的來源 | runtime 可重建；其他檔案需依來源與程式重現 | 刪除會降低本次可追溯性 | 保留 |
| K03 | `/home/uxin/yolo/yolo_optimize/experiments/pose_masf_priority_v1/artifacts/datasets` | Keep／目錄 | 13,297／1,829,540 | canonical runtime View 與驗證 cache；可重建但保留以便重現 | runtime 可重建；其他檔案需依來源與程式重現 | 刪除會降低本次可追溯性 | 保留 |
| K04 | `/home/uxin/yolo/yolo_optimize/experiments/pose_masf_priority_v1/artifacts/queue-v1` | Keep／目錄 | 5／130,514 | append-only 執行事件與每個已完成工作的 log | runtime 可重建；其他檔案需依來源與程式重現 | 刪除會降低本次可追溯性 | 保留 |

bytes 統計不追蹤符號連結、不包含外部 canonical 影像容量；不以此數值推算真正可回收磁碟空間。原 Attention E2 full-resume 與原正式權重不在清理範圍內。
