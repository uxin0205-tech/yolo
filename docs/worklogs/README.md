# 工作紀錄索引

本目錄保存 `/home/uxin/yolo` 及其子專案的中文工作紀錄，讓程式修改、實驗、驗證、困難與解法可被追溯。

## 紀錄規則

1. 每項修改、實驗或驗證建立或更新一份 `YYYY-MM-DD-主題.md`。
2. 每份紀錄至少包含：任務與範圍、變更內容與原因、驗證方式與結果、困難與解法、未解事項或風險。
3. 沒有遇到困難時，仍在「困難與解法」明記「無」。
4. 資料集相關工作另記實際絕對路徑、任務類型、split，以及是否改動樣本或標註。
5. 新增紀錄時，同步在下方索引加入連結；所有內容使用中文，必要的程式名稱、指令、路徑與原文除外。

## 紀錄範本

```markdown
# YYYY-MM-DD：工作主題

## 任務與範圍

## 變更內容與原因

## 驗證方式與結果

## 困難與解法

## 未解事項或風險
```

## 索引

- [2026-08-22：建立 BBAT5 固定資料集與中文紀錄全域規則](2026-08-22-bbat5-global-memory.md)
- [2026-08-22：完成 architecture_2 Phase A 與 BBAT5 v1](2026-08-22-architecture-2-phase-a.md)
- [2026-08-22：以 Initial 0822 發布全域基本規則](2026-08-22-initial-0822-publish.md)
- [2026-08-22：發布 architecture_2 完整 BBAT5 GitHub Dataset（歷史，0823 已取代）](2026-08-22-architecture-2-github-dataset.md)
- [2026-08-22：統一全專案 BBAT5 資料集](2026-08-22-bbat5-dataset-unification.md)
- [2026-08-23：釐清 BBAT5 Pose／Detect Task View](2026-08-23-bbat5-task-views-clarification.md)
- [2026-08-23：發布 original 資料並清除 Git 權重](2026-08-23-original-data-publication.md)
- [2026-08-27：YOLO26m Attention final GitHub 發布](2026-08-27-yolo-attention-final-github-publication.md)
- [2026-08-26：architecture_2 固定20%初篩與 QAT-lite](2026-08-26-architecture-2-float20-qat-lite.md)
- [2026-08-27：architecture_2 C2／C3完整訓練與量化自動接續](2026-08-27-architecture2-c2-c3-auto-continuation.md)
- [2026-08-27：architecture_2 Full35 Float20 與安全 GPU 佇列](2026-08-27-architecture2-full35-float20-queue.md)
- [2026-08-27：architecture_2 Float20 成果匯出與 profiling 裝置修正](2026-08-27-architecture2-float20-profile-device-fix.md)
- [2026-08-28：architecture_2 不採用、清理與發布](2026-08-28-architecture2-archive-and-publication.md)

返回[根層 README](../../README.md)。
