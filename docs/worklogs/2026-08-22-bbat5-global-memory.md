# 2026-08-22：建立 BBAT5 固定資料集與中文紀錄全域規則

## 任務與範圍

把 BBAT5 detection／pose 的唯一預設資料集位置寫入 `/home/uxin/yolo` 的全域代理規則，禁止未經授權自行重切或替換資料集；同時建立全中文報告、變更紀錄，以及困難與解法紀錄的共用規範與 README 入口。

## 變更內容與原因

- 更新根層 `AGENTS.md`，加入固定資料集、控制變因、中文回報與必須留下工作紀錄的全域規則。
- 新增根層 `README.md`，集中連結代理規則、資料集契約與工作紀錄。
- 新增 `docs/agents/bbat5-datasets.md`，明確區分 detection 與 pose 的固定目錄、禁止事項及路徑問題處理方式。
- 新增 `docs/worklogs/README.md`，定義紀錄格式並維護索引。
- 更新目前研究專案 `yolo_achitechure/achitechure_1/README.md`，加入上述全域文件入口。
- 本次沒有複製、重新切分、抽樣、修改任何資料集影像或標註，也沒有改動既有 split。

## 驗證方式與結果

- 已確認 detection 目錄與 `data.yaml` 存在：`/home/uxin/yolo/original/pose/detect_dataset/`。
- 已確認 pose 目錄與 `data.yaml` 存在：`/home/uxin/yolo/original/pose/dataset/`。
- 已逐一解析根層 README、資料集規範、工作紀錄索引與目前研究 README 的相對連結，所有目標均存在。
- `git diff --check` 通過，新增與修改的文件也未發現行尾空白。

## 困難與解法

1. 根層原本沒有 `README.md`，無法直接加入文件連結。解法是建立精簡的全域 README，作為固定資料集政策與工作紀錄的共同入口。
2. Detection 的 `data.yaml` 內含舊使用者路徑 `/home/uxin0/yolo/original/pose/detect_dataset`。為避免在「建立全域記憶」任務中暗中改動資料集設定，本次不修改該檔，而是在資料集契約中明記正式執行前只可做最小路徑修正且必須記錄。
3. 工作樹在本次開始前已有其他未追蹤目錄與檔案。解法是把變更限制在本紀錄列出的全域文件與目前研究 README，不清理或改寫其他內容。
4. 標準補丁工具因環境的 `bwrap` loopback 建立失敗而無法寫入。解法是改用 `git apply` 套用相同的結構化補丁，完成後再檢查差異與空白錯誤。

## 未解事項或風險

- Detection 正式訓練或驗證前，仍需把設定引用導向 `/home/uxin/yolo/original/pose/detect_dataset/`；該動作只能修正路徑，不能改變資料內容或 split，並須另行記錄。
