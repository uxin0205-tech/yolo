# 第七階段完成、資料夾清理與可攜模型交接

## 任務與範圍

使用者要求整理目前整個研究資料夾，另設交接資料夾存放可直接使用的模型。本輪整理 `yolo_activation`，交付原始 `.pt`、必要 runtime、載入入口與完整設定。沿用刪除已確認無用產物、不封存的授權；不改其他階段模型，不啟動 Stage8。

## 變更內容與原因

Stage7 queue 實際於 2026-11-03 01:00:29（Asia/Taipei）完成：四候選各 10 輪，共 40 輪；qSiLU 已通過 activation 相對門檻，未觸發額外 5 輪。四組選定模型重載後完整十項 AP 與原結果差異全部為零。比較選定 PolyQuality；RepConv 共同 parent 仍固定 qSiLU E8。

先唯讀參考 `yolo_optimize/model/load_model.py`、其 README 與 2026-09-15 交付紀錄，採用原權重、零參數骨架、必要 runtime 與獨立程序等價驗證。來源 SHA 與差異記於 `result/stage7/handoff-reference.json`。本輪保留 D1 完整整數 exp＋reciprocal，不套用 optimize 舊 normalization。

交付包位於 `yolo_activation/result/stage7/handoff/`。qSiLU 與 PolyQuality 權重搬入此處，原候選路徑保留相容連結，沒有複製第二份權重。runtime 是來源可追溯的凍結匯出，Attention 仍在 `yolo_attention` 維護；`AffineBranchScore` 類別逐字抽取，避免匯入 checkpoint／訓練入口。其餘需要的數學與前向類別保留原碼。

舊 Full35 runtime views 約有 81 萬個資料連結，另有預覽圖、快取及未使用流程產物。先抽取小型結果與配置，保留外部仍用四份權重及相關來源，再直接清理；正式 BBAT5 v1 與 COCO 影像／標註均不更動。已實際清除 810,934 個 runtime 連結、4,844 個檔案等；67 個指定目錄與 4,595 個逐檔項目已記錄，另移除 721 個空資料夾。移除配置區塊合計 5.2484 GiB（有效內容 2.1547 GiB），整個資料夾由約 6.3 GiB 降至 1.1 GiB；48 個受保護檔案 SHA 前後一致。精確清單為 `result/stage7/cleanup-20261103.json`。

## 驗證方式與結果

原模型 CPU 160／640 各建立合成輸入參考，兩模型每個尺寸各 24 個輸出 tensors。可攜包以 `python -I -B` 新程序讀取，阻擋研究程式、資料集與歷史權重讀取；最終獨立程序驗證 exit code 0，兩模型 × 兩尺寸共 96 個輸出 tensors 逐值相同（最大絕對誤差 0），單雙任務 routing、錯誤 activation／常數拒絕檢查均通過。沒有讀取包外研究目錄，最終紀錄為 `handoff/validation/result.json`。本輪不重新跑 GPU AP。

敏感度覆核發現 `attention` 區域沒有可替換 activation，兩筆結果等於 parent；已明確標為 0 個替換位置，不把 no-op 解讀為低敏感度證據。其餘 10 筆才是有效的區域替換比較。

另完成以下交付檢查：

- 清理後原 canonical `stage7.model.load` 仍可載入交接的兩份權重；完整來源鎖未改，確認不需要已刪的 zero checkpoint。結果在 `post-cleanup-verification.json`。
- 本輪 370 個實體證據檔移到 `result/stage7/evidence/activation-20261102/`，SHA 全部相同；8 個 checkpoint／sidecar 連結已重新計算。舊 queue 與頂層 handoff 只留相容連結。
- 交付包約 214.52 MiB，378 個核心／說明檔 SHA 均吻合；包含 2 份原模型與約 3.77 MB 凍結 runtime 原碼。qSiLU SHA `2f91ec1862ad62304f1e2727e9b9ee64a27c5a46a45137f83a74dc7f14c46a69`，PolyQuality SHA `447b06043eae81cc4422c4b1a05f40d7843526f2ac3eec15860e4b454d2aac7d`。
- 五個新增／修改 Python 工具的 `ruff check --no-cache` 通過；本次相關 `git diff --check` 通過。無壞 symlink，新增主要文件的本地連結通過。
- 圖表已只使用 JSON 重新產生並目視核對；不需要 torch、GPU 或 checkpoint。現行資料夾 README、正式研究計畫、根層 README、清理入口及工作紀錄索引同步完成。
- 事後核對 `yolo_quantize` adapter 仍需要 `training/full35/activation-recipe.yaml` 與 canonical activation 程式，這些都保留。已抽取舊證據而刪除 runtime views，未刪其必要權重或配方。
- 既有 evaluator 的 val 使用 `imgsz=640, rect=True, stride=32`，不是所有 batch 都固定正方形。交接說明已區分正式 evaluator 與 640 × 640 CPU 範例，未新增形狀實驗。

本輪從未啟動 GPU 訓練或 AP。最後檢查可見另一個 `mambapose` GPU 程序；未停止、修改或占用它的 GPU 工作。

## 困難與解法

環境未安裝部分非推論必要套件（例如 SciPy），初版環境紀錄因此中止；改為如實記錄實際安裝版本與缺少的可選套件，不下載或偽造版本。獨立驗證的檔案阻擋器在清除 `/tmp` 暫存設定時，將 `dir_fd` 相對路徑誤判為研究目錄；模型檢查完成後才解除阻擋再清暫存，保留推論期間的隔離檢查。

## 未解事項或風險

原四項獨立模型精度目標仍未全部達成。單 seed、CPU 同環境移植驗證不等同新機安裝或硬體量測；Conv／QKV／P·V／projection 與 activation 仍是浮點任務運算，D1 normalization 為已選整數 reference。未啟動 RepConv、整網量化或特徵圖形狀實驗。未建立 archive／backup，未 commit／push。
