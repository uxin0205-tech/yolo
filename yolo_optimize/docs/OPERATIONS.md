# 整理後的操作與恢復邊界

## 執行位置

所有實驗移到 `experiments/`。從 optimize 根層執行時，使用完整相對入口，例如 `experiments/inference/pose_branch_v1/probe.py`；若舊命令假設當前目錄就是研究根，先 `cd experiments`。維護工具改用 `tools/`，不再使用 `scripts/maintenance/`。

目前環境為 `/home/uxin/yolo/yolo_combine/.venv/bin/python`，CPU 檢查可設定 `CUDA_VISIBLE_DEVICES=''` 及 `PYTHONDONTWRITEBYTECODE=1`。這不是可攜環境保證；外部 custom source bundle、實際 Ultralytics 程式及教師權重依賴仍保留。

## 模型載入

現行 qSiLU E2 的重建類別為 `experiments/activation/bridge_v1/verify_selected.py::SelectedSource`。先以同目錄 `initialize()` 啟用已核對來源，再 `SelectedSource().build_task_models('bittrue')`，得到同源 Detect／Pose。此步是 CPU 重建，不是啟動該檔案的 main；main 會執行完整 GPU 驗證，不應誤用。

模型使用固定 SHA 與 `weights_only=True`，自訂 pickle 僅允許明確 safe globals。舊歷史腳本中仍存在先前的寬鬆載入方式，保留作方法追溯，不推薦作新工作入口；本次不藉目錄整理全面改動歷史模型語意。

`inference/*.pt` 是推論模型；同 run 的 `checkpoints/*.pt` 才可能有 optimizer、EMA、scaler 及 RNG。搬移未改寫 checkpoint 內部 metadata，若舊 metadata 內有絕對路徑，續訓前依 manifest 解析新位置。沒有驗證任意歷史 checkpoint 的 exact resume，不可直接重跑完成的 queue。

## 資料與結果

BBAT5 v1 仍是 5,964 train／683 val；COCO 保留 118,287 train／5,000 val。未重新切分或改標註。runtime View 僅更新必要的本機 YAML 路徑，不是新資料版本；cache／image links 保留。

路徑修正不產生新精度結果。驗證前先確認來源 SHA、相同 backend、資料 split、class ID、BN eps、qSiLU 重建、PWL `[-10,0]`，再比較 AP。正式推論仍用 one2one；one2many 的 bat 增益與 ball 損失需一起報告。

## 發布與保存

GitHub 的舊 optimize 發布樹由此次整理版取代，使用新 commit `5090 Done 0912`，不 force push。大型權重、資料、cache、PDF 及封存包只留本機；線上提供相對報告連結與來源清單，不把索引冒充已上傳模型。

修改前小型文字檔保存在 docs/history 的遷移原文快照；大型模型仍由既有封存保障。這些是同磁碟保存，不是異機備份。本次沒有刪除資料或宣稱回收容量。
