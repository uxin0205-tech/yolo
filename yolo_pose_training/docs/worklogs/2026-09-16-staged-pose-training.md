# 2026-09-16 官方 YOLO26m Pose 分階段訓練

## 變更與原因

使用者要求從官方 yolo26m-pose.pt 重新訓練 ball／bat + 2-keypoint，依 20／50／40 epochs 的三階段 AdamW 設定，最後可嘗試 MuSGD，GPU 啟動且排隊完成即可交付。新增 Stage 4 MuSGD 候選，最多 100 epochs，patience 30，LR 0.001，960，關閉 Mosaic。各階段以前一階段 best.pt 初始化，保存 YAML、best、last、完整續訓 checkpoint。

使用官方 ultralytics 8.4.90，新增專用 .venv，唯讀引用父環境 PyTorch CUDA 依賴，避免既有 yolo_p2 修改版套件。下載官方 assets v8.4.0 的權重，SHA256 與版本寫入 artifacts/environment.json。

資料 ID 為 bbat5-v1，任務 pose，canonical YAML 為 /home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml，registry 為 /home/uxin/yolo/configs/datasets/bbat5-v1.yaml。只使用既有 formal train／val，維持 assignment 與 labels 不變；以逐檔 symlink runtime View 隔離 cache。

佇列依序訓練，每 600 秒監測，每 5 秒確認程序退出；錯誤停止相依階段。結束後統一以 960 評估四個 best.pt，依 Box／Pose mAP50–95 平均挑選候選並自動產生報告，保留每類別結果。

## 驗證方式與結果

- RTX 5090 32 GB，啟動前沒有其他訓練程序；CUDA 可用。
- 核對官方套件實际匯入路徑與 MuSGD optimizer 實作，確認 rle 參數支援。
- 官方初始模型確認為 m scale、Pose26 head、reg_max=1；训练会依数据改为 nc=2、kpt_shape=[2,3]。
- runtime View 逐檔確認連結指向 canonical；train images／labels 各 5,964、val 各 683，標註摘要記錄於 dataset-audit.json。
- 所有 Python 腳本語法檢查通過，四個階段 YAML 通過官方 get_cfg 驗證。
- checkpoint 保存 callback 於官方 strip_optimizer 之前執行，每輪保存可續訓副本。
- 已啟動 tmux session bbat5_pose_20260916；最新 GPU 與訓練狀態見下方交付檢查。

## 困難與解法

預設沙箱 bwrap 無法初始化 loopback，改以審核核准的沙箱外操作執行。一次自動審核因模型容量不足拒絕，經唯讀確認未寫入後，縮小操作範圍並成功重試，未有待確認阻礙。原父環境使用修改版 Ultralytics，改用專案內官方 wheel；新 venv 缺少 cv2，使用 .pth 唯讀引用父環境依賴解決。Matplotlib 提示 3D projection 不可用，Pose 訓練不需要 3D 圖。

## 未解事項與風險

交付只代表訓練已開始與佇列已排妥，尚未驗證完整四階段收斂、MuSGD 品質與最終比較程式的真實模型輸出。這是單一路徑的接續實驗，不是完整超參數搜尋；MuSGD 不保證優於 AdamW。formal val 用於選模，沒有獨立 test split。背景監測記錄與停止錯誤，不代表代理交付後仍在線修復。主機重新開機或程序被外部終止後需人工恢復。沒有修改或提交父儲存庫既有檔案。

## 交付檢查

Stage 1 已完成 3 個 epoch，results.csv 所有數值為有限值。best.pt、last.pt、last-resumable.pt 均已實際產生；讀取續訓 checkpoint 確認 optimizer 狀態存在，nc=2、kpt_shape=[2,3]。佇列狀態 running，後續三階段待自動執行。詳細檢查記錄在 artifacts/startup-verification.json。
