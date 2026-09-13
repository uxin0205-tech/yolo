# 2026-09-13：Pose MASF 圖、分析與 B-only CPU 實作發布

## 範圍與決定

使用者要求先上傳目前圖與分析，commit 名稱精確為 `5090 Done 0913`。A 組 Pose-head-only 加訓取消；B 組程式與 CPU 前置已完成，但依最新優先順序，本次不啟動 GPU、不啟動等待佇列。原 Attention E2 及 scale_bias 仍保持暫停。

## 變更及原因

- 保存原 Pose／移接 Pose MASF／alpha-zero 的全量比較、成本、殘差、來源 SHA 與重建程式；這些是既有已完成結果，不是新 B 組成果。
- 更新完整 layer 0–23 架構與 one2many／one2one 梯度圖，取消 A 組加訓標籤，保留原 E2 直通參考。
- 新 `training_b.py` 實作 beta=1、alpha=0、Pose-only optimizer／EMA、8 microbatch 累積及完整回合快照。新 `queue_b.py` 提供空閒 admission 與 smoke→B5→alpha-off 分析，尚未啟動。
- 新設定 `execution-config.json` 取代原 A/B 設計；原 proposed-training.json 標示 superseded，不刪除設計血緣。
- 同步報告、目錄入口及工作紀錄；無 A 組的收益歸因限制明載。

## 驗證與結果

CPU 命令：`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/uxin/yolo/.venv/bin/python -B experiments/pose_masf_training_v1/preflight_b.py`。

結果 passed：固定 E2 輸出等價、兩種 PWL backend 重建、beta=1 的真實 Pose head 合成梯度、alpha 第一步啟動／context 第二步非零、所有固定參數／BN／EMA 不變、optimizer 命名完整與完整續訓快照 roundtrip。373 microbatch／5964 張對應 47 更新、尾端 76 張。此 CPU 檢查不是 BBAT 真實 loss 或 GPU 精度測試。

圖以 Graphviz 重建；發布檢查包含 Python 語法、JSON 有限數值、SVG／PNG 格式、Markdown 本地連結、檔案 SHA 與 Git diff --check。詳細數量保存在發布稽核 JSON；沒有重跑已完成 GPU 實驗。

## 困難及解法

主 checkout 位於舊 commit，並有大量使用者未提交內容：從最新 origin/main 建立本次專用臨時工作區，只複製明確列出的圖、報告、程式與數值證據，不 reset、不全面 stage、不覆寫無關變更。

sandbox 的 bwrap 網路初始化失敗：必要 shell 操作使用核准的 escalated 執行；檔案內容仍透過 apply_patch 編輯。圖檔檢視同受 sandbox 影響，改以唯讀 base64 傳送既有 PNG 檢視。

本機其他 mambapose 工作占用 GPU；本次不終止、不干擾，也不宣稱 B 已在 GPU 執行。

## 保存與未解事項

原始 checkpoint、CPU 測試快照、cache、資料集、log 一律保留本機，沒有刪除。發布只含必要程式、圖、設定與數值 JSON／CSV；大型權重不納入。GitHub 不含權重時，重現仍需本機固定 parent 及既有客製來源，不能將 repository clone 說成可直接無依賴推論。

未完成：真實 GPU smoke、B 5 epoch、alpha-off 新結果與後續監測。CPU 通過不代表 AMP、GPU 記憶體或精度一定通過；等待佇列只負責事件，不具有自主模型診斷能力。缺少 A5，未來 B5−E2 不能拆分 MASF 與額外訓練收益。

發布成功收據在本次 push 後另存，避免在未成功時先聲稱已上傳。
