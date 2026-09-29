# 2026-09-11：全階段詳細統整、 checkpoint 封存與推論接續

## 變更內容與原因

使用者要求保存全部到目前研究、 checkpoint 及詳細報告，並允許有用的後續優化。使用 finish-work 技能，主代理盤點全部既有階段與 dirty worktree，不使用子代理、不 commit／push 、不清理或覆寫。新增總報告、封存／索引程式，更新主要入口與 plan 狀態；歴史計畫與失敗紀錄完整保留。

## 保存與驗證

主封存 `archives/research-through-20260911-v1/` 完成 5,817 檔／113,656,761,817 bytes，其中 406 個 checkpoint／模型產物 104,705,418,919 bytes；source 與 copy 逐檔 SHA256 完全相同，隨後 5,817 路徑／大小檢查通過。 qSiLU 與 headKD 來源 hash 與既有驗證紀錄相同。原始 JSON 解析／AP 非有限異常 0；保存 138 個 summary／停止入口與 9,727 筆带來源 pointer 的 AP 記錄。

沒有載入全部歷史 checkpoint，只驗證位元組保存；inference 檔不是 exact resume 。來源資料與 Python 環境 binary 不複製，固定 registry／設定保存；runtime 影像／labels 、 cache 、 symlink 排除，optimize 未找到 symlink .pt 。大型權重不進 Git 。這是同磁碟 copy/reflink，不是假稱異機容災備份。

主包後的新推論、最終報告／索引與本機實際 Ultralytics 原始碼由 addendum 獨立保存並比對 SHA，不修改第一包。因套件有本機客製，不能只靠 pip freeze 重建。

## 後續推論實驗

同 qSiLU 權重切 Pose one2many＋class-aware NMS，conf0.001 、 iou0.7 、 max_det300 、 imgsz640 、 batch16 、 workers4 、 FP32 。 CPU160 輸出／NMS 契約通過；Float／BitTrue 各完整 canonical BBAT5 val683，COCO 路徑不變、不重跑 COCO 。 checkpoint 前後 SHA 完全相同，沒有 optimizer 或訓練。 600 秒 child.wait 監測提前於約 21 秒收到 JOB_DONE 。

BitTrue 相對既有 one2one：bat 框+0.012767584 、 batPose+0.032442655；ball 框−0.010844021 、 ballPose−0.016198590 。整體 Pose+0.008122032 不能掩蓋 ball 損失，因此不全面切換。保留原 qSiLU 預設；class routing 只是下一個待驗證候選，不拼接兩份最高 AP，不忽略雙 head 成本。

## 困難與解法

sandbox bwrap 不可用，必要命令經權限審核；無來源變更或資料重切。 MASF 工作紀錄一度在 optimize 本地 docs 找不到，核對後確認在上層 yolo/docs，原相對連結有效，已更正說明而不捏造失效連結。新推論與封存均未發生執行錯誤。歷史大量「未開始」／「執行中」訊息保留為歷史，新增權威狀態入口避免混淆。

## 未解事項／風險

最終檢查：3 組新 Markdown 連結檢查、4 個新 Python 程式 AST、plan JSON 語法通過；索引產生器再次確認 5,817 個路徑／大小、406 個 checkpoint 項目及 138 個 summary 項目。測試範圍限定於本次變更，未重新執行全部歷史訓練或測試套件。

尚無全面精度恢復模型；未做多 seed 、獨立 test 、使用者影片最終驗收、板端 latency／energy 。沒有新訓練 queue；class routing 、小範圍 shared Conv 、創新區域 KD 未執行。 PTQ/QAT 延期邊界保留。總報告與程式驗證結果以本次最終檢查為準，未重跑既有正常 GPU 工作。
